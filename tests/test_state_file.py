from __future__ import annotations

import json
import os
import stat

import pytest

from sandfleet.state_file import DurableStateFile


def decode(payload):
    return payload


def test_save_retains_previous_fsynced_generation(tmp_path):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)

    store.save({"generation": 1})
    store.save({"generation": 2})

    assert json.loads(path.read_text()) == {"generation": 2}
    assert json.loads(store.backup_path.read_text()) == {"generation": 1}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_interruption_between_rotation_and_promotion_loads_backup(
    tmp_path, monkeypatch
):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    store.save({"generation": 1})
    replace = os.replace
    calls = 0

    def interrupt_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated controller death")
        replace(source, destination)

    monkeypatch.setattr("sandfleet.state_file.os.replace", interrupt_second_replace)
    with pytest.raises(OSError, match="simulated controller death"):
        store.save({"generation": 2})
    monkeypatch.setattr("sandfleet.state_file.os.replace", replace)

    recovered, recovery = store.load(decode)

    assert recovered == {"generation": 1}
    assert recovery is not None
    assert recovery.source == store.backup_path
    assert json.loads(path.read_text()) == {"generation": 1}


def test_interruption_before_first_promotion_recovers_fsynced_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    replace = os.replace

    def interrupt_replace(_source, _destination):
        raise OSError("simulated controller death")

    monkeypatch.setattr("sandfleet.state_file.os.replace", interrupt_replace)
    with pytest.raises(OSError, match="simulated controller death"):
        store.save({"generation": 1})
    monkeypatch.setattr("sandfleet.state_file.os.replace", replace)

    recovered, recovery = store.load(decode)

    assert recovered == {"generation": 1}
    assert recovery is not None
    assert recovery.source == store.temporary_path
    assert json.loads(path.read_text()) == {"generation": 1}


def test_interruption_before_snapshot_fsync_preserves_current(tmp_path, monkeypatch):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    store.save({"generation": 1})
    fsync = os.fsync

    def interrupt_file_fsync(descriptor):
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated controller death")
        fsync(descriptor)

    monkeypatch.setattr("sandfleet.state_file.os.fsync", interrupt_file_fsync)
    with pytest.raises(OSError, match="simulated controller death"):
        store.save({"generation": 2})
    monkeypatch.setattr("sandfleet.state_file.os.fsync", fsync)

    recovered, recovery = store.load(decode)

    assert recovered == {"generation": 1}
    assert recovery is None


@pytest.mark.parametrize(
    ("failed_directory_fsync", "expected_generation"), [(1, 1), (2, 2)]
)
def test_interruption_during_promotion_keeps_a_valid_generation(
    tmp_path, monkeypatch, failed_directory_fsync: int, expected_generation: int
):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    store.save({"generation": 1})
    fsync = os.fsync
    directory_fsyncs = 0

    def interrupt_directory_fsync(descriptor):
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == failed_directory_fsync:
                raise OSError("simulated controller death")
        fsync(descriptor)

    monkeypatch.setattr("sandfleet.state_file.os.fsync", interrupt_directory_fsync)
    with pytest.raises(OSError, match="simulated controller death"):
        store.save({"generation": 2})
    monkeypatch.setattr("sandfleet.state_file.os.fsync", fsync)

    recovered, _recovery = store.load(decode)

    assert recovered == {"generation": expected_generation}


@pytest.mark.parametrize("corruption", [b"", b"x" * 1_048_576])
def test_truncated_primary_recovers_and_quarantines(tmp_path, corruption: bytes):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    store.save({"generation": 1, "blob": "a" * 1_100_000})
    store.save({"generation": 2, "blob": "b" * 1_100_000})
    path.write_bytes(corruption)

    recovered, recovery = store.load(decode)

    assert recovered == {"generation": 1, "blob": "a" * 1_100_000}
    assert recovery is not None
    assert recovery.quarantined is not None
    assert recovery.quarantined.read_bytes() == corruption
    assert json.loads(path.read_text())["generation"] == 1


def test_no_valid_generation_fails_without_starting_empty(tmp_path):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    path.write_text("{")
    store.backup_path.write_text("")

    with pytest.raises(RuntimeError, match="No valid Sandfleet state generation"):
        store.load(decode)

    assert path.read_text() == "{"
    assert store.backup_path.read_text() == ""


def test_save_fsyncs_snapshot_and_directory(tmp_path, monkeypatch):
    path = tmp_path / "controller-state.json"
    store = DurableStateFile[dict](path)
    fsynced: list[str] = []

    def record_fsync(descriptor):
        mode = os.fstat(descriptor).st_mode
        fsynced.append("directory" if stat.S_ISDIR(mode) else "file")

    monkeypatch.setattr("sandfleet.state_file.os.fsync", record_fsync)

    store.save({"generation": 1})
    store.save({"generation": 2})

    assert fsynced == ["file", "directory", "file", "directory", "directory"]
