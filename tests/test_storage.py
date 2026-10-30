import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sandfleet import storage
from sandfleet.step_backend import SlurmStepSandbox
from sandfleet.step_worker import ApptainerInstance, WorkerSession


@pytest.mark.parametrize("size", [True, 0, 63, 1025, 1.5, "512"])
def test_disk_request_is_bounded_and_opt_in(size):
    with pytest.raises(ValueError):
        storage.validate_storage(size, 1024)


def test_legacy_storage_remains_optional():
    storage.validate_storage(None, 0)
    storage.validate_storage(1024, 1024)
    with pytest.raises(ValueError):
        storage.validate_storage(1024, 0)


def test_sif_digest_must_match(tmp_path):
    path = tmp_path / "image.sif"
    path.write_bytes(b"immutable test image")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert storage.image_digest(str(path), expected) == expected
    with pytest.raises(ValueError, match="mismatch"):
        storage.image_digest(str(path), "a" * 64)


def test_mounts_cannot_escape_overlay():
    root = "1 0 0:1 / / rw - fuse.fuse-overlayfs overlay rw\n"
    storage.overlay_mounts(root, ["/", "/tmp", "/root", "/workspace"])
    with pytest.raises(RuntimeError, match="escapes"):
        storage.overlay_mounts(root + "2 1 0:2 / /tmp rw - tmpfs tmpfs rw", ["/tmp"])


@pytest.mark.parametrize(
    "filesystems",
    [[], ["tmpfs"], ["lustre"], ["nfs"], ["xfs", "tmpfs"], ["xfs", "nfs"]],
)
def test_overlay_rejects_ram_and_shared_disks(monkeypatch, tmp_path, filesystems):
    overlay = storage.DiskOverlay(str(tmp_path), 64, "apptainer")
    checked = Mock(
        return_value=json.dumps({"filesystems": [{"fstype": fs} for fs in filesystems]})
    )
    monkeypatch.setattr(storage, "checked", checked)
    with pytest.raises(ValueError, match="local"):
        overlay.create()
    checked.assert_called_once()
    assert overlay.directory is None


def test_overlay_accepts_stacked_local_mounts(monkeypatch, tmp_path):
    overlay = storage.DiskOverlay(str(tmp_path), 64, "apptainer")
    mount = {"target": str(tmp_path), "source": "/dev/local", "fstype": "xfs"}

    def checked(argv, **_kwargs):
        if argv[0] == "findmnt":
            return json.dumps({"filesystems": [mount, mount]})
        if argv[0] == "fallocate":
            assert argv == [
                "fallocate",
                "--keep-size",
                "--length",
                str(64 * storage.MIB),
                "--",
                str(overlay.image),
            ]
            return ""
        assert argv[:4] == ["apptainer", "overlay", "create", "--fakeroot"]
        # Exercise the full allocation check without invoking Apptainer.
        with Path(argv[-1]).open("wb") as output:
            for _ in range(64):
                output.write(b"x" * storage.MIB)
        return ""

    monkeypatch.setattr(storage, "checked", checked)
    image = overlay.create()
    assert image.stat().st_size == 64 * storage.MIB
    assert image.parent.parent == tmp_path
    overlay.remove()
    assert not image.exists()


def sparse_overlay(monkeypatch, tmp_path, *, size_delta=0, allocate=None):
    overlay = storage.DiskOverlay(str(tmp_path), 64, "apptainer")
    calls = []

    def checked(argv, **_kwargs):
        calls.append(argv)
        if argv[0] == "findmnt":
            return '{"filesystems": [{"fstype": "xfs"}]}'
        if argv[0] == "apptainer":
            with Path(argv[-1]).open("wb") as image:
                image.write(b"formatted filesystem sentinel")
                image.truncate(64 * storage.MIB + size_delta)
            return ""
        assert argv == [
            "fallocate",
            "--keep-size",
            "--length",
            str(64 * storage.MIB),
            "--",
            str(overlay.image),
        ]
        if allocate is not None:
            allocate(overlay.image)
        return ""

    monkeypatch.setattr(storage, "checked", checked)
    return overlay, calls


def test_overlay_allocates_sparse_image_without_overwriting_contents(
    monkeypatch, tmp_path
):
    def allocate(path):
        # Model fallocate's content-preserving behavior on non-Linux test hosts.
        with path.open("r+b") as image:
            for offset in range(0, 64 * storage.MIB, storage.MIB):
                image.seek(offset)
                contents = image.read(storage.MIB)
                image.seek(offset)
                image.write(contents)

    overlay, calls = sparse_overlay(monkeypatch, tmp_path, allocate=allocate)
    fsync = Mock(wraps=storage.os.fsync)
    monkeypatch.setattr(storage.os, "fsync", fsync)
    image = overlay.create()
    assert [call[0] for call in calls] == ["findmnt", "apptainer", "fallocate"]
    assert calls[1] == [
        "apptainer",
        "overlay",
        "create",
        "--fakeroot",
        "--sparse",
        "--size",
        "64",
        str(image),
    ]
    fsync.assert_called_once()
    assert image.stat().st_blocks * 512 >= image.stat().st_size
    with image.open("rb") as source:
        assert source.read(29) == b"formatted filesystem sentinel"
        source.seek(-32, 2)
        assert source.read() == bytes(32)
    overlay.remove()


def test_overlay_formatter_failure_does_not_attempt_allocation(monkeypatch, tmp_path):
    overlay = storage.DiskOverlay(str(tmp_path), 64, "apptainer")
    failure = RuntimeError("sparse formatter failed")

    def checked(argv, **_kwargs):
        if argv[0] == "findmnt":
            return '{"filesystems": [{"fstype": "xfs"}]}'
        assert argv[:5] == [
            "apptainer",
            "overlay",
            "create",
            "--fakeroot",
            "--sparse",
        ]
        Path(argv[-1]).write_bytes(b"incomplete format")
        raise failure

    monkeypatch.setattr(storage, "checked", checked)
    with pytest.raises(RuntimeError, match="sparse formatter failed") as caught:
        overlay.create()
    assert caught.value is failure
    assert overlay.image is None and overlay.directory is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("size_delta", [-1, 1])
def test_overlay_rejects_wrong_size_before_allocation(
    monkeypatch, tmp_path, size_delta
):
    overlay, calls = sparse_overlay(monkeypatch, tmp_path, size_delta=size_delta)
    with pytest.raises(RuntimeError, match="incorrect nominal size"):
        overlay.create()
    assert [call[0] for call in calls] == ["findmnt", "apptainer"]
    assert overlay.image is None and overlay.directory is None
    assert not list(tmp_path.iterdir())


def test_overlay_rejects_remaining_holes_after_allocator_success(monkeypatch, tmp_path):
    overlay, calls = sparse_overlay(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="not fully allocated"):
        overlay.create()
    assert calls[-1][0] == "fallocate"
    assert overlay.image is None and overlay.directory is None
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_overlay_allocation_failure_preserves_error(
    monkeypatch, tmp_path, cleanup_fails
):
    failure = RuntimeError("allocation failed: no space")

    def allocate(_path):
        raise failure

    overlay, _calls = sparse_overlay(monkeypatch, tmp_path, allocate=allocate)
    if cleanup_fails:
        monkeypatch.setattr(
            overlay, "remove", Mock(side_effect=OSError("cleanup busy"))
        )
    with pytest.raises(RuntimeError, match="allocation failed") as caught:
        overlay.create()
    assert caught.value is failure
    if cleanup_fails:
        assert overlay.image.exists()
        assert any("cleanup busy" in note for note in failure.__notes__)
    else:
        assert overlay.image is None and overlay.directory is None
        assert not list(tmp_path.iterdir())


def test_overlay_requires_filesystem_type(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "checked", Mock(return_value='{"filesystems": [{}]}'))
    overlay = storage.DiskOverlay(str(tmp_path), 64, "apptainer")
    with pytest.raises(KeyError, match="fstype"):
        overlay.create()
    assert overlay.directory is None


def test_disk_env_drops_inherited_tokens_and_mounts(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDFLEET_SERVICE_TOKEN", "do-not-inherit")
    monkeypatch.setenv("APPTAINER_BINDPATH", "/private:/private")
    overlay = storage.DiskOverlay(str(tmp_path), 64, "apptainer")
    assert "SANDFLEET_SERVICE_TOKEN" not in overlay.env()
    assert "APPTAINER_BINDPATH" not in overlay.env()


def disk_instance(monkeypatch, tmp_path):
    monkeypatch.setattr("sandfleet.step_worker.image_digest", lambda *_a: "a" * 64)
    instance = ApptainerInstance(
        image="/image.sif",
        timeout=60,
        pwd="/workspace",
        cache_dir=None,
        tmp_dir=None,
        extra_start_flags=[],
        apptainer_binary="apptainer",
        storage_mb=64,
        image_sha256="a" * 64,
        storage_root=str(tmp_path),
    )
    instance.overlay = Mock()
    instance.overlay.env.return_value = {}
    instance.name = "only-this-instance"
    return instance


@pytest.mark.parametrize("stderr", [b"busy", b"no instance found"])
def test_stop_failure_retains_identity_and_overlay(monkeypatch, tmp_path, stderr):
    instance = disk_instance(monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance, "exec_instance_list", lambda **_kw: [{"instance": instance.name}]
    )
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=1, stderr=stderr),
    )
    with pytest.raises(RuntimeError, match="stop failed"):
        instance.close()
    instance.overlay.remove.assert_not_called()
    assert instance.name == "only-this-instance"


def test_instance_disappears_between_list_and_stop(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    listing = Mock(
        side_effect=[
            [{"instance": instance.name}],
            [{"instance": "someone-else"}],
        ]
    )
    monkeypatch.setattr(instance, "exec_instance_list", listing)
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        Mock(return_value=SimpleNamespace(returncode=1, stderr=b"no instance found")),
    )
    instance.close()
    instance.overlay.remove.assert_called_once()
    assert instance.name is None
    assert listing.call_count == 2
    assert listing.call_args.kwargs == {"timeout": 2}


@pytest.mark.parametrize("confirming", [False, True])
@pytest.mark.parametrize(
    "instances", [{}, "", None, [None], [{}], [{"instance": 1}], [{"instance": ""}]]
)
def test_malformed_instance_listing_retains_disk(
    monkeypatch, tmp_path, instances, confirming
):
    instance = disk_instance(monkeypatch, tmp_path)
    responses = []
    if confirming:
        responses.extend(
            [
                SimpleNamespace(
                    stdout=json.dumps({"instances": [{"instance": instance.name}]})
                ),
                SimpleNamespace(returncode=1, stderr=b"no instance found"),
            ]
        )
    responses.append(SimpleNamespace(stdout=json.dumps({"instances": instances})))
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        Mock(side_effect=responses),
    )
    with pytest.raises(
        RuntimeError,
        match="stop failed" if confirming else "Malformed Apptainer instance listing",
    ):
        instance.close()
    instance.overlay.remove.assert_not_called()
    assert instance.name == "only-this-instance"


@pytest.mark.parametrize("listing_result", [RuntimeError("list failed"), [{}]])
def test_failed_stop_cannot_assume_absence(monkeypatch, tmp_path, listing_result):
    instance = disk_instance(monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance,
        "exec_instance_list",
        Mock(
            side_effect=[
                [{"instance": instance.name}],
                listing_result,
            ]
        ),
    )
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        Mock(return_value=SimpleNamespace(returncode=1, stderr=b"no instance found")),
    )
    with pytest.raises(RuntimeError, match="stop failed") as caught:
        instance.close()
    assert "Could not verify instance absence" in caught.value.__notes__[0]
    instance.overlay.remove.assert_not_called()
    assert instance.name == "only-this-instance"
    assert not instance._instance_stopped


def test_still_listed_stop_retains_overlay(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance, "exec_instance_list", lambda **_kw: [{"instance": instance.name}]
    )
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stderr=b""),
    )
    with pytest.raises(RuntimeError, match="still listed"):
        instance.close()
    instance.overlay.remove.assert_not_called()


def test_stop_waits_for_instance_disappearance(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    listings = iter([[{"instance": instance.name}], []])
    monkeypatch.setattr(instance, "exec_instance_list", lambda **_kw: next(listings))
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stderr=b""),
    )
    instance._stop_instance()
    instance.overlay.remove.assert_not_called()


def test_absent_attempted_instance_cleans_only_owned_disk(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance, "exec_instance_list", lambda: [{"instance": "someone-else"}]
    )
    stop = Mock()
    monkeypatch.setattr(instance, "_stop_instance", stop)
    instance.close()
    stop.assert_not_called()
    instance.overlay.remove.assert_called_once()


def test_failed_start_session_retains_instance_for_stop(monkeypatch, tmp_path):
    monkeypatch.setattr("sandfleet.step_worker.current_cgroup", lambda **_kwargs: {})
    instance = Mock()
    instance.start.side_effect = RuntimeError("partial start")
    monkeypatch.setattr(
        "sandfleet.step_worker.ApptainerInstance", Mock(return_value=instance)
    )
    session = WorkerSession(str(tmp_path), 1024)
    with pytest.raises(RuntimeError, match="partial start"):
        session._start({"config": {"storage_mb": 1024}})
    assert session.instance is instance
    session.close()
    instance.close.assert_called_once()


def test_populated_disk_cgroup_retains_cleanup_state(tmp_path):
    group = tmp_path / "group"
    group.mkdir()
    (group / "cgroup.events").write_text("populated 1\n")
    worker = tmp_path / "owned-worker"
    worker.mkdir()
    backend = SlurmStepSandbox(cpus=1, memory_mb=1024, max_storage_mb=1024)
    backend.worker_dir = str(worker)
    backend.metadata = {"path": str(group)}
    with pytest.raises(RuntimeError, match="still populated"):
        backend.discard()
    assert Path(backend.worker_dir).exists()
    (group / "cgroup.events").write_text("populated 0\n")
    backend.discard()
    assert backend.worker_dir is None and not worker.exists()


def test_vanished_step_directory_is_unhealthy_and_discardable(tmp_path):
    # Slurm removes the job's /tmp (and each step directory) on teardown.
    worker = tmp_path / "owned-worker"
    worker.mkdir()
    backend = SlurmStepSandbox(cpus=1, memory_mb=1024, max_storage_mb=1024)
    backend.process = Mock()
    backend.process.poll.return_value = None
    backend.connection = Mock()
    backend.worker_dir = str(worker)
    assert backend.healthy and not backend.storage_lost
    worker.rmdir()
    assert backend.storage_lost and not backend.healthy
    backend.process.poll.return_value = 0
    backend.discard()
    assert backend.worker_dir is None


def test_abort_does_not_wait_for_rpc_lock():
    backend = SlurmStepSandbox(cpus=1, memory_mb=1024)
    backend.process = Mock()
    backend.process.poll.return_value = None
    with backend._lock:
        backend.abort()
    assert backend._interrupted.is_set()
    backend.process.terminate.assert_called_once()


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ("", KeyError),
        ("frozen 0\n", KeyError),
        ("populated invalid\n", RuntimeError),
        ("malformed\n", ValueError),
    ],
)
def test_malformed_disk_cgroup_events_never_allow_storage_removal(
    tmp_path, contents, error
):
    group = tmp_path / "group"
    group.mkdir()
    (group / "cgroup.events").write_text(contents)
    worker = tmp_path / "owned-worker"
    worker.mkdir()
    backend = SlurmStepSandbox(cpus=1, memory_mb=1024, max_storage_mb=1024)
    backend.worker_dir = str(worker)
    backend.metadata = {"path": str(group)}
    with pytest.raises(error):
        backend.discard()
    assert backend.worker_dir == str(worker)
    assert backend.metadata == {"path": str(group)}
    assert worker.exists()


def test_remote_file_errors_do_not_discard_healthy_worker(monkeypatch):
    backend = SlurmStepSandbox(cpus=1, memory_mb=1024)
    backend.connection = Mock()
    backend.process = Mock()
    monkeypatch.setattr(
        backend,
        "_receive",
        lambda *_a: {
            "ok": False,
            "error_type": "FileNotFoundError",
            "message": "absent",
            "traceback": "",
        },
    )
    discard = Mock()
    monkeypatch.setattr(backend, "_terminate_worker", discard)
    with pytest.raises(FileNotFoundError):
        backend._rpc("read_file", response_timeout=30)
    discard.assert_not_called()
