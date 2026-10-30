from __future__ import annotations

import json

import pytest

from sandfleet import agent, controller
from sandfleet.agent import audit_fields
from sandfleet.audit import AUDIT_COMMAND_BYTES, AuditLog, truncate_utf8
from sandfleet.http import encode_bytes


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_audit_log_appends_context_and_fields(tmp_path):
    log = AuditLog(tmp_path / "a" / "w.jsonl", max_bytes=1 << 20, context={"pool": "p"})
    log.record({"op": "exec", "command": "ls"})
    log.record({"op": "exec", "command": "pwd"})
    records = lines(tmp_path / "a" / "w.jsonl")
    assert [record["command"] for record in records] == ["ls", "pwd"]
    assert all(
        record["pool"] == "p" and record["ts"].endswith("Z") for record in records
    )
    assert (tmp_path / "a" / "w.jsonl").stat().st_mode & 0o777 == 0o600


def test_audit_log_rotation_bounds_disk_use(tmp_path):
    path = tmp_path / "w.jsonl"
    log = AuditLog(path, max_bytes=1000, context={})
    for index in range(100):
        log.record({"n": index, "pad": "x" * 50})
    assert path.stat().st_size <= 1000
    assert (tmp_path / "w.jsonl.1").stat().st_size <= 1000
    assert not (tmp_path / "w.jsonl.2").exists()
    assert lines(path)[-1]["n"] == 99


def test_audit_write_failure_is_counted_not_raised(tmp_path, capsys):
    log = AuditLog(tmp_path / "w.jsonl", max_bytes=1000, context={})
    log.path = tmp_path / "missing-dir" / "w.jsonl"
    log.record({"op": "exec"})
    log.record({"op": "exec"})
    assert log.failures == 2
    assert "audit write" in capsys.readouterr().err


def test_truncate_utf8_keeps_whole_characters():
    text = "é" * AUDIT_COMMAND_BYTES
    truncated, cut = truncate_utf8(text)
    assert cut and len(truncated.encode()) <= AUDIT_COMMAND_BYTES
    assert truncate_utf8("ls") == ("ls", False)


def test_audit_fields_omit_contents_and_env_values():
    exec_env = audit_fields(
        "exec-env",
        {"command": "x" * 5000, "env": {"TOKEN": "secret"}, "command_id": "c1"},
        {"exit_code": 3, "cancelled": False, "timed_out": True},
    )
    assert exec_env["command_truncated"] is True
    assert len(exec_env["command"]) == AUDIT_COMMAND_BYTES
    assert exec_env["env_keys"] == ["TOKEN"]
    assert "secret" not in json.dumps(exec_env)
    assert (exec_env["exit_code"], exec_env["timed_out"]) == (3, True)
    write = audit_fields(
        "write-file", {"path": "/a", "content_b64": encode_bytes(b"hello")}, None
    )
    assert write == {"path": "/a", "bytes": 5, "offset": None}
    assert "hello" not in json.dumps(write)


@pytest.mark.parametrize(("flag", "expected"), [([], "logs/audit"), (["off"], None)])
def test_controller_audit_log_is_on_by_default(tmp_path, flag, expected):
    args = controller.build_parser().parse_args(
        [
            "--partition=p",
            "--account=a",
            "--source-root=/src",
            f"--log-dir={tmp_path}/logs",
            *(f"--sandbox-audit-log={value}" for value in flag),
        ]
    )
    state = controller.build_state(
        args,
        url="http://c:1",
        token="t",
        client_token="c",
        metrics_token="m",
        service_token_file="/t",
    )
    audit_dir = state.launcher.settings.audit_log_dir
    assert audit_dir == (None if expected is None else f"{tmp_path}/{expected}")


def test_agent_audit_file_is_per_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    args = agent.build_parser().parse_args(
        [
            "--controller-url=http://c:1",
            "--pool=p",
            "--slots=1",
            "--cpus=1",
            "--memory-mb=64",
            f"--audit-log-dir={tmp_path}",
        ]
    )
    log = agent.build_audit(args, "77-abc")
    assert log.path == tmp_path / "audit-77-abc.jsonl"
    assert log.context["job_id"] == "77" and log.context["pool"] == "p"
    assert log.max_bytes == 256 * 1024 * 1024
