"""End-to-end control-channel validation for command-scoped cancellation.

Drives the real ``serve()`` transport — a real AF_UNIX main socket, a real
second control socket, and the real daemon control thread — with a fake instance
that runs commands as real local processes. This exercises the concurrency the
feature exists for: a cancel delivered over the control channel while the main
channel is blocked inside a long-running exec.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import threading
import time
from multiprocessing.connection import Client

import pytest

from sandfleet import step_worker


class FakeInstance:
    """Runs commands as real local processes so the worker's real killpg path is
    exercised through the socket transport."""

    def __init__(self, **_config) -> None:
        self.name = "fake-instance"
        self.storage_status = None
        self.network_metadata: dict = {}
        self.readonly_mounts_status: list = []

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def run_command(
        self, command, timeout, *, env=None, on_start=None, command_id=None
    ):
        process = subprocess.Popen(  # noqa: S602,S603
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        if on_start is not None:
            on_start(process)
        stdout, stderr = process.communicate()
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": process.returncode,
            "output_truncated": False,
        }


@pytest.fixture
def worker(tmp_path, monkeypatch):
    """A live worker served over real sockets with a fake instance."""
    monkeypatch.setattr(step_worker, "ApptainerInstance", FakeInstance)
    monkeypatch.setattr(step_worker, "current_cgroup", lambda **_kwargs: {})
    monkeypatch.setattr(step_worker, "memory_events", lambda: {"oom": 0, "oom_kill": 0})
    socket_path = str(tmp_path / "worker.sock")
    control_path = str(tmp_path / "control.sock")
    authkey = b"k" * 32

    thread = threading.Thread(
        target=step_worker.serve,
        args=(socket_path, authkey),
        kwargs={"control_socket_path": control_path},
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (
        os.path.exists(socket_path) and os.path.exists(control_path)
    ):
        time.sleep(0.01)
    main = Client(socket_path, family="AF_UNIX", authkey=authkey)
    control = Client(control_path, family="AF_UNIX", authkey=authkey)
    main.send({"operation": "start", "config": {"image": "x", "timeout": 60}})
    assert main.recv()["ok"] is True

    try:
        yield main, control
    finally:
        with contextlib.suppress(Exception):
            main.send({"operation": "shutdown"})
            main.recv()
        main.close()
        control.close()
        thread.join(timeout=5)


def _run_async(main, request):
    outcome: dict = {}

    def call():
        main.send(request)
        outcome["response"] = main.recv()

    thread = threading.Thread(target=call)
    thread.start()
    return thread, outcome


def test_cancel_over_control_channel_interrupts_a_blocked_exec(worker, tmp_path):
    main, control = worker
    marker = tmp_path / "started"
    command = f"echo go > {marker}; sleep 30"
    started = time.monotonic()
    thread, outcome = _run_async(
        main, {"operation": "run_command", "command": command, "command_id": "job-1"}
    )
    # Wait until the process is really running so the cancel exercises the
    # running path (not the before-start marker path).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.01)
    assert marker.exists()

    control.send({"operation": "cancel", "command_id": "job-1"})
    cancel_response = control.recv()
    assert cancel_response["ok"] is True
    assert cancel_response["result"] == {
        "cancelled": True,
        "interrupted": True,
        "state": "running",
    }

    thread.join(timeout=15)
    assert not thread.is_alive()
    assert time.monotonic() - started < 20  # nowhere near the 30s sleep
    result = outcome["response"]
    assert result["ok"] is True
    assert result["result"]["cancelled"] is True
    assert result["result"]["exit_code"] == step_worker.CANCELLED_SIGNALLED_EXIT


def test_control_cancel_kills_whole_group_including_term_ignoring_child(
    worker, tmp_path, monkeypatch
):
    monkeypatch.setattr(step_worker, "CANCEL_TERM_GRACE_SECONDS", 0.5)
    main, control = worker
    child_file = tmp_path / "child.pid"
    started = tmp_path / "started"
    command = (
        "( trap '' TERM; while :; do sleep 1; done ) >/dev/null 2>&1 & "
        f"echo $! > {child_file}; echo go > {started}; exec sleep 300"
    )
    thread, _ = _run_async(
        main, {"operation": "run_command", "command": command, "command_id": "grp"}
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not child_file.exists():
        time.sleep(0.01)
    child_pid = int(child_file.read_text().strip())
    assert os.kill(child_pid, 0) is None  # alive before cancel

    control.send({"operation": "cancel", "command_id": "grp"})
    assert control.recv()["result"]["interrupted"] is True
    thread.join(timeout=15)

    # The SIGTERM-ignoring child must be reaped, not leaked.
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_control_channel_works_while_main_is_blocked(worker, tmp_path):
    # The whole point: a control status/cancel is answered even though the main
    # channel is mid-exec and cannot service another request.
    main, control = worker
    marker = tmp_path / "started"
    thread, outcome = _run_async(
        main,
        {
            "operation": "run_command",
            "command": f"echo go > {marker}; sleep 20",
            "command_id": "busy",
        },
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not marker.exists():
        time.sleep(0.01)
    assert marker.exists()

    # Main is now blocked in exec; the control channel still responds promptly.
    control_started = time.monotonic()
    control.send({"operation": "status"})
    status = control.recv()
    assert time.monotonic() - control_started < 5
    assert status["ok"] is True
    assert status["result"]["instance_name"] == "fake-instance"

    control.send({"operation": "cancel", "command_id": "busy"})
    assert control.recv()["result"]["cancelled"] is True
    thread.join(timeout=15)
    assert outcome["response"]["result"]["cancelled"] is True


def test_unknown_control_operation_is_reported_not_fatal(worker):
    main, control = worker
    control.send({"operation": "bogus"})
    response = control.recv()
    assert response["ok"] is False
    assert "bogus" in response["message"]
    # The channel survives the error and still serves a valid request.
    control.send({"operation": "status"})
    assert control.recv()["ok"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
