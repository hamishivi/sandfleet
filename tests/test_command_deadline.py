"""Deadline ownership, cleanup failures, and foreground/output-drain ordering."""

import os
from pathlib import Path
import signal
import threading
from types import SimpleNamespace

import pytest

from sandfleet import step_worker as sw
from sandfleet.errors import SandfleetError


class ManualTimer:
    latest = None

    def __init__(self, interval, function):
        self.function = function
        self.daemon = False
        self.joined = False
        ManualTimer.latest = self

    def start(self):
        pass

    def cancel(self):
        pass

    def join(self):
        self.joined = True


class DeadlineInstance:
    timeout = 10
    uses_pid_namespace = True

    def __init__(self, completion_first=False, error=None):
        self.completion_first = completion_first
        self.error = error
        self.kills = 0

    def cancel_command_group(self, command_id):
        self.kills += 1
        if isinstance(self.error, Exception):
            raise self.error
        return (False, self.error) if self.error else (True, None)

    def run_command(self, command, timeout, *, on_start, on_complete, **kwargs):
        on_start(SimpleNamespace(pid=4321))
        if self.completion_first:
            on_complete()
        ManualTimer.latest.function()
        on_complete()
        return {"stdout": "", "stderr": "", "exit_code": 0}


@pytest.mark.parametrize("completion_first", [True, False])
def test_deadline_ownership_does_not_depend_on_exit_code(monkeypatch, completion_first):
    monkeypatch.setattr(sw.threading, "Timer", ManualTimer)
    session = sw.WorkerSession()
    instance = DeadlineInstance(completion_first)
    session.instance = instance
    result = session._run_command({"command": "x", "command_id": "one"})
    assert result["timed_out"] is not completion_first
    assert instance.kills == (0 if completion_first else 1)
    assert ManualTimer.latest.joined


@pytest.mark.parametrize("error", ["SURVIVOR", OSError("cancel exec failed")])
def test_deadline_cleanup_failure_is_visible(monkeypatch, error):
    monkeypatch.setattr(sw.threading, "Timer", ManualTimer)
    session = sw.WorkerSession()
    session.instance = DeadlineInstance(error=error)
    expected = OSError if isinstance(error, OSError) else SandfleetError
    with pytest.raises(expected, match=str(error)):
        session._run_command({"command": "x", "command_id": "failed"})
    assert "failed" not in session._active
    assert "failed" in session._finished


def test_result_waits_for_deadline_cleanup():
    entered = threading.Event()
    release = threading.Event()
    published = threading.Event()

    class BlockingInstance(DeadlineInstance):
        timeout = 0.01

        def cancel_command_group(self, command_id):
            entered.set()
            assert release.wait(5)
            return True, None

        def run_command(self, command, timeout, *, on_start, on_complete, **kwargs):
            on_start(SimpleNamespace(pid=4321))
            assert entered.wait(5)
            on_complete()
            return {"stdout": "", "stderr": "", "exit_code": 0}

    session = sw.WorkerSession()
    session.instance = BlockingInstance()
    result = {}

    def run():
        result.update(session._run_command({"command": "x", "command_id": "blocked"}))
        published.set()

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(5)
        assert not published.wait(0.1)
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert result["timed_out"] is True


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="Linux process groups")
def test_completed_foreground_with_inherited_pipe_preserves_child(tmp_path):
    """Real processes and full exec/capture path; only Apptainer transport is local."""
    shim = tmp_path / "apptainer-shim"
    # Drops `exec`, its flags and `instance://<name>`, then runs the command locally.
    shim.write_text(
        '#!/bin/sh\nuntil [ "${1#instance://}" != "$1" ]; do shift; done\nshift\nexec "$@"\n'
    )
    shim.chmod(0o700)

    class LocalInstance(sw.ApptainerInstance):
        def __init__(self):
            self.name = "deadline-pipe-regression"
            self.start_flags = ["--containall"]
            self.timeout = 1
            self.apptainer = str(shim)
            self.network = None
            self._instance_stopped = False
            self._drain_threads = []
            self.pids_limit = 0

        def env(self, *, pwd=None):
            return dict(os.environ)

    session = sw.WorkerSession()
    session.instance = LocalInstance()
    child_file = tmp_path / "child"
    child = None
    try:
        # Child retains stdout; capture drains for five seconds AFTER shell exit.
        result = session._run_command(
            {
                "command": f"sleep 30 & echo $! > {child_file}; true",
                "command_id": "inherited-pipe",
                "timeout": 1,
            }
        )
        child = int(child_file.read_text())
        assert result["exit_code"] == 0
        assert result["timed_out"] is False
        state = Path(f"/proc/{child}/stat").read_text().rsplit(") ", 1)[1].split()[0]
        assert state not in {"Z", "X"}
    finally:
        if child is None and child_file.exists():
            child = int(child_file.read_text())
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass
        session.instance.join_drain_threads()
