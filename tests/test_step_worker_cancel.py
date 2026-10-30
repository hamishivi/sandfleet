"""Command-scoped cancellation primitive in the step worker.

These exercise the concurrency-critical core (registry + interruptible exec +
before-start / after-completion / failed-interruption / whole-group-kill
semantics) in-process, with a fake instance that runs real local processes so
the worker's real killpg path is exercised. The socket/agent/client wiring is
covered separately.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time

import pytest

from sandfleet import step_worker
from sandfleet.step_worker import WorkerSession


class FakeInstance:
    """Stands in for ApptainerInstance: runs the command as a real local process
    so the worker's real killpg-based cancellation path is exercised."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_command(
        self, command, timeout, *, env=None, on_start=None, command_id=None
    ):
        self.calls.append(command)
        if on_start is None:
            return {"stdout": "legacy", "stderr": "", "exit_code": 0}
        process = subprocess.Popen(  # noqa: S603
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        on_start(process)
        stdout, stderr = process.communicate()
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": process.returncode,
        }


def _session() -> tuple[WorkerSession, FakeInstance]:
    session = WorkerSession()
    instance = FakeInstance()
    session.instance = instance  # type: ignore[assignment]
    return session, instance


def _wait(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_run_command_without_id_is_unchanged_legacy_path():
    session, instance = _session()
    result = session._run_command({"command": "echo hi"})
    assert result == {"stdout": "legacy", "stderr": "", "exit_code": 0}
    assert "cancelled" not in result  # legacy shape untouched
    assert instance.calls == ["echo hi"]


def test_cancel_before_start_never_runs_the_command():
    session, instance = _session()
    pending = session.cancel_command("c-before")
    assert pending == {"cancelled": True, "interrupted": False, "state": "pending"}

    result = session._run_command({"command": "echo hi", "command_id": "c-before"})
    assert result["cancelled"] is True
    assert result["exit_code"] == step_worker.CANCELLED_BEFORE_START_EXIT
    assert instance.calls == []  # never executed


def test_cancel_running_interrupts_only_that_command():
    session, _ = _session()
    outcome: dict = {}

    def run():
        outcome["result"] = session._run_command(
            {"command": "sleep 30", "command_id": "c-run"}
        )

    thread = threading.Thread(target=run)
    started = time.monotonic()
    thread.start()
    assert _wait(lambda: "c-run" in session._active)

    response = session.cancel_command("c-run")
    thread.join(timeout=15)
    assert not thread.is_alive()

    assert response == {"cancelled": True, "interrupted": True, "state": "running"}
    assert outcome["result"]["cancelled"] is True
    assert outcome["result"]["exit_code"] == step_worker.CANCELLED_SIGNALLED_EXIT
    assert time.monotonic() - started < 20  # nowhere near the 30s sleep
    assert "c-run" not in session._active
    assert session.instance is not None  # instance not torn down


def test_cancel_stops_whole_group_including_term_ignoring_child(monkeypatch, tmp_path):
    # Leader exits on SIGTERM, but spawns a child in the same group that IGNORES
    # SIGTERM. Confirming only the leader would leak the child; confirming the
    # whole group is empty catches it.
    session, _ = _session()
    child_file = tmp_path / "child.pid"
    parent_cmd = (
        "( trap '' TERM; while :; do sleep 1; done ) >/dev/null 2>&1 & "
        f"echo $! > {child_file}; exec sleep 300"
    )
    monkeypatch.setattr(step_worker, "CANCEL_TERM_GRACE_SECONDS", 0.5)

    def run():
        session._run_command({"command": parent_cmd, "command_id": "grp"})

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert _wait(lambda: child_file.exists() and "grp" in session._active)
    child_pid = int(child_file.read_text().strip())
    assert os.kill(child_pid, 0) is None  # child alive before cancel

    response = session.cancel_command("grp")
    thread.join(timeout=15)

    assert response["interrupted"] is True
    # The SIGTERM-ignoring child must be gone, not leaked.
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_cancel_after_completion_is_idempotent_noop():
    session, _ = _session()
    result = session._run_command({"command": "echo done", "command_id": "c-done"})
    assert result["cancelled"] is False
    assert result["exit_code"] == 0

    late = session.cancel_command("c-done")
    assert late == {
        "cancelled": False,
        "interrupted": False,
        "state": "already_finished",
    }


def test_reused_command_id_is_rejected():
    session, _ = _session()
    session._run_command({"command": "echo one", "command_id": "dup"})
    # A completed id may not be reused: an old cancellation must never be able to
    # land on a different execution that happens to share the id.
    with pytest.raises(ValueError, match="already been used"):
        session._run_command({"command": "echo two", "command_id": "dup"})


def test_command_env_path_is_cancellable_too():
    # A cancellable command may still carry env overrides; the env transport and
    # the command_id registry must compose.
    session, instance = _session()
    pending = session.cancel_command("env-before")
    assert pending["state"] == "pending"
    result = session._run_command_env(
        {"command": "echo hi", "command_id": "env-before", "env": {"FOO": "bar"}}
    )
    assert result["cancelled"] is True
    assert result["exit_code"] == step_worker.CANCELLED_BEFORE_START_EXIT
    assert instance.calls == []  # cancelled before start, env path included


def test_failed_interruption_is_visible_not_silent(monkeypatch):
    session, _ = _session()
    process = subprocess.Popen(["sleep", "30"], start_new_session=True)  # noqa: S607,S603
    try:
        # Registry stores the retained pgid (== pid under start_new_session).
        session._active["c-stuck"] = process.pid
        # Simulate a group that refuses to die: swallow signals and never gone.
        monkeypatch.setattr(step_worker.os, "killpg", lambda *a, **k: None)
        monkeypatch.setattr(
            WorkerSession, "_wait_group_gone", staticmethod(lambda *a, **k: False)
        )
        response = session.cancel_command("c-stuck")
        assert response["cancelled"] is True
        assert response["interrupted"] is False
        assert "did not exit" in response["error"]
        # The failed interruption is recorded on the cancel barrier so a waiting
        # run path adopts it instead of synthesizing a successful cancellation.
        op = session._cancel_inflight.get("c-stuck")
        assert op is not None
        assert op.interrupted is False
        assert "did not exit" in op.error
    finally:
        process.kill()
        process.wait(timeout=5)


def test_racing_cancel_signal_failure_is_surfaced(monkeypatch):
    # A cancel that lands in the on_start race window and then fails to stop the
    # process must not be silently lost: run_command surfaces it.
    session, _ = _session()

    class RacingInstance:
        def run_command(
            self, command, timeout, *, env=None, on_start=None, command_id=None
        ):
            proc = subprocess.Popen(["sleep", "0.3"], start_new_session=True)  # noqa: S607,S603
            session._cancelled["rid"] = None  # cancel arrives in the window
            on_start(proc)
            proc.wait()
            return {"stdout": "", "stderr": "", "exit_code": 0}

    session.instance = RacingInstance()  # type: ignore[assignment]
    monkeypatch.setattr(
        WorkerSession, "_signal_process", lambda self, p: (False, "boom")
    )
    with pytest.raises(step_worker.SandfleetError, match="Could not interrupt"):
        session._run_command({"command": "x", "command_id": "rid"})


def test_signal_process_kills_group_after_leader_is_reaped(tmp_path):
    # Regression: a new-session leader backgrounds a child in its group, then
    # exits and is reaped. Deriving the pgid from the (now gone) leader would
    # short-circuit to a false success and leak the child. The retained pgid must
    # still be signalled -- absence of the leader is not absence of the group.
    session, _ = _session()
    child_file = tmp_path / "child.pid"
    leader = subprocess.Popen(  # noqa: S602,S603
        ["bash", "-c", f"sleep 60 & echo $! > {child_file}; exit 0"],
        start_new_session=True,
    )
    pgid = leader.pid  # start_new_session => leader is the group leader
    leader.wait(timeout=5)  # reap the leader; the group lives on via the child
    assert _wait(lambda: child_file.exists())
    child_pid = int(child_file.read_text().strip())
    assert os.kill(child_pid, 0) is None  # child alive, leader already reaped

    interrupted, error = session._signal_process(pgid)
    assert interrupted is True
    assert error is None
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)  # the child was actually reaped, not leaked


def test_signal_process_refuses_unsafe_process_group():
    session, _ = _session()
    for unsafe in (0, 1):
        interrupted, error = session._signal_process(unsafe)
        assert interrupted is False
        assert "unsafe process group" in error


def test_running_path_signal_failure_is_recorded_and_surfaced(monkeypatch):
    # Regression: a cancel on an already-running command whose signal FAILS must
    # not let the run path report a synthetic cancelled=true/137; the failure is
    # recorded on the running path and surfaced as an error.
    session, _ = _session()
    monkeypatch.setattr(
        WorkerSession, "_signal_process", lambda self, pgid: (False, "failed")
    )
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = session._run_command(
                {"command": "sleep 3", "command_id": "r2"}
            )
        except Exception as error:  # noqa: BLE001
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    assert _wait(lambda: "r2" in session._active)

    response = session.cancel_command("r2")
    assert response["cancelled"] is True
    assert response["interrupted"] is False  # honest: the signal failed

    thread.join(timeout=15)
    assert not thread.is_alive()
    # The run path raised rather than claiming a successful cancellation.
    assert "result" not in outcome
    assert isinstance(outcome.get("error"), step_worker.SandfleetError)
    assert "Could not interrupt" in str(outcome["error"])


class _StubOverlay:
    def env(self):
        return {}


def _disk_apptainer_instance(name: str):
    """A minimally-constructed ApptainerInstance in disk (PID-namespace) mode.

    Bypasses __init__ (which needs a real image/overlay) and sets only the
    attributes the in-instance cancel path reads."""
    inst = step_worker.ApptainerInstance.__new__(step_worker.ApptainerInstance)
    inst.name = name
    inst.overlay = _StubOverlay()
    inst.start_flags = step_worker.DISK_START_FLAGS
    inst.apptainer = "apptainer"
    inst.pwd = "/work"
    inst.cache_dir = None
    inst.tmp_dir = None
    inst.network = None
    inst.gpu_enabled = False
    inst.sandbox_proxy_url = None
    return inst


def _cleanup_command_files(inst, command_id):
    base = inst._command_pidfile(command_id)
    for path in (base, base + ".tmp", base + ".stop"):
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)


def test_uses_pid_namespace_tracks_start_flags_not_storage():
    inst = _disk_apptainer_instance("sandfleet-1-abc")
    assert inst.uses_pid_namespace is True
    inst.overlay = None
    inst.start_flags = step_worker.DEFAULT_START_FLAGS
    assert inst.uses_pid_namespace is True
    for flag in ("--containall", "-C", "--pid", "-p"):
        inst.start_flags = (flag,)
        assert inst.uses_pid_namespace is True
    inst.start_flags = ("--contain", "--writable-tmpfs")
    assert inst.uses_pid_namespace is False


def test_command_pidfile_is_scoped_to_instance_and_command_and_sanitized():
    a = _disk_apptainer_instance("sandfleet-1-aaa")
    b = _disk_apptainer_instance("sandfleet-2-bbb")
    # Same command id, different instance generation => different file.
    assert a._command_pidfile("cid") != b._command_pidfile("cid")
    # Same instance, different command id => different file.
    assert a._command_pidfile("c1") != a._command_pidfile("c2")
    # Path traversal / odd chars in the id cannot escape the naming scheme.
    dangerous = a._command_pidfile("../../etc/passwd")
    assert "/" not in dangerous[len("/tmp/") :]
    assert dangerous.startswith("/tmp/sandfleet-cancel-sandfleet-1-aaa-")


def test_command_pidfile_is_collision_free_across_sanitizer_aliases():
    # Sanitization alone would map a/b and a?b to the same file; the sha256
    # component keeps distinct ids distinct.
    a = _disk_apptainer_instance("sandfleet-1-aaa")
    assert a._command_pidfile("a/b") != a._command_pidfile("a?b")
    assert a._command_pidfile("x y") != a._command_pidfile("x_y")


def test_in_instance_wrapper_publishes_a_killable_pgid(tmp_path):
    import uuid as _uuid

    inst = _disk_apptainer_instance("sandfleet-pub-" + _uuid.uuid4().hex[:6])
    command_id = "cmd-" + _uuid.uuid4().hex[:6]
    pidfile = inst._command_pidfile(command_id)
    launch = inst._wrap_for_in_instance_cancel(command_id, "sleep 5")
    proc = subprocess.Popen(["bash", "-c", launch], start_new_session=True)  # noqa: S602,S603
    try:
        assert _wait(lambda: os.path.exists(pidfile) and os.path.getsize(pidfile) > 0)
        pgid = int(open(pidfile).read().strip())  # noqa: SIM115
        assert pgid > 1
        # The published pgid names a real, killable in-session group.
        os.killpg(pgid, signal.SIGKILL)
        assert _wait(lambda: proc.poll() is not None)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
        _cleanup_command_files(inst, command_id)


def test_in_instance_wrapper_honors_a_precreated_stop_sentinel():
    # Cancel that lands BEFORE publication drops a .stop sentinel; the command
    # must self-abort right after publishing, never running its payload.
    import uuid as _uuid

    inst = _disk_apptainer_instance("sandfleet-stop-" + _uuid.uuid4().hex[:6])
    command_id = "cmd-" + _uuid.uuid4().hex[:6]
    from pathlib import Path

    Path(inst._command_cancelfile(command_id)).write_text("")
    launch = inst._wrap_for_in_instance_cancel(command_id, "sleep 30")
    started = time.monotonic()
    proc = subprocess.Popen(["bash", "-c", launch], start_new_session=True)  # noqa: S602,S603
    try:
        proc.wait(timeout=10)
        assert time.monotonic() - started < 8  # never ran the 30s payload
        assert proc.returncode == step_worker.CANCELLED_SIGNALLED_EXIT
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        _cleanup_command_files(inst, command_id)


def test_cancel_command_group_noop_when_not_pid_namespaced(monkeypatch):
    inst = _disk_apptainer_instance("sandfleet-1-abc")
    inst.overlay = None  # host-namespace instance: killpg already reaches it
    inst.start_flags = ("--contain",)
    called = False

    def fake_run(*a, **k):
        nonlocal called
        called = True
        raise AssertionError("must not exec for a host-namespace instance")

    monkeypatch.setattr(step_worker.subprocess, "run", fake_run)
    assert inst.cancel_command_group("cid") == (True, None)
    assert called is False


def _fake_completed(returncode, stderr=b""):
    return subprocess.CompletedProcess(["apptainer"], returncode, b"", stderr)


@pytest.mark.parametrize("disk_backed", [False, True])
def test_cancel_command_group_success_targets_scoped_pidfile(monkeypatch, disk_backed):
    inst = _disk_apptainer_instance("sandfleet-1-abc")
    if not disk_backed:
        inst.overlay = None
        inst.start_flags = step_worker.DEFAULT_START_FLAGS
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        return _fake_completed(0)

    monkeypatch.setattr(step_worker.subprocess, "run", fake_run)
    gone, error = inst.cancel_command_group("cid")
    assert gone is True and error is None
    assert "instance://sandfleet-1-abc" in captured["argv"]
    snippet = captured["argv"][-1]
    assert inst._command_pidfile("cid") in snippet  # scoped, collision-free path
    # It refuses to signal an unsafe pgid and confirms the whole group is gone by
    # scanning /proc (never kill -0, which conflates EPERM with ESRCH).
    assert 'if [ "$pgid" -le 1 ]' in snippet
    assert 'kill -TERM -"$pgid"' in snippet
    assert 'kill -KILL -"$pgid"' in snippet
    assert "group_alive()" in snippet
    assert "/proc/[0-9]*/stat" in snippet
    assert 'kill -0 -"$pgid"' not in snippet


def test_default_tmpfs_command_publishes_in_namespace_cancellation_identity(
    monkeypatch,
):
    inst = step_worker.ApptainerInstance(
        image="python:3.12",
        timeout=30,
        pwd="/work",
        cache_dir=None,
        tmp_dir=None,
        extra_start_flags=[],
        apptainer_binary="apptainer",
    )
    inst.name = "sandfleet-1-tmpfs"
    captured = {}

    def fake_exec(argv, **kwargs):
        captured["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    monkeypatch.setattr(inst, "exec", fake_exec)
    result = inst.run_command("sleep 30", 30, command_id="tmpfs-command")
    assert result["exit_code"] == 0
    assert "setsid -w" in captured["argv"][-1]
    assert inst._command_pidfile("tmpfs-command") in captured["argv"][-1]


def test_cancel_command_group_absent_pidfile_fails_loud_not_gone(monkeypatch):
    # exit 2 == the pidfile never appeared: the command's state is unknown, so
    # this must NOT be reported as a confirmed reap.
    inst = _disk_apptainer_instance("sandfleet-1-abc")
    monkeypatch.setattr(
        step_worker.subprocess, "run", lambda *a, **k: _fake_completed(2, b"NO_PIDFILE")
    )
    gone, error = inst.cancel_command_group("cid")
    assert gone is False
    assert "not reaped" in error


def test_cancel_command_group_invalid_pidfile_fails_loud(monkeypatch):
    # exit 3 == pidfile present but invalid/unsafe pgid: corruption, fail loud.
    inst = _disk_apptainer_instance("sandfleet-1-abc")
    monkeypatch.setattr(
        step_worker.subprocess,
        "run",
        lambda *a, **k: _fake_completed(3, b"BAD_PIDFILE"),
    )
    gone, error = inst.cancel_command_group("cid")
    assert gone is False
    assert "not reaped" in error


def test_cancel_command_group_surfaces_a_survivor(monkeypatch):
    inst = _disk_apptainer_instance("sandfleet-1-abc")
    monkeypatch.setattr(
        step_worker.subprocess, "run", lambda *a, **k: _fake_completed(1, b"")
    )
    gone, error = inst.cancel_command_group("cid")
    assert gone is False
    assert "not reaped" in error


def test_cancel_command_group_surfaces_a_timeout(monkeypatch):
    inst = _disk_apptainer_instance("sandfleet-1-abc")

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="apptainer", timeout=1)

    monkeypatch.setattr(step_worker.subprocess, "run", fake_run)
    gone, error = inst.cancel_command_group("cid")
    assert gone is False
    assert "timed out" in error


def test_interrupt_requires_both_sides_to_succeed(monkeypatch):
    session, _ = _session()
    monkeypatch.setattr(
        WorkerSession, "_signal_process", lambda self, pgid: (True, None)
    )
    # Both the in-instance reap and the host-client stop succeeded.
    interrupted, error = session._interrupt(12345, lambda: (True, None))
    assert interrupted is True
    assert error is None


def test_interrupt_surfaces_host_failure_even_when_in_instance_reap_ok(monkeypatch):
    # The host-side result must not be discarded: a stuck host exec client (the
    # main channel would stay blocked) is an explicit failure.
    session, _ = _session()
    monkeypatch.setattr(
        WorkerSession, "_signal_process", lambda self, pgid: (False, "host stuck")
    )
    interrupted, error = session._interrupt(12345, lambda: (True, None))
    assert interrupted is False
    assert "host stuck" in error


def test_interrupt_surfaces_in_instance_kill_failure(monkeypatch):
    session, _ = _session()
    monkeypatch.setattr(
        WorkerSession, "_signal_process", lambda self, pgid: (True, None)
    )
    interrupted, error = session._interrupt(12345, lambda: (False, "survivor left"))
    assert interrupted is False
    assert "survivor left" in error


def test_interrupt_surfaces_both_failures(monkeypatch):
    session, _ = _session()
    monkeypatch.setattr(
        WorkerSession, "_signal_process", lambda self, pgid: (False, "host stuck")
    )
    interrupted, error = session._interrupt(12345, lambda: (False, "survivor left"))
    assert interrupted is False
    assert "survivor left" in error
    assert "host stuck" in error


class FakeDiskInstance:
    """Disk-mode fake: runs the command locally but claims a PID namespace, so
    WorkerSession routes cancel through the in-instance killer."""

    uses_pid_namespace = True
    timeout = 60

    def __init__(self) -> None:
        self.group_killed: list[str] = []
        self.kill_result: tuple[bool, str | None] = (True, None)

    def run_command(
        self,
        command,
        timeout,
        *,
        env=None,
        on_start=None,
        command_id=None,
        on_complete=None,
        worker_deadline=False,
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
        if on_complete is not None:
            on_complete()
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": process.returncode,
        }

    def cancel_command_group(self, command_id):
        self.group_killed.append(command_id)
        return self.kill_result


def test_disk_mode_cancel_routes_through_the_in_instance_killer():
    session = WorkerSession()
    instance = FakeDiskInstance()
    session.instance = instance  # type: ignore[assignment]
    outcome: dict = {}

    def run():
        outcome["result"] = session._run_command(
            {"command": "sleep 30", "command_id": "disk-run"}
        )

    thread = threading.Thread(target=run)
    thread.start()
    assert _wait(lambda: "disk-run" in session._active)

    response = session.cancel_command("disk-run")
    thread.join(timeout=15)
    assert not thread.is_alive()

    assert instance.group_killed == ["disk-run"]  # in-instance reap was invoked
    assert response == {"cancelled": True, "interrupted": True, "state": "running"}
    assert outcome["result"]["cancelled"] is True
    assert "disk-run" not in session._active
    assert "disk-run" not in session._instance_killers


def test_completion_race_failed_cancel_is_not_published_as_success(monkeypatch):
    # Barrier-controlled completion race (per review 5611810258/5611860169):
    # pause signalling, let the exec command RETURN, then release a FAILED signal
    # result. The exec must not publish a synthetic cancelled=true/137 -- it must
    # block on the cancel barrier and adopt the failure, raising instead.
    session, _ = _session()
    entered = threading.Event()
    release = threading.Event()

    def slow_failed_signal(self, pgid):
        entered.set()
        release.wait(10)
        return (False, "signal failed")

    monkeypatch.setattr(WorkerSession, "_signal_process", slow_failed_signal)

    outcome: dict = {}

    def run():
        try:
            outcome["result"] = session._run_command(
                {"command": "sleep 0.2", "command_id": "race"}
            )
        except Exception as error:  # noqa: BLE001
            outcome["error"] = error

    runner = threading.Thread(target=run)
    runner.start()
    assert _wait(lambda: "race" in session._active)

    canceller = threading.Thread(target=lambda: session.cancel_command("race"))
    canceller.start()
    assert entered.wait(5)  # the cancel is now inside (paused) signalling

    # The exec command (sleep 0.2) finishes now, while signalling is still paused.
    time.sleep(0.6)
    # It must still be blocked on the cancel barrier, not already returned.
    assert runner.is_alive()

    release.set()  # signalling returns FAILURE
    canceller.join(timeout=5)
    runner.join(timeout=5)
    assert not runner.is_alive()

    # Never published a synthetic success; adopted the cancel's failure.
    assert "result" not in outcome
    assert isinstance(outcome.get("error"), step_worker.SandfleetError)
    assert "Could not interrupt" in str(outcome["error"])
    assert "race" not in session._cancel_inflight  # barrier cleaned up


def test_completion_race_successful_cancel_still_reports_cancelled(monkeypatch):
    # The barrier must not penalise the normal path: a cancel that succeeds while
    # the exec is finishing still yields a clean cancelled=true/137.
    session, _ = _session()
    entered = threading.Event()
    release = threading.Event()

    def slow_ok_signal(self, pgid):
        entered.set()
        release.wait(10)
        return (True, None)

    monkeypatch.setattr(WorkerSession, "_signal_process", slow_ok_signal)
    outcome: dict = {}

    def run():
        outcome["result"] = session._run_command(
            {"command": "sleep 0.2", "command_id": "race-ok"}
        )

    runner = threading.Thread(target=run)
    runner.start()
    assert _wait(lambda: "race-ok" in session._active)
    canceller = threading.Thread(target=lambda: session.cancel_command("race-ok"))
    canceller.start()
    assert entered.wait(5)
    time.sleep(0.6)
    release.set()
    canceller.join(timeout=5)
    runner.join(timeout=5)

    assert outcome["result"]["cancelled"] is True
    assert outcome["result"]["exit_code"] == step_worker.CANCELLED_SIGNALLED_EXIT
    assert "race-ok" not in session._cancel_inflight


def test_capture_process_truncates_a_runaway_writer_without_blocking():
    # >1 MB emitted fast: a still-running writer must not block wait(); capture is
    # bounded and truncation stays detectable (len == cap == MAX_OUTPUT_BYTES + 1).
    proc = subprocess.Popen(  # noqa: S602,S603
        ["bash", "-c", "yes X | head -c 3000000"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    started = time.monotonic()
    result = step_worker.capture_process(
        proc, max_capture=step_worker.MAX_OUTPUT_BYTES + 1
    )
    assert time.monotonic() - started < 20
    assert result.returncode == 0
    assert len(result.stdout) == step_worker.MAX_OUTPUT_BYTES + 1


def test_capture_process_returns_despite_background_pipe_holder():
    # A backgrounded child inherits stdout and lives on after the direct child
    # exits; the bounded path must return promptly, not block on the held pipe.
    proc = subprocess.Popen(  # noqa: S602,S603
        ["bash", "-c", "(sleep 30 &) ; echo done"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    started = time.monotonic()
    result = step_worker.capture_process(
        proc, max_capture=step_worker.MAX_OUTPUT_BYTES + 1, grace_sec=1.0
    )
    assert time.monotonic() - started < 15
    assert b"done" in result.stdout
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
