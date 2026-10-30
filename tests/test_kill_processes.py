"""Killing every sandbox process except init (the `kill-processes` operation).

The kill is host-side: the worker joins the instance's pinned user+PID namespaces
and runs the kernel's atomic kill(-1), then verifies with a /proc scan. These
tests never let a real signal escape to a host process: nsenter, the scan, the
namespace walk and pidfd_send_signal are all mocked, with a tripwire that fails
if the real nsenter or pidfd_send_signal is ever used. The WorkerSession
integration tests use a fake instance that only kills its own real child group.
"""

from __future__ import annotations

import contextlib
import errno
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from sandfleet import step_worker
from sandfleet.step_worker import WorkerSession, kill_namespace_processes

TARGET_INODE = 4026556268
WORKER_INODE = 4026531836


@pytest.fixture(autouse=True)
def no_real_signals(monkeypatch):
    """Tripwire: a host-scan test must never reach the real kill primitives."""

    def forbidden(name):
        def guard(*_args, **_kwargs):
            raise AssertionError(f"real {name} called in a unit test")

        return guard

    monkeypatch.setattr(step_worker.signal, "pidfd_send_signal", forbidden("pidfd"))
    monkeypatch.setattr(step_worker.subprocess, "run", forbidden("subprocess.run"))


# --- the namespace membership walk (fail-closed) -----------------------------


class _Stat:
    def __init__(self, inode, dev=7):
        self.st_ino = inode
        self.st_dev = dev


class _FakeNamespaces:
    """A fake PID-namespace parent chain for pid_in_namespace().

    ``chains`` maps a pid to the inodes from its own namespace up toward the host;
    the top has no accessible parent (EPERM). ``open_errors`` maps a pid to an
    OSError raised by os.open; ``status`` maps a pid to /proc/<pid>/status text.
    """

    def __init__(
        self, monkeypatch, chains, *, open_errors=None, status=None, status_errors=None
    ):
        self.chains = chains
        self.open_errors = open_errors or {}
        self.status = status or {}
        self.status_errors = status_errors or {}
        self._fd = 900
        self._inode: dict[int, int] = {}
        self._rest: dict[int, list[int]] = {}
        monkeypatch.setattr(step_worker.os, "open", self._open)
        monkeypatch.setattr(step_worker.os, "fstat", self._fstat)
        monkeypatch.setattr(step_worker.os, "close", lambda _fd: None)
        monkeypatch.setattr(step_worker.fcntl, "ioctl", self._ioctl)
        monkeypatch.setattr(step_worker.os, "getuid", lambda: 1000)
        fake = self

        def read_text(path_self, *_a, **_k):
            pid = int(str(path_self).split("/")[2])
            if pid in fake.status_errors:
                raise fake.status_errors[pid]
            if pid in fake.status:
                return fake.status[pid]
            raise FileNotFoundError(path_self)

        monkeypatch.setattr(step_worker.Path, "read_text", read_text)

    def _open(self, path, _flags):
        pid = int(str(path).split("/")[2])
        if pid in self.open_errors:
            raise self.open_errors[pid]
        if pid not in self.chains:
            raise FileNotFoundError(path)
        return self._descriptor(list(self.chains[pid]))

    def _descriptor(self, inodes):
        self._fd += 1
        self._inode[self._fd] = inodes[0]
        self._rest[self._fd] = inodes[1:]
        return self._fd

    def _fstat(self, fd):
        return _Stat(self._inode[fd])

    def _ioctl(self, fd, request):
        assert request == step_worker.NS_GET_PARENT
        rest = self._rest[fd]
        if not rest:
            raise PermissionError(errno.EPERM, "EPERM")
        return self._descriptor(rest)


def test_member_directly_in_the_namespace(monkeypatch):
    _FakeNamespaces(monkeypatch, {42: [TARGET_INODE]})
    assert step_worker.pid_in_namespace(42, TARGET_INODE) is True


def test_member_in_a_nested_namespace(monkeypatch):
    _FakeNamespaces(monkeypatch, {42: [5, TARGET_INODE, 1]})
    assert step_worker.pid_in_namespace(42, TARGET_INODE) is True


def test_outside_process_is_not_a_member(monkeypatch):
    _FakeNamespaces(monkeypatch, {42: [7, 8]})
    assert step_worker.pid_in_namespace(42, TARGET_INODE) is False


def test_vanished_process_is_not_a_member(monkeypatch):
    _FakeNamespaces(monkeypatch, {}, open_errors={42: FileNotFoundError("gone")})
    assert step_worker.pid_in_namespace(42, TARGET_INODE) is False


def test_permission_error_on_a_process_that_may_be_ours_fails_closed(monkeypatch):
    _FakeNamespaces(
        monkeypatch,
        {},
        open_errors={42: PermissionError(errno.EACCES, "denied")},
        status={42: "Name:\tx\nUid:\t1000\t1000\t1000\t1000\n"},
    )
    with pytest.raises(RuntimeError, match="may belong to the sandbox"):
        step_worker.pid_in_namespace(42, TARGET_INODE)


def test_permission_error_on_another_uid_is_skipped(monkeypatch):
    _FakeNamespaces(
        monkeypatch,
        {},
        open_errors={42: PermissionError(errno.EACCES, "denied")},
        status={42: "Name:\tx\nUid:\t0\t0\t0\t0\n"},
    )
    assert step_worker.pid_in_namespace(42, TARGET_INODE) is False


def test_permission_error_with_unreadable_status_fails_closed(monkeypatch):
    _FakeNamespaces(
        monkeypatch,
        {},
        open_errors={42: PermissionError(errno.EPERM, "denied")},
        status_errors={42: PermissionError(errno.EACCES, "status denied")},
    )
    with pytest.raises(RuntimeError, match="may belong to the sandbox"):
        step_worker.pid_in_namespace(42, TARGET_INODE)


def test_permission_error_on_a_process_that_vanished_is_skipped(monkeypatch):
    # Denied open, then the process disappeared (status gone): not a member.
    _FakeNamespaces(
        monkeypatch, {}, open_errors={42: PermissionError(errno.EACCES, "denied")}
    )
    assert step_worker.pid_in_namespace(42, TARGET_INODE) is False


# --- the host-side kill loop (atomic kill(-1) + verify) ----------------------


TARGET = (7, TARGET_INODE)
WORKER = (7, WORKER_INODE)


def _patch_loop(
    monkeypatch, scans, *, needs_killing=lambda _pid: True, init_alive=True
):
    kills = {"n": 0}
    monkeypatch.setattr(step_worker.os, "fstat", lambda _fd: _Stat(TARGET_INODE))
    monkeypatch.setattr(step_worker, "_worker_pid_namespace_identity", lambda: WORKER)
    monkeypatch.setattr(
        step_worker, "_live_pid_namespace_identity", lambda _pid: TARGET
    )
    monkeypatch.setattr(step_worker.os, "pidfd_open", lambda _pid: 777)
    monkeypatch.setattr(step_worker.os, "close", lambda _fd: None)

    def ensure(_pidfd):
        if not init_alive:
            raise RuntimeError("Sandbox init exited during the kill; aborting")

    monkeypatch.setattr(step_worker, "_ensure_init_alive", ensure)
    monkeypatch.setattr(
        step_worker,
        "_run_namespace_kill",
        lambda *_a, **_k: kills.__setitem__("n", kills["n"] + 1),
    )
    calls = iter(scans)
    monkeypatch.setattr(
        step_worker, "_namespace_member_pids", lambda *_a: list(next(calls))
    )
    monkeypatch.setattr(step_worker, "_needs_killing", needs_killing)
    return kills


def test_kill_loop_counts_distinct_and_kills_atomically(monkeypatch):
    kills = _patch_loop(monkeypatch, [[11, 12], [12], [12], []])
    assert kill_namespace_processes(2, 10, 11, 5) == 2
    assert kills["n"] == 2  # kill(-1) each loop until a scan is clean


def test_kill_loop_handles_a_fork_after_the_first_kill(monkeypatch):
    # A process forked after the first scan appears only on a later scan.
    kills = _patch_loop(monkeypatch, [[11], [99], [99], []])
    assert kill_namespace_processes(2, 10, 11, 5) == 2  # 11 and the forked 99
    assert kills["n"] == 2


def test_kill_loop_ignores_plain_zombies(monkeypatch):
    kills = _patch_loop(monkeypatch, [[11], [11]], needs_killing=lambda _pid: False)
    assert kill_namespace_processes(2, 10, 11, 5) == 1
    assert kills["n"] == 1


def test_kill_loop_fails_loudly_on_survivors(monkeypatch):
    monkeypatch.setattr(step_worker.time, "monotonic", lambda: 1000.0)
    _patch_loop(monkeypatch, [[11], [11]])
    with pytest.raises(RuntimeError, match=r"survived SIGKILL: \[11\]"):
        kill_namespace_processes(2, 10, 11, 0)


def test_kill_loop_aborts_if_init_dies_mid_loop(monkeypatch):
    _patch_loop(monkeypatch, [[11]], init_alive=False)
    with pytest.raises(RuntimeError, match="init exited during the kill"):
        kill_namespace_processes(2, 10, 11, 5)


def test_kill_loop_refuses_a_reused_pinned_namespace(monkeypatch):
    monkeypatch.setattr(step_worker.os, "fstat", lambda _fd: _Stat(TARGET_INODE))
    monkeypatch.setattr(step_worker, "_worker_pid_namespace_identity", lambda: WORKER)
    monkeypatch.setattr(
        step_worker, "_live_pid_namespace_identity", lambda _pid: WORKER
    )
    with pytest.raises(RuntimeError, match="no longer owns the pinned"):
        kill_namespace_processes(2, 10, 11, 5)


def test_kill_loop_refuses_the_workers_own_namespace(monkeypatch):
    # Defence in depth: the pinned fd resolving to the worker's PID namespace.
    monkeypatch.setattr(step_worker.os, "fstat", lambda _fd: _Stat(WORKER_INODE))
    monkeypatch.setattr(step_worker, "_worker_pid_namespace_identity", lambda: WORKER)
    with pytest.raises(RuntimeError, match="is the worker's"):
        kill_namespace_processes(2, 10, 11, 5)


# --- the namespace-kill helper -----------------------------------------------


class _FakeHelper:
    """Stands in for the helper's Popen; never starts a real process."""

    instances: list[_FakeHelper] = []

    def __init__(self, argv, *, returncode=0, err=b"", hang=False, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4242
        self.returncode = returncode
        self._stderr = err
        self._hang = hang
        self.timeouts: list[float | None] = []
        _FakeHelper.instances.append(self)

    def communicate(self, timeout=None):
        self.timeouts.append(timeout)
        if self._hang and len(self.timeouts) == 1:
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return b"", self._stderr


def _patch_helper(monkeypatch, **options):
    _FakeHelper.instances = []
    monkeypatch.setattr(
        step_worker.subprocess,
        "Popen",
        lambda argv, **kwargs: _FakeHelper(argv, **options, **kwargs),
    )


def test_namespace_kill_helper_joins_user_and_pid_namespaces(monkeypatch):
    _patch_helper(monkeypatch)
    step_worker._run_namespace_kill(10, 11, (7, 200), (7, 100), timeout=4.0)
    [helper] = _FakeHelper.instances
    argv = helper.argv
    assert argv[0] == sys.executable
    assert "-I" in argv and "-S" in argv  # isolated interpreter
    assert "--mount" not in " ".join(argv)  # no mount namespace, no image code
    assert "nsenter" not in " ".join(argv)  # no nsenter dependency
    program = argv[argv.index("-c") + 1]
    assert "os.setns(user_fd, CLONE_NEWUSER)" in program
    assert "os.setns(pid_fd, CLONE_NEWPID)" in program
    assert "PR_SET_DUMPABLE" in program and "fail(4," in program  # prctl checked
    assert "os.kill(-1, 9)" in program
    # fd numbers then target (dev,ino) then worker (dev,ino) as argv.
    assert argv[-6:] == ["10", "11", "7", "200", "7", "100"]
    assert helper.kwargs["pass_fds"] == (10, 11)
    assert helper.kwargs["start_new_session"] is True  # killable as a group
    assert helper.timeouts == [4.0]


def test_namespace_kill_helper_raises_on_any_nonzero_exit(monkeypatch):
    _patch_helper(monkeypatch, returncode=3, err=b"child is not in the pinned")
    with pytest.raises(RuntimeError, match="child is not in the pinned"):
        step_worker._run_namespace_kill(10, 11, (7, 200), (7, 100), timeout=4.0)


def test_timed_out_helper_group_is_sigkilled(monkeypatch):
    # A sandbox could SIGSTOP the helper's child; the timeout must SIGKILL the
    # helper's whole session/group (child included), then raise.
    _patch_helper(monkeypatch, hang=True)
    killed = []
    monkeypatch.setattr(
        step_worker.os, "killpg", lambda pgid, sig: killed.append((pgid, sig))
    )
    with pytest.raises(RuntimeError, match="timed out after 4.0s"):
        step_worker._run_namespace_kill(10, 11, (7, 200), (7, 100), timeout=4.0)
    [helper] = _FakeHelper.instances
    assert killed == [(helper.pid, signal.SIGKILL)]
    assert helper.timeouts == [4.0, None]  # reaped after the group kill


# --- ApptainerInstance guards ------------------------------------------------


def _instance(pid_ns_fd=11, user_ns_fd=10):
    inst = step_worker.ApptainerInstance.__new__(step_worker.ApptainerInstance)
    inst.name = "sandfleet-1-abc"
    inst.pid_ns_fd = pid_ns_fd
    inst.user_ns_fd = user_ns_fd
    return inst


def test_instance_kill_requires_pinned_namespaces():
    inst = _instance(pid_ns_fd=None, user_ns_fd=None)
    with pytest.raises(RuntimeError, match="not pinned"):
        inst.kill_processes()


def test_instance_kill_requires_a_started_sandbox():
    inst = _instance()
    inst.name = None
    with pytest.raises(RuntimeError, match="not been started"):
        inst.kill_processes()


def test_instance_kill_delegates_to_the_host_scan(monkeypatch):
    inst = _instance()
    seen = {}
    monkeypatch.setattr(
        inst, "_instance_pid", lambda timeout: seen.update(pid_timeout=timeout) or 4321
    )

    def fake(init_pid, user_ns_fd, pid_ns_fd, timeout):
        seen.update(pid=init_pid, user=user_ns_fd, ns=pid_ns_fd, timeout=timeout)
        return 4

    monkeypatch.setattr(step_worker, "kill_namespace_processes", fake)
    assert inst.kill_processes() == 4
    assert seen == {
        "pid_timeout": 10,
        "pid": 4321,
        "user": 10,
        "ns": 11,
        "timeout": step_worker.KILL_PROCESSES_TIMEOUT_SECONDS,
    }


def test_instance_kill_caps_each_step_by_the_remaining_budget(monkeypatch):
    inst = _instance()
    seen = {}
    monkeypatch.setattr(step_worker.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        inst, "_instance_pid", lambda timeout: seen.update(pid_timeout=timeout) or 1
    )
    monkeypatch.setattr(
        step_worker,
        "kill_namespace_processes",
        lambda *a: seen.update(kill_timeout=a[3]) or 0,
    )
    inst.kill_processes(deadline=103.0)  # 3s left: both steps capped to 3s
    assert seen == {"pid_timeout": 3.0, "kill_timeout": 3.0}


def test_instance_kill_refuses_to_start_with_no_budget_left(monkeypatch):
    inst = _instance()
    monkeypatch.setattr(step_worker.time, "monotonic", lambda: 100.0)
    with pytest.raises(RuntimeError, match="ran out of its time budget"):
        inst.kill_processes(deadline=99.0)


def test_budget_arithmetic_fits_the_control_rpc_timeout():
    from sandfleet import step_backend

    assert (
        step_worker.LIFECYCLE_LOCK_TIMEOUT_SECONDS
        + step_worker.KILL_PROCESSES_TIMEOUT_SECONDS
        < step_worker.KILL_PROCESSES_BUDGET_SECONDS
        < step_backend.KILL_PROCESSES_CONTROL_TIMEOUT_SECONDS
    )


def test_signal_graces_fit_the_deadline(monkeypatch):
    monkeypatch.setattr(step_worker.time, "monotonic", lambda: 100.0)
    term, kill = WorkerSession._signal_graces(104.0)
    assert term + kill <= 4.0 and term == 2.0
    assert WorkerSession._signal_graces(None) == (
        step_worker.CANCEL_TERM_GRACE_SECONDS,
        step_worker.CANCEL_KILL_GRACE_SECONDS,
    )
    assert WorkerSession._signal_graces(90.0) == (0.0, 0.0)


def test_fallback_interrupt_caps_the_in_instance_killer(monkeypatch):
    monkeypatch.setattr(step_worker.time, "monotonic", lambda: 100.0)
    session = WorkerSession()
    seen = {}
    monkeypatch.setattr(session, "_signal_process", lambda pgid, deadline: (True, None))

    def killer(**options):
        seen.update(options)
        return True, None

    assert session._interrupt(1234, killer, deadline=110.0) == (True, None)
    assert seen == {"timeout": 5.0}  # half the remaining 10s


# --- WorkerSession: reconciling in-flight commands ---------------------------


class FakeInstance:
    """Runs commands as real local process groups; kill_processes kills them."""

    def __init__(self, kill_delay=0.0):
        self.timeout = 60
        self.process: subprocess.Popen | None = None
        self.started = threading.Event()
        self.kill_error: Exception | None = None
        self.kill_delay = kill_delay

    def run_command(self, command, timeout, *, env=None, on_start=None, **_options):
        process = subprocess.Popen(  # noqa: S603
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        self.process = process
        if on_start is not None:
            on_start(process)
        self.started.set()
        stdout, stderr = process.communicate()
        return {
            "stdout": stdout.decode(),
            "stderr": stderr.decode(),
            "exit_code": process.returncode,
        }

    def kill_processes(self, deadline=None):
        if self.kill_delay:
            time.sleep(self.kill_delay)
        if self.kill_error is not None:
            raise self.kill_error
        if self.process is None or self.process.poll() is not None:
            return 0
        os.killpg(self.process.pid, signal.SIGKILL)
        return 1


def _session(kill_delay=0.0):
    session = WorkerSession()
    instance = FakeInstance(kill_delay=kill_delay)
    session.instance = instance  # type: ignore[assignment]
    return session, instance


def _start(session, command_id, command="sleep 30"):
    outcome: dict = {}

    def run():
        try:
            outcome["result"] = session._run_command(
                {"command": command, "command_id": command_id}
            )
        except Exception as error:  # noqa: BLE001 - asserted by the tests
            outcome["error"] = error

    thread = threading.Thread(target=run)
    thread.start()
    return thread, outcome


def test_idle_kill_reports_no_cancelled_commands():
    session, _ = _session()
    assert session.kill_processes() == {"killed": 0, "cancelled_commands": []}


def test_slow_kill_still_reports_the_command_cancelled():
    # A slow kill lets the exec reach _await_command_outcome before the barrier
    # marks the command cancelled; the outcome must still be cancelled=True.
    session, instance = _session(kill_delay=0.3)
    thread, outcome = _start(session, "job-1")
    assert instance.started.wait(5)
    assert session.kill_processes() == {"killed": 1, "cancelled_commands": ["job-1"]}
    thread.join(10)
    assert not thread.is_alive()
    result = outcome["result"]
    assert result["cancelled"] is True
    assert result["exit_code"] == step_worker.CANCELLED_SIGNALLED_EXIT
    assert session.cancel_command("job-1")["state"] == "already_finished"
    # No stale registry entries.
    assert session._active == {} and session._cancel_inflight == {}
    assert "job-1" not in session._cancelled


def test_failed_kill_does_not_leave_a_command_cancelled_without_a_deadline():
    session, instance = _session()
    thread, outcome = _start(session, "job-1")
    assert instance.started.wait(5)
    instance.kill_error = RuntimeError("Sandbox processes survived SIGKILL: [9]")
    try:
        with pytest.raises(RuntimeError, match="survived SIGKILL"):
            session.kill_processes()
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(instance.process.pid, signal.SIGKILL)
    thread.join(10)
    assert not thread.is_alive()
    result = outcome.get("result")
    assert result is not None and result["cancelled"] is True
    assert session._active == {} and session._cancel_inflight == {}


def test_kill_fails_fast_when_a_lifecycle_change_holds_the_lock(monkeypatch):
    monkeypatch.setattr(step_worker, "LIFECYCLE_LOCK_TIMEOUT_SECONDS", 0.2)
    session, _ = _session()
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with session._lifecycle_lock:
            holding.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    assert holding.wait(5)
    try:
        with pytest.raises(step_worker.SandfleetError, match="starting or stopping"):
            session.kill_processes()
    finally:
        release.set()
        holder.join(5)


def test_kill_requires_a_started_sandbox():
    with pytest.raises(RuntimeError, match="not been started"):
        WorkerSession().kill_processes()


def test_kill_is_a_control_operation():
    session, _ = _session()
    assert step_worker._handle_control(session, {"operation": "kill_processes"}) == {
        "killed": 0,
        "cancelled_commands": [],
    }
