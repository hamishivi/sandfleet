"""Focused unit regressions for the exec output-capture path.

These exercise the SHIPPED function, `sandfleet.step_worker.capture_process`,
which `ApptainerInstance._exec_internal` calls -- not a copy of its logic. An
earlier revision of this file reimplemented the drain, so production could have
regressed with every test green.

No sandbox, Slurm or Apptainer required: `capture_process` takes a started
process, so ordinary subprocesses drive it in CI.

One test per defect found in review:
  short output with a surviving background child   read(n) waited for n or EOF
  byte-complete unlimited capture                  transfers must not truncate
  capping only on the command path                 the shared path carries files
  error propagation                                errors became partial output
  drain-thread tracking and teardown               threads were never joined
"""

import subprocess
import threading
import time

import pytest

from sandfleet.step_worker import ApptainerInstance, capture_process

CAP = 1_000_001  # what run_command passes: MAX_OUTPUT_BYTES + 1


def sh(cmd):
    return subprocess.Popen(
        ["bash", "-c", cmd],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_short_output_with_surviving_background_child():
    """The regression that made an earlier revision worse than the bug itself."""
    start = time.monotonic()
    result = capture_process(
        sh("echo hello; sleep 30 &"), max_capture=CAP, grace_sec=2.0
    )
    assert (
        result.stdout == b"hello\n"
    ), "short output must survive a live background child"
    assert result.returncode == 0
    assert time.monotonic() - start < 25, "must not wait on the background child"


def test_unlimited_capture_is_byte_complete():
    """File transfer shares this path; partial data would be silent corruption."""
    n = 3 * 1024 * 1024
    result = capture_process(sh(f"python3 -c \"import sys;sys.stdout.write('a'*{n})\""))
    assert result.returncode == 0
    assert len(result.stdout) == n


def test_cap_applies_only_when_requested():
    n = 2_000_000
    cmd = f"python3 -c \"import sys;sys.stdout.write('b'*{n})\""
    capped = capture_process(sh(cmd), max_capture=CAP)
    full = capture_process(sh(cmd))
    assert len(capped.stdout) == CAP, "the +1 keeps truncation detectable"
    assert len(full.stdout) == n, "the shared path must not cap"


def test_stdin_stderr_and_exit_code():
    result = capture_process(
        sh("cat; echo e >&2; exit 7"), stdin=b"fed\n", max_capture=CAP
    )
    assert result.returncode == 7
    assert b"fed" in result.stdout and b"e" in result.stderr


def test_capture_error_propagates():
    """A read failure must raise, not return a successful partial result."""

    class ExplodingStream:
        def read1(self, _n):
            raise OSError("read failed")

        def close(self):
            pass

    proc = sh("echo ignored")
    proc.wait()
    proc.stdout = ExplodingStream()
    with pytest.raises(OSError, match="read failed"):
        capture_process(proc, max_capture=CAP, grace_sec=1.0)


def test_drain_threads_are_tracked_and_pruned():
    track: list = []
    capture_process(sh("echo x"), max_capture=CAP, track=track)
    assert len(track) == 2, "both drains registered for the owner to join"
    for _ in range(4):
        capture_process(sh("echo x"), max_capture=CAP, track=track)
    assert len(track) <= 4, "finished drains are pruned, not accumulated"


def test_close_joins_drains_and_refuses_teardown_while_one_survives():
    """Drives the REAL close()/join_drain_threads() path.

    The previous version of this test joined the threads itself after the writer
    exited, so it would have passed even if close() never joined anything.
    """
    instance = ApptainerInstance.__new__(ApptainerInstance)
    instance.name = None  # skip the instance-stop branch
    instance.overlay = None
    instance.network = None
    instance._instance_stopped = True
    instance.seccomp_path = None
    instance.network_metadata = {"sentinel": True}
    instance.storage_status = {"sentinel": True}

    release = threading.Event()
    started = threading.Event()

    def stubborn():
        started.set()
        release.wait(30)

    thread = threading.Thread(target=stubborn, daemon=True)
    instance._drain_threads = [thread]
    thread.start()
    started.wait(5)

    # close() must refuse while a drain survives, and must NOT clear state.
    with pytest.raises(RuntimeError, match="still running after join"):
        _close_with_short_join(instance)
    assert instance.network_metadata == {
        "sentinel": True
    }, "state must be retained for retry"
    assert instance.storage_status == {"sentinel": True}

    # Once the drain ends, the same call succeeds and clears state.
    release.set()
    thread.join(5)
    _close_with_short_join(instance)
    assert instance.network_metadata == {}
    assert instance.storage_status is None
    assert instance._drain_threads == []


def _close_with_short_join(instance):
    """close() with a short join budget, so the test does not wait 5s twice."""
    original = ApptainerInstance.join_drain_threads

    def quick(self, timeout=0.5):
        return original(self, timeout=timeout)

    ApptainerInstance.join_drain_threads = quick
    try:
        return instance.close()
    finally:
        ApptainerInstance.join_drain_threads = original


def test_stdin_write_errors_are_not_swallowed():
    """Only BrokenPipeError is expected on stdin; other OSErrors must surface."""

    class ExplodingStdin:
        def write(self, _data):
            raise OSError("disk gone")

        def close(self):
            pass

    proc = sh("cat")
    proc.stdin.close()
    proc.stdin = ExplodingStdin()
    with pytest.raises(OSError, match="disk gone"):
        capture_process(proc, stdin=b"x", max_capture=CAP, grace_sec=1.0)
    proc.kill()
