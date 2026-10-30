"""Worker process that owns one Apptainer instance inside a Slurm step."""

from __future__ import annotations

import argparse
import collections
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

from sandfleet.errors import SandboxOOMError, SandfleetError
from sandfleet.environment import validate_command_env
from sandfleet.resources import GPU_CONTAINER_MODES
from sandfleet.mounts import (
    AGENT_PHASE,
    ReadonlyMount,
    check_instance_mounts,
    check_phase_roots,
    check_protected_paths,
    check_target_overlap,
    resolve_mount_roots,
    resolve_mounts,
    seccomp_denies_mounts,
    validate_mount_request,
    validate_phase_roots,
    check_mount_report,
)
from sandfleet.seccomp import (
    default_profile,
    native_architecture,
    write_default_profile,
)
from sandfleet.storage import (
    DISK_START_FLAGS,
    DiskOverlay,
    image_digest,
    overlay_mounts,
    validate_storage,
)
from sandfleet.network import (
    LoopbackProxy,
    forwarder_endpoint,
    sandbox_proxy_url,
    sanitized_apptainer_env,
    validate_network_config,
    validate_network_flags,
)

MAX_OUTPUT_BYTES = 1_000_000
# Time to drain the direct child's output after it exits; background holders of
# the pipe are not waited for.
DRAIN_GRACE_SEC = 5.0
# Cancellation escalates SIGTERM -> SIGKILL on the command's process group only.
CANCEL_TERM_GRACE_SECONDS = 10.0
CANCEL_KILL_GRACE_SECONDS = 5.0
# PID-namespace instances publish the in-container pgid to a pidfile as the
# command's first action; this only needs to cover apptainer exec spin-up.
CANCEL_PIDFILE_WAIT_SECONDS = 5.0
IN_INSTANCE_POLL_SECONDS = 0.1
# kill_processes keeps sending SIGKILL until only the instance's init remains;
# processes still present after this long are reported as a failure.
KILL_PROCESSES_TIMEOUT_SECONDS = 10
# Overall worker-side deadline for one kill-processes call. Every step draws on
# what remains of it: the lifecycle-lock wait (at most
# LIFECYCLE_LOCK_TIMEOUT_SECONDS), the instance PID lookup, the kill+verify loop
# (at most KILL_PROCESSES_TIMEOUT_SECONDS, each helper run capped too) and, on
# failure, the fallback interrupts of in-flight commands. The agent's control
# RPC timeout (step_backend) adds transport headroom on top of this.
KILL_PROCESSES_BUDGET_SECONDS = 75.0
LIFECYCLE_LOCK_TIMEOUT_SECONDS = 30
# _IO(0xb7, 2) from linux/nsfs.h: the namespace this one is nested inside.
NS_GET_PARENT = 0xB702
# Bounds on remembered finished/cancelled command ids.
FINISHED_CACHE_SIZE = 4096
CANCELLED_CACHE_SIZE = 4096
# Exit codes reported for cancelled commands.
CANCELLED_BEFORE_START_EXIT = -1
CANCELLED_SIGNALLED_EXIT = 137  # 128 + SIGKILL
# --no-mount tmp keeps the image's own /tmp and /var/tmp (task setup often leaves
# inputs and verifier answer files there) instead of mounting empty ones over
# them, as disk mode already does. Writes to them go to the writable tmpfs.
DEFAULT_START_FLAGS = (
    "--fakeroot",
    "--writable-tmpfs",
    "--containall",
    "--no-home",
    "--no-mount",
    "tmp",
    "--cleanenv",
)
# apt otherwise drops to the image's _apt user, which fakeroot maps to a host
# subuid that cluster identity services cannot resolve.
APT_CONFIG_SCRIPT = (
    "umask 022 && mkdir -p /etc/apt/apt.conf.d && echo 'APT::Sandbox::User \"root\";'"
    " > /etc/apt/apt.conf.d/99-no-sandbox-user"
)
# TEMPORARY: shadow apt with a stub that exits 100 (apt's error code). On unless the worker's
# environment has SANDFLEET_SANDBOX_BLOCK_APT=0. Best effort: never fails start.
APT_BLOCK_SCRIPT = """
[ -d /usr/bin ] && for b in apt apt-get aptitude; do
  cat > /usr/bin/$b <<'EOF' && chmod 755 /usr/bin/$b
#!/bin/sh
echo 'E: apt is disabled in this sandbox.' >&2
exit 100
EOF
done 2>/dev/null
true"""
# TEMPORARY: any non-root container user maps to a host subuid as well, so also shadow
# su, sudo, runuser, setpriv and chroot (--userspec) with a stub that exits 1. On unless
# SANDFLEET_SANDBOX_BLOCK_USER_SWITCH=0. Removes rather than writes through existing
# entries, which may be symlinks (e.g. to busybox). Best effort: never fails start; other
# setuid binaries and os.setuid() are stopped by DROP_SETUID_CAPS below.
USER_SWITCH_BLOCK_SCRIPT = """
for d in /usr/bin /usr/sbin /bin /sbin; do
  [ -d $d ] && for b in su sudo runuser setpriv chroot; do
    rm -f $d/$b && cat > $d/$b <<'EOF' && chmod 755 $d/$b
#!/bin/sh
echo 'E: switching users is disabled in this sandbox.' >&2
exit 1
EOF
  done
done 2>/dev/null
true"""
# Every exec drops CAP_SETUID and CAP_SETGID from all five capability sets, so no sandbox
# process can switch to a non-root user (and thus a host subuid): setuid()/setgroups() fail
# with EPERM and daemons that drop privileges themselves (nginx, postgres, sshd) cannot.
# On unless SANDFLEET_SANDBOX_DROP_SETUID_CAPS=0. Apptainer applies --drop-caps to exec'd
# processes only; the instance's own init (appinit) keeps every capability and runs no
# sandbox code. CAP_SYS_PTRACE is dropped too, so no sandbox process can ptrace appinit
# (whose caps are not a subset of its own) to regain them. Tracing one's own children
# still works.
DROP_SETUID_CAPS = ("CAP_SETUID", "CAP_SETGID", "CAP_SYS_PTRACE")
# Every exec also drops CAP_NET_ADMIN and CAP_NET_RAW, so no sandbox process can
# reconfigure its network namespace or send raw packets. CAP_SYS_PTRACE goes with them,
# so nothing can ptrace appinit to regain them, whatever
# SANDFLEET_SANDBOX_DROP_SETUID_CAPS says. On unless SANDFLEET_SANDBOX_DROP_NET_CAPS=0,
# which internet leases refuse. The same set is dropped at instance start, which
# Apptainer applies to the %startscript process tree but not to appinit. The worker
# creates an internet lease's proxy listener from outside, via nsenter into the user
# namespace it owns, which these process caps do not affect.
DROP_NET_CAPS = ("CAP_NET_ADMIN", "CAP_NET_RAW", "CAP_SYS_PTRACE")
# Every run_command lowers RLIMIT_NPROC (soft and hard) inside the sandbox to this many
# tasks; SANDFLEET_SANDBOX_PIDS_LIMIT overrides it and 0 disables it. Since Linux 5.14
# the limit is checked against the tasks (threads included) of the caller's uid in its
# own user namespace, and each fakeroot instance has its own, so this caps one sandbox
# rather than every process of the shared host uid. Apptainer's own helpers in the
# instance (roughly 25 threads) count too. Raising the hard limit again needs
# CAP_SYS_RESOURCE in the initial namespace, which no sandbox has.
DEFAULT_PIDS_LIMIT = 4096
# Mounts of these filesystem types inside a sandbox mean host network storage leaked in
# (for example a site `bind path`); startup refuses them.
NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "smbfs",
        "lustre",
        "gpfs",
        "ceph",
        "glusterfs",
        "beegfs",
        "9p",
        "afs",
    }
)
# FUSE filesystems Apptainer itself mounts from local images (fuse2fs mounts ext3
# overlays as fuse.ext4). Any other FUSE mount (sshfs, s3fs, gcsfuse, glusterfs,
# ...) may front remote storage and is refused like a network filesystem.
LOCAL_FUSE_FILESYSTEMS = frozenset(
    {"fuse.fuse-overlayfs", "fuse.squashfuse", "fuse.squashfuse_ll", "fuse.ext4"}
)
NVIDIA_DATA_DEVICE = re.compile(r"nvidia[0-9]+")
CLIENT_GPU_FLAGS = frozenset({"--nv", "--nvccli", "--rocm"})
# Set only for leases that opted into operator-enabled sandbox internet.
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def env_enabled(name: str) -> bool:
    return os.environ.get(name, "1").strip().lower() not in ("0", "false", "no", "off")


def sandbox_pids_limit() -> int:
    raw = os.environ.get("SANDFLEET_SANDBOX_PIDS_LIMIT", str(DEFAULT_PIDS_LIMIT))
    try:
        limit = int(raw)
    except ValueError:
        limit = -1
    if limit < 0:
        raise ValueError(
            f"SANDFLEET_SANDBOX_PIDS_LIMIT must be a nonnegative integer, got {raw!r}"
        )
    return limit


def sandbox_seccomp_profile() -> str | None:
    """Return the operator's profile path, "default", or None when disabled."""
    raw = os.environ.get("SANDFLEET_SANDBOX_SECCOMP", "1").strip()
    if raw.startswith("/"):
        if not os.path.isfile(raw):
            raise ValueError(
                f"SANDFLEET_SANDBOX_SECCOMP profile {raw!r} does not exist"
            )
        return raw
    return "default" if env_enabled("SANDFLEET_SANDBOX_SECCOMP") else None


def network_filesystem_mounts(
    mountinfo: str, allowed_mount_points: frozenset[str] = frozenset()
) -> list[str]:
    """Return 'mountpoint (fstype)' for each network or unrecognized FUSE mount.

    ``allowed_mount_points`` are the lease's operator-allowlisted read-only
    mounts, which may be network storage by design.
    """
    found = []
    for line in mountinfo.splitlines():
        fields, separator, rest = line.partition(" - ")
        if not separator:
            raise RuntimeError(f"Malformed mountinfo line: {line!r}")
        fstype = rest.split()[0]
        fuse = fstype in {"fuse", "fuseblk"} or fstype.startswith("fuse.")
        point = fields.split()[4]
        decoded = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), point)
        if decoded in allowed_mount_points:
            continue
        if (
            fstype in NETWORK_FILESYSTEMS
            or fstype.startswith("nfs")
            or (fuse and fstype not in LOCAL_FUSE_FILESYSTEMS)
        ):
            found.append(f"{point} ({fstype})")
    return found


def resolve_gpu_container_mode(mode: str) -> str:
    """Resolve `auto` to `nvccli` only on hosts where `--nv` cannot work."""
    if mode != "auto":
        return mode
    if not os.path.isdir("/dev/nvidia-caps-imex-channels"):
        return "nv"
    if shutil.which("nvidia-container-cli") is None:
        raise RuntimeError(
            "GPU sandboxes on IMEX hosts need nvidia-container-cli (--nvccli)"
        )
    return "nvccli"


def normalize_image(image: str) -> str:
    if "://" in image:
        return image
    if image.startswith(("/", "./")) or image.endswith(".sif"):
        return image
    return f"docker://{image}"


def accessible_nvidia_devices() -> dict[str, Any]:
    """Functionally probe full-GPU device nodes under the current step cgroup."""

    accessible: list[dict[str, Any]] = []
    denied: list[str] = []
    errors: list[dict[str, Any]] = []
    for path in sorted(Path("/dev").glob("nvidia*")):
        if NVIDIA_DATA_DEVICE.fullmatch(path.name) is None:
            continue
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EPERM}:
                denied.append(str(path))
            else:
                errors.append(
                    {
                        "path": str(path),
                        "errno": error.errno,
                        "message": str(error),
                    }
                )
            continue
        try:
            stat = path.stat()
            accessible.append(
                {
                    "path": str(path),
                    "major": os.major(stat.st_rdev),
                    "minor": os.minor(stat.st_rdev),
                }
            )
        finally:
            os.close(descriptor)
    identities: list[dict[str, Any]] = []
    if accessible:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                errors.append(
                    {
                        "path": "nvidia-smi",
                        "errno": result.returncode,
                        "message": result.stderr.strip() or "nvidia-smi failed",
                    }
                )
            else:
                for line in result.stdout.splitlines():
                    index, uuid_value, name, memory_mb = (
                        value.strip() for value in line.split(",", 3)
                    )
                    identities.append(
                        {
                            "index": int(index),
                            "uuid": uuid_value,
                            "name": name,
                            "memory_mb": int(memory_mb),
                        }
                    )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append({"path": "nvidia-smi", "errno": None, "message": str(error)})
    return {
        "accessible": accessible,
        "identities": identities,
        "denied": denied,
        "errors": errors,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
    }


def current_cgroup(*, include_gpu: bool = False) -> dict[str, Any]:
    lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    unified = [line.partition("::")[2] for line in lines if "::" in line]
    if len(unified) != 1 or not unified[0]:
        raise RuntimeError("Sandfleet requires one unified cgroup v2 hierarchy")

    root = Path("/sys/fs/cgroup")
    path = root / unified[0].lstrip("/")
    metadata = {
        "path": str(path),
        "memory.max": (path / "memory.max").read_text(encoding="utf-8").strip(),
        "memory.current": (path / "memory.current").read_text(encoding="utf-8").strip(),
        "cpuset.cpus.effective": (path / "cpuset.cpus.effective")
        .read_text(encoding="utf-8")
        .strip(),
    }

    limits: list[tuple[int, Path]] = []
    cursor = path
    while cursor == root or root in cursor.parents:
        limit_file = cursor / "memory.max"
        if cursor == root and not limit_file.exists():
            break
        raw = limit_file.read_text(encoding="utf-8").strip()
        if raw != "max":
            limits.append((int(raw), cursor))
        if cursor == root:
            break
        cursor = cursor.parent
    if not limits:
        raise RuntimeError("Slurm step has no finite cgroup memory limit")
    limit, limit_path = min(limits, key=lambda item: item[0])
    metadata["memory.max.effective"] = str(limit)
    metadata["memory.max.effective_path"] = str(limit_path)
    if include_gpu:
        metadata["gpu"] = accessible_nvidia_devices()
    return metadata


def memory_events() -> dict[str, int]:
    metadata = current_cgroup()
    path = Path(metadata["memory.max.effective_path"])
    lines = (path / "memory.events").read_text(encoding="utf-8").splitlines()
    return {key: int(raw) for key, raw in (line.split() for line in lines)}


def _drain_stream(stream, state: dict, key: str, max_capture: int | None) -> None:
    """Read one stream to EOF, retaining at most `max_capture` bytes.

    Data beyond the limit is discarded, so a runaway writer costs no memory and
    never blocks on a full pipe.
    """
    buffer = state[key]
    try:
        while True:
            block = stream.read1(65536)
            if not block:
                break
            if state["detached"]:
                continue
            if max_capture is None:
                buffer.extend(block)
            elif len(buffer) < max_capture:
                buffer.extend(block[: max_capture - len(buffer)])
    except BaseException as exc:  # noqa: BLE001 - re-raised by capture_process
        state["error"] = exc
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _join_output_threads(threads, max_capture, grace_sec):
    if max_capture is None:
        for thread in threads:
            thread.join()
    else:
        deadline = time.monotonic() + grace_sec
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))


def capture_process(
    proc: subprocess.Popen,
    *,
    stdin: bytes | None = None,
    max_capture: int | None = None,
    grace_sec: float = DRAIN_GRACE_SEC,
    track: list | None = None,
    on_complete: Callable[[], None] | None = None,
) -> subprocess.CompletedProcess:
    """Run a started process to completion without waiting on inherited pipes.

    CONTRACT

    * Waits on the DIRECT child only. A process the command backgrounded keeps
      running and is not killed; it simply no longer holds the request open.
      `subprocess.run(capture_output=True)` cannot be used here because
      `communicate()` returns at EOF on both streams rather than at child exit,
      so a surviving pipe holder blocks it indefinitely.
    * Reads with `read1()`. `BufferedReader.read(n)` waits for n bytes or EOF, so
      short output plus a live background holder would return nothing.
    * `max_capture=None` retains everything and joins WITHOUT a timeout, which is
      what file transfer requires: partial data would be silent corruption. This
      exec's pipes are its own, so waiting for real EOF cannot hang on a process
      backgrounded by an earlier exec.
    * A non-None `max_capture` retains at most that many bytes, discards the
      rest, and joins against a single shared deadline. Only this path can
      truncate, and only it is used for command output.
    * Capture errors are raised, never returned as a successful partial result.
    * Buffers are detached on return, so a still-running drain can neither mutate
      returned data nor keep growing. Threads are appended to `track` if given,
      so the owner can join them at teardown.
    """
    state: dict[str, Any] = {
        "stdout": bytearray(),
        "stderr": bytearray(),
        "error": None,
        "detached": False,
    }

    threads = [
        threading.Thread(
            target=_drain_stream,
            args=(proc.stdout, state, "stdout", max_capture),
            name=f"sfdrain-out-{proc.pid}",
            daemon=True,
        ),
        threading.Thread(
            target=_drain_stream,
            args=(proc.stderr, state, "stderr", max_capture),
            name=f"sfdrain-err-{proc.pid}",
            daemon=True,
        ),
    ]
    if track is not None:
        track[:] = [thread for thread in track if thread.is_alive()]
        track.extend(threads)
    for thread in threads:
        thread.start()
    try:
        if stdin:
            proc.stdin.write(stdin)
        if proc.stdin is not None:
            proc.stdin.close()
    except BrokenPipeError:
        # The child closed stdin early; that is normal and not an error. Any
        # other OSError is unexpected and must stay visible.
        pass
    proc.wait()
    # Foreground completion precedes draining pipes inherited by background work.
    if on_complete is not None:
        on_complete()
    _join_output_threads(threads, max_capture, grace_sec)
    if state["error"] is not None:
        raise state["error"]
    stdout, stderr = bytes(state["stdout"]), bytes(state["stderr"])
    state["detached"] = True
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)


def _process_might_be_ours(pid: int) -> bool:
    """Whether a process we were denied access to could be one of our sandboxes.

    Fakeroot maps a sandbox's processes to our host UID, so a process showing our
    UID (or one we cannot inspect) must not be silently skipped. A process that
    has since disappeared is not a member and is skipped.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, ProcessLookupError):
        return False  # gone between the failed open and here: not a member
    except OSError:
        return True  # cannot inspect: fail closed
    for line in status.splitlines():
        if line.startswith("Uid:"):
            return os.getuid() in {int(value) for value in line.split()[1:]}
    return True  # no Uid line: fail closed


def pid_in_namespace(pid: int, target_inode: int) -> bool:
    """Whether ``pid``'s PID namespace is ``target_inode`` or nested inside it.

    Walks parent namespaces (``NS_GET_PARENT``) from the process toward the
    host; a host or sibling-sandbox process reaches the top (``EPERM``) without
    passing through the target. Fails closed: a process that has merely gone is
    excluded, but a permission error on a process that might be ours, or any
    other error, is raised rather than silently treated as "not a member".
    """
    try:
        descriptor = os.open(f"/proc/{pid}/ns/pid", os.O_RDONLY | os.O_CLOEXEC)
    except (FileNotFoundError, ProcessLookupError):
        return False  # the process is gone
    except PermissionError as error:
        if _process_might_be_ours(pid):
            raise RuntimeError(
                f"Cannot classify PID {pid} that may belong to the sandbox: {error}"
            ) from error
        return False  # a process of another UID: not one of ours
    seen: set[int] = set()
    try:
        while True:
            inode = os.fstat(descriptor).st_ino
            if inode == target_inode:
                return True
            if inode in seen:
                return False
            seen.add(inode)
            try:
                parent = fcntl.ioctl(descriptor, NS_GET_PARENT)
            except PermissionError:
                return False  # reached the top of the accessible hierarchy
            os.close(descriptor)
            descriptor = parent
    finally:
        os.close(descriptor)


def _namespace_member_pids(instance_pid: int, target_inode: int) -> list[int]:
    """Host PIDs whose PID namespace is the instance's or a descendant of it."""
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == instance_pid:
            continue  # never the instance's own init
        if pid_in_namespace(pid, target_inode):
            members.append(pid)
    return members


def _needs_killing(pid: int) -> bool:
    """A process still worth killing: not a plain reapable zombie.

    A zombie thread-group leader whose other threads are still alive (State ``Z``
    with ``Threads`` > 1, e.g. after ``pthread_exit`` in the main thread) is not
    gone; a single-threaded zombie is dead and its parent will reap it.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return False
    state = threads = None
    for line in status.splitlines():
        if line.startswith("State:"):
            state = line.split(None, 2)[1]
        elif line.startswith("Threads:"):
            threads = int(line.split()[1])
    if state is None or threads is None:
        return False
    return not (state == "Z" and threads <= 1)


def _ensure_init_alive(init_pidfd: int) -> None:
    try:
        signal.pidfd_send_signal(init_pidfd, 0)
    except ProcessLookupError as error:
        raise RuntimeError(
            "Sandbox init exited during the kill; aborting rather than risk "
            "signalling a reused PID namespace"
        ) from error


# Helper that joins the instance's user+PID namespaces and SIGKILLs everything
# in them. Spawned with `python -I -S` (isolated, no site, no env) directly by
# the worker, so no nsenter dependency and no image code runs. It setns the user
# namespace (owner UID -> full caps), then the PID namespace (joined by children
# created after setns), drops dumpability to deny ptrace from a host-mount-ns
# process, then forks. The child verifies it is actually in the pinned PID
# namespace (st_dev/st_ino from the pinned fd) and not the worker's own, then
# kill(-1, SIGKILL). The kernel's kill(-1) skips the caller and PID 1, so the
# child survives and exits 0; only a clean exit 0 (child and helper) is success.
# Failures write a reason to stderr; the helper passes the child's exit code on.
_NAMESPACE_KILL_PROGRAM = (
    "import os, ctypes, sys\n"
    "CLONE_NEWUSER = 0x10000000\n"
    "CLONE_NEWPID = 0x20000000\n"
    "PR_SET_DUMPABLE = 4\n"
    "def fail(code, reason):\n"
    "    os.write(2, (reason + '\\n').encode())\n"
    "    os._exit(code)\n"
    "user_fd, pid_fd = int(sys.argv[1]), int(sys.argv[2])\n"
    "tdev, tino, wdev, wino = (int(x) for x in sys.argv[3:7])\n"
    "os.setns(user_fd, CLONE_NEWUSER)\n"
    "os.setns(pid_fd, CLONE_NEWPID)\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:\n"
    "    fail(4, 'prctl(PR_SET_DUMPABLE, 0) failed: errno %d' % ctypes.get_errno())\n"
    "pid = os.fork()\n"
    "if pid == 0:\n"
    "    st = os.stat('/proc/self/ns/pid')\n"
    "    here = (st.st_dev, st.st_ino)\n"
    "    if here != (tdev, tino):\n"
    "        fail(3, 'child is not in the pinned PID namespace: %r' % (here,))\n"
    "    if here == (wdev, wino):\n"
    "        fail(3, 'child is in the worker PID namespace')\n"
    "    try:\n"
    "        os.kill(-1, 9)\n"
    "    except ProcessLookupError:\n"
    "        pass\n"
    "    os._exit(0)\n"
    "_, status = os.waitpid(pid, 0)\n"
    "if os.WIFEXITED(status):\n"
    "    os._exit(os.WEXITSTATUS(status))\n"
    "fail(5, 'kill child died from signal %d' % os.WTERMSIG(status))\n"
)


def _run_namespace_kill(
    user_ns_fd: int,
    pid_ns_fd: int,
    target: tuple[int, int],
    worker: tuple[int, int],
    timeout: float,
) -> None:
    """Run the namespace-kill helper once over the pinned descriptors.

    The helper runs in its own session, so on timeout the whole group is
    SIGKILLed: the forked child sits in the sandbox's PID namespace, where the
    sandbox could SIGSTOP it, and subprocess's own timeout would kill only the
    helper. SIGKILL also ends a stopped process.
    """
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _NAMESPACE_KILL_PROGRAM,
            str(user_ns_fd),
            str(pid_ns_fd),
            str(target[0]),
            str(target[1]),
            str(worker[0]),
            str(worker[1]),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(user_ns_fd, pid_ns_fd),
        start_new_session=True,
    )
    try:
        _, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError(
            f"Namespace kill helper timed out after {timeout:.1f}s"
        ) from error
    # The kernel's kill(-1) skips the caller, so the helper and its child exit 0
    # on success; any other code means it could not join or verify the namespace.
    if process.returncode != 0:
        detail = (
            stderr.decode("utf-8", errors="replace").strip()
            or f"exit code {process.returncode}"
        )
        raise RuntimeError(f"Namespace kill helper failed: {detail}")


# Helper that reports, for each target path, what the sandbox sees there. The
# overlay is a FUSE mount owned by the instance's user namespace, and FUSE refuses
# callers outside it, so the helper joins that namespace (only it: not the mount
# namespace, so the worker's own interpreter runs, never image code). It walks
# each target under the instance root the worker opened (/proc/<init>/root) one
# component at a time with O_NOFOLLOW | O_DIRECTORY, so a symlink anywhere fails
# instead of resolving, and prints one JSON line per target.
_MOUNT_REPORT_PROGRAM = (
    "import json, os, sys\n"
    "CLONE_NEWUSER = 0x10000000\n"
    "os.setns(int(sys.argv[1]), CLONE_NEWUSER)\n"
    "root = int(sys.argv[2])\n"
    "flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC\n"
    "for target in sys.argv[3:]:\n"
    "    fd = os.dup(root)\n"
    "    try:\n"
    "        for part in target.strip('/').split('/'):\n"
    "            following = os.open(part, flags, dir_fd=fd)\n"
    "            os.close(fd)\n"
    "            fd = following\n"
    "        st = os.fstat(fd)\n"
    "        read_only = bool(os.fstatvfs(fd).f_flag & os.ST_RDONLY)\n"
    "        report = {'target': target, 'dev': st.st_dev, 'ino': st.st_ino,\n"
    "                  'read_only': read_only}\n"
    "    except OSError as error:\n"
    "        report = {'target': target, 'error': str(error)}\n"
    "    finally:\n"
    "        os.close(fd)\n"
    "    print(json.dumps(report), flush=True)\n"
)
MOUNT_REPORT_TIMEOUT_SECONDS = 30


def read_mountinfo(pid: int) -> str:
    with open(f"/proc/{pid}/mountinfo", encoding="utf-8") as handle:
        return handle.read()


def report_mounts(user_ns_fd: int, root_fd: int, targets: list[str]) -> list[dict]:
    """Run the mount-report helper over the pinned user namespace and root."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _MOUNT_REPORT_PROGRAM,
            str(user_ns_fd),
            str(root_fd),
            *targets,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=(user_ns_fd, root_fd),
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=MOUNT_REPORT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise RuntimeError("Mount report helper timed out") from error
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Mount report helper failed: {detail or process.returncode}"
        )
    reports = [json.loads(line) for line in stdout.decode().splitlines()]
    if len(reports) != len(targets):
        raise RuntimeError("Mount report helper returned the wrong number of reports")
    return reports


def kill_namespace_processes(
    init_pid: int, user_ns_fd: int, pid_ns_fd: int, timeout: float
) -> int:
    """SIGKILL every process in the instance's PID namespace except its init.

    Runs the namespace-kill helper (atomic ``kill(-1)`` inside the pinned
    namespaces), then verifies with a fresh host-side scan, looping until a scan
    finds nothing that needs killing. Aborts if the init dies mid-loop. ``timeout``
    bounds the whole loop, and each helper run is capped by the remaining budget.
    Returns the number of distinct processes seen; raises if any survive.
    """
    stat = os.fstat(pid_ns_fd)
    target = (stat.st_dev, stat.st_ino)
    worker = _worker_pid_namespace_identity()
    if target == worker:
        raise RuntimeError("Pinned PID namespace is the worker's; refusing to kill")
    if _live_pid_namespace_identity(init_pid) != target:
        raise RuntimeError(
            "Sandbox init no longer owns the pinned PID namespace; refusing to kill"
        )
    init_pidfd = os.pidfd_open(init_pid)
    try:
        if _live_pid_namespace_identity(init_pid) != target:
            raise RuntimeError(
                "Sandbox init PID changed while pinning; refusing to kill"
            )
        deadline = time.monotonic() + timeout
        killed: set[int] = set()
        while True:
            _ensure_init_alive(init_pidfd)
            killed.update(_namespace_member_pids(init_pid, stat.st_ino))
            remaining = deadline - time.monotonic()
            run_timeout = min(KILL_PROCESSES_TIMEOUT_SECONDS, max(0.1, remaining))
            _run_namespace_kill(user_ns_fd, pid_ns_fd, target, worker, run_timeout)
            _ensure_init_alive(init_pidfd)
            members = _namespace_member_pids(init_pid, stat.st_ino)
            killed.update(members)
            survivors = [pid for pid in members if _needs_killing(pid)]
            if not survivors:
                return len(killed)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Sandbox processes survived SIGKILL: {sorted(survivors)}"
                )
            time.sleep(0.05)
    finally:
        os.close(init_pidfd)


def _budget(deadline: float, cap: float) -> float:
    """Seconds a step may take: ``cap`` limited by what remains until ``deadline``.

    Raises once the deadline has passed, so no step starts with no budget left.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("kill-processes ran out of its time budget")
    return min(cap, remaining)


def _worker_pid_namespace_identity() -> tuple[int, int]:
    stat = os.stat("/proc/self/ns/pid")
    return (stat.st_dev, stat.st_ino)


def _live_pid_namespace_identity(pid: int) -> tuple[int, int]:
    try:
        stat = os.stat(f"/proc/{pid}/ns/pid")
    except OSError as error:
        raise RuntimeError(f"Sandbox init {pid} is gone: {error}") from error
    return (stat.st_dev, stat.st_ino)


class ApptainerInstance:
    def __init__(
        self,
        *,
        image: str,
        timeout: int,
        pwd: str,
        cache_dir: str | None,
        tmp_dir: str | None,
        extra_start_flags: list[str],
        apptainer_binary: str,
        gpu_enabled: bool = False,
        storage_mb: int | None = None,
        image_sha256: str | None = None,
        storage_root: str | None = None,
        network_mode: str = "loopback",
        gpu_container_mode: str = "auto",
        proxy_url: str | None = None,
        relay_max_pending: int = 1,
        readonly_mounts: list[ReadonlyMount] | tuple[ReadonlyMount, ...] = (),
    ):
        validate_network_config(network_mode)
        validate_network_flags(extra_start_flags)
        self.pids_limit = sandbox_pids_limit()
        self.seccomp = self._seccomp_profile(extra_start_flags)
        self.seccomp_path: str | None = None
        self.pwd = pwd
        # Operator-allowlisted read-only binds: the start request's agent-phase
        # mounts, plus any add_readonly_mounts() adds later. Every later restart
        # of this instance keeps them all.
        self.readonly_mounts = self._checked_start_mounts(list(readonly_mounts))
        if proxy_url is not None and not env_enabled("SANDFLEET_SANDBOX_DROP_NET_CAPS"):
            raise ValueError(
                "Sandbox internet requires SANDFLEET_SANDBOX_DROP_NET_CAPS: with "
                "CAP_NET_ADMIN a sandbox could reconfigure its network namespace"
            )
        self.image = normalize_image(image)
        self.timeout = timeout
        # Drain threads from capture_process(), joined at teardown.
        self._drain_threads: list[threading.Thread] = []
        self.cache_dir = cache_dir
        self.tmp_dir = tmp_dir
        if any(
            any(
                flag == name or flag.startswith(f"{name}=") for name in CLIENT_GPU_FLAGS
            )
            or flag == "--device"
            or flag.startswith("--device=")
            for flag in extra_start_flags
        ):
            raise ValueError(
                "GPU container flags are controlled by the Sandfleet worker"
            )
        if gpu_container_mode not in GPU_CONTAINER_MODES:
            raise ValueError(
                f"gpu_container_mode must be one of {GPU_CONTAINER_MODES!r}"
            )
        if gpu_enabled:
            gpu_container_mode = resolve_gpu_container_mode(gpu_container_mode)
        self.gpu_container_mode = gpu_container_mode
        if type(relay_max_pending) is not int or relay_max_pending < 1:
            raise ValueError("relay_max_pending must be a positive integer")
        self.relay_max_pending = relay_max_pending
        self.gpu_flags = (f"--{gpu_container_mode}",) if gpu_enabled else ()
        self.network_mode = network_mode
        # Both kinds of lease get only loopback, with no routes or DNS. Internet
        # leases also get a proxy listener on 127.0.0.1 relayed to the forwarder.
        self.sandbox_proxy_url = (
            None if proxy_url is None else sandbox_proxy_url(proxy_url)
        )
        self.network_files: dict[str, str] = {}
        self.network_flags = ("--net", "--network=none")
        if env_enabled("SANDFLEET_SANDBOX_DROP_NET_CAPS"):
            self.network_flags += ("--drop-caps", ",".join(DROP_NET_CAPS))
        self.start_flags = (
            DEFAULT_START_FLAGS
            + self.gpu_flags
            + tuple(extra_start_flags)
            + self.network_flags
            # Writable-tmpfs mode only; disk mode builds its own flags.
            + tuple(
                flag for mount in self.readonly_mounts for flag in mount.bind_flag()
            )
        )
        self.apptainer = apptainer_binary
        self.gpu_enabled = gpu_enabled
        self.name: str | None = None
        self.storage_mb = storage_mb
        self.image_sha256 = image_sha256
        self.storage_status: dict[str, Any] | None = None
        self.overlay = None
        if storage_mb is not None:
            if not storage_root or extra_start_flags:
                raise ValueError(
                    "Disk mode requires operator storage_root and forbids extra_start_flags"
                )
            image_digest(self.image, image_sha256)
            self.overlay = DiskOverlay(storage_root, storage_mb, self.apptainer)
        self._instance_stopped = False
        # The instance's own user and PID namespaces, pinned for its lifetime so
        # kill-processes can join them (and compare inodes) even if the instance
        # stops mid-kill: the fds keep the namespace objects alive, so their
        # inodes cannot be reused by another sandbox while we hold them.
        self.pid_ns_fd: int | None = None
        self.user_ns_fd: int | None = None
        self.upstream = None if proxy_url is None else forwarder_endpoint(proxy_url)
        # A fresh relay per started instance (a closed relay cannot restart).
        self.network: LoopbackProxy | None = None
        self.network_metadata: dict[str, Any] = {}

    @staticmethod
    def _seccomp_profile(extra_start_flags: list[str]) -> str | None:
        profile = sandbox_seccomp_profile()
        if profile is not None and any(
            flag == "--security" or flag.startswith("--security=")
            for flag in extra_start_flags
        ):
            raise ValueError("Sandfleet manages the sandbox seccomp profile")
        if profile == "default":
            native_architecture()  # unsupported hosts fail before any start
        return profile

    def env(self, *, pwd: str | None = None) -> dict[str, str]:
        value = (
            sanitized_apptainer_env(os.environ)
            if self.overlay is None
            else self.overlay.env()
        )
        value["APPTAINER_PWD"] = self.pwd if pwd is None else pwd
        if self.cache_dir and self.overlay is None:
            value["APPTAINER_CACHEDIR"] = self.cache_dir
        if self.tmp_dir and self.overlay is None:
            value["APPTAINER_TMPDIR"] = self.tmp_dir
        if self.gpu_enabled:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if not visible:
                raise RuntimeError(
                    "GPU sandbox step has no CUDA_VISIBLE_DEVICES allocation"
                )
            value["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = visible
            if self.gpu_container_mode == "nvccli":
                # Under --containall, nvccli binds only these devices (none if unset).
                value["NVIDIA_VISIBLE_DEVICES"] = visible
            value["APPTAINERENV_CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if self.sandbox_proxy_url is not None:
            for name in PROXY_ENV_NAMES:
                value[f"APPTAINERENV_{name}"] = self.sandbox_proxy_url
            # The sandbox's own addresses must bypass the proxy (it refuses them).
            value["APPTAINERENV_NO_PROXY"] = value["APPTAINERENV_no_proxy"] = ",".join(
                (
                    "localhost",
                    "127.0.0.1",
                    "::1",
                    "0.0.0.0",
                    socket.gethostname(),
                )
            )
        return value

    def start(self) -> None:
        if shutil.which(self.apptainer) is None:
            raise RuntimeError(f"Apptainer binary {self.apptainer!r} is not available")
        self.close()
        self._launch(reuse_overlay=False)

    def _launch(self, *, reuse_overlay: bool) -> None:
        """Start a stopped instance; ``reuse_overlay`` keeps the existing disk."""
        if self.overlay is not None:
            self.start_flags = self._disk_start_flags(reuse=reuse_overlay)
        name = f"sandfleet-{os.getpid()}-{uuid.uuid4().hex[:10]}"
        command = [
            self.apptainer,
            "instance",
            "start",
            *self.start_flags,
            *self._security_flags(),
            self.image,
            name,
        ]
        # Keep the attempted identity even on failure, so cleanup can be retried.
        self.name = name
        self._instance_stopped = False
        result = subprocess.run(
            command, capture_output=True, env=self.env(pwd="/"), timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"apptainer instance start failed (exit={result.returncode}): "
                f"{result.stderr.decode('utf-8', 'replace').strip()}"
            )
        try:
            self._start_network()
            self._bootstrap()
            if self.readonly_mounts:
                self._verify_readonly_mounts()
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"Sandbox cleanup also failed: {cleanup_error}")
            raise

    def _security_flags(self) -> tuple[str, ...]:
        if self.seccomp is None:
            return ()
        if self.seccomp != "default":
            return ("--security", f"seccomp:{self.seccomp}")
        if self.seccomp_path is None:
            self.seccomp_path = write_default_profile()
        return ("--security", f"seccomp:{self.seccomp_path}")

    def _disk_start_flags(self, *, reuse: bool = False) -> tuple[str, ...]:
        if reuse:
            image = self.overlay.image
            if image is None or not image.is_file():
                raise RuntimeError(
                    "The sandbox's disk overlay is missing; cannot restart"
                )
        else:
            # A persistent overlay needs every bind destination to exist already.
            image = self.overlay.create(
                create_dirs=[mount.target for mount in self.readonly_mounts]
            )
        # --no-mount bind-paths also suppresses generated resolver/hosts files.
        # Keep host auto-binds disabled; only these owned files are bound read-only.
        # Sandboxes have no route to these resolvers; this matches the file
        # Apptainer binds outside disk mode.
        self.network_files = {
            "resolv.conf": Path("/etc/resolv.conf").read_text(),
            "hosts": "127.0.0.1 localhost\n::1 localhost\n",
        }
        binds: tuple[str, ...] = ()
        for name, content in self.network_files.items():
            path = image.with_name(name)
            path.write_text(content)
            # Services may drop root inside the sandbox and still need NSS/DNS.
            # The backing directory stays private; the sandbox mount is read-only.
            path.chmod(0o644)
            binds += ("--bind", f"{path}:/etc/{name}:ro")
        for mount in self.readonly_mounts:
            binds += mount.bind_flag()
        return (
            DISK_START_FLAGS
            + self.gpu_flags
            + ("--overlay", str(image))
            + binds
            + self.network_flags
        )

    def _instance_pid(self, timeout: float = 10) -> int:
        result = subprocess.run(
            [self.apptainer, "instance", "list", "--json", self.name],
            capture_output=True,
            env=self.env(),
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not discover sandbox network namespace: "
                + result.stderr.decode("utf-8", "replace")
            )
        instances = json.loads(result.stdout)["instances"]
        matches = [item for item in instances if item["instance"] == self.name]
        if (
            len(matches) != 1
            or type(matches[0]["pid"]) is not int
            or matches[0]["pid"] <= 0
        ):
            raise RuntimeError("Apptainer returned no unique live sandbox PID")
        return matches[0]["pid"]

    def _start_network(self) -> None:
        pid = self._instance_pid()
        host_namespace = os.readlink("/proc/self/ns/net")
        namespace = os.readlink(f"/proc/{pid}/ns/net")
        if namespace == host_namespace:
            raise RuntimeError("Sandbox is sharing the worker network namespace")
        self._pin_namespaces(pid)
        # Apptainer brings up loopback in every namespace it creates; offline
        # instances get nothing else.
        if self.upstream is not None:
            self.network = LoopbackProxy(
                self.upstream, max_pending=self.relay_max_pending
            )
            self.network.start(pid, namespace)
        self._verify_sandbox(namespace)
        self.network_metadata = {
            "mode": "none" if self.network is None else self.network_mode,
            "namespace": namespace,
            "host_namespace": host_namespace,
            "helper_pid": None,
            "internet": self.network is not None,
        }

    @staticmethod
    def _namespace_inode(path: str) -> int:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            return os.fstat(descriptor).st_ino
        finally:
            os.close(descriptor)

    def _pin_namespaces(self, pid: int) -> None:
        # pidfd_open first, so the pidfd_send_signal(0) below proves the ns fds we
        # opened by path belong to the same still-live init (defeats PID reuse
        # between reading the PID and opening the fds).
        init_pidfd = os.pidfd_open(pid)
        pid_fd = user_fd = None
        try:
            worker_pid_inode = self._namespace_inode("/proc/self/ns/pid")
            worker_user_inode = self._namespace_inode("/proc/self/ns/user")
            pid_fd = os.open(f"/proc/{pid}/ns/pid", os.O_RDONLY | os.O_CLOEXEC)
            user_fd = os.open(f"/proc/{pid}/ns/user", os.O_RDONLY | os.O_CLOEXEC)
            signal.pidfd_send_signal(init_pidfd, 0)
            if os.fstat(pid_fd).st_ino == worker_pid_inode:
                raise RuntimeError("Sandbox is sharing the worker PID namespace")
            if os.fstat(user_fd).st_ino == worker_user_inode:
                raise RuntimeError("Sandbox is sharing the worker user namespace")
        except BaseException:
            for descriptor in (pid_fd, user_fd):
                if descriptor is not None:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
            raise
        finally:
            os.close(init_pidfd)
        self.pid_ns_fd = pid_fd
        self.user_ns_fd = user_fd

    def _close_namespace_fds(self) -> None:
        for attribute in ("pid_ns_fd", "user_ns_fd"):
            descriptor = getattr(self, attribute, None)
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                setattr(self, attribute, None)

    def _verify_sandbox(self, expected_namespace: str) -> None:
        """Check network, seccomp and mounts from one real instance exec."""
        split = "--sandfleet-verify--"
        result = self._exec_internal(
            [
                "sh",
                "-c",
                "readlink /proc/self/ns/net && cat /proc/net/route && "
                f"echo {split} && cat /proc/self/status && "
                f"echo {split} && cat /proc/self/mountinfo && "
                f"echo {split} && readlink /proc/self/ns/pid",
            ],
            pwd="/",
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not verify sandbox network and confinement: "
                + result.stderr.decode("utf-8", "replace")
            )
        sections = result.stdout.decode("utf-8").split(f"{split}\n")
        if len(sections) != 4:
            raise RuntimeError("Malformed sandbox verification output")
        network, status, mountinfo, pid_namespace = sections
        self._verify_network(expected_namespace, network)
        self._verify_confinement(status, mountinfo)
        self._verify_pid_namespace(pid_namespace.strip())

    def _verify_pid_namespace(self, link: str) -> None:
        # Commands must run in the very PID namespace kill-processes pinned.
        expected = f"pid:[{os.fstat(self.pid_ns_fd).st_ino}]"
        if link != expected:
            raise RuntimeError(
                f"Sandbox exec runs in PID namespace {link!r}, not the pinned "
                f"{expected!r}; refusing to start"
            )

    def _verify_confinement(self, status: str, mountinfo: str) -> None:
        if self.seccomp is not None and "\nSeccomp:\t2\n" not in f"\n{status}":
            raise RuntimeError(
                "Sandbox commands are not under the seccomp filter; refusing to start"
            )
        leaked = network_filesystem_mounts(
            mountinfo, frozenset(mount.target for mount in self.readonly_mounts)
        )
        if leaked:
            raise RuntimeError(
                f"Host network or unrecognized FUSE filesystems are visible in the "
                f"sandbox: {leaked}; remove them from the site's Apptainer bind paths"
            )

    def _verify_network(self, expected_namespace: str, output: str) -> None:
        lines = output.splitlines()
        if (
            not lines
            or not re.fullmatch(r"net:\[\d+\]", lines[0])
            or lines[0] != expected_namespace
        ):
            raise RuntimeError(
                "Sandbox exec did not join its private network namespace"
            )
        routes = [line for line in lines[2:] if line.strip()]
        # Offline and internet leases alike: loopback only, no routes at all.
        if routes:
            raise RuntimeError(
                f"Sandbox has IPv4 routes {routes!r}; the policy allows none. "
                "Refusing a sandbox that could reach the network directly"
            )

    def _bootstrap(self) -> None:
        script = APT_CONFIG_SCRIPT + " || exit"
        for name, block in (
            ("SANDFLEET_SANDBOX_BLOCK_APT", APT_BLOCK_SCRIPT),
            ("SANDFLEET_SANDBOX_BLOCK_USER_SWITCH", USER_SWITCH_BLOCK_SCRIPT),
        ):
            if env_enabled(name):
                script += block
        apt = subprocess.run(
            [
                self.apptainer,
                "exec",
                *self.exec_flags,
                f"instance://{self.name}",
                "sh",
                "-c",
                script,
            ],
            capture_output=True,
            env=self.env(pwd="/"),
        )
        if apt.returncode != 0:
            raise RuntimeError(
                "could not configure apt: "
                f"{apt.stderr.decode('utf-8', 'replace').strip()}"
            )
        if self.pwd != "/":
            bootstrap = subprocess.run(
                [
                    self.apptainer,
                    "exec",
                    *self.exec_flags,
                    f"instance://{self.name}",
                    "mkdir",
                    "-p",
                    self.pwd,
                ],
                capture_output=True,
                env=self.env(pwd="/"),
            )
            if bootstrap.returncode != 0:
                raise RuntimeError(
                    f"could not create working directory {self.pwd!r}: "
                    f"{bootstrap.stderr.decode('utf-8', 'replace').strip()}"
                )
        if self.overlay is not None:
            self._verify_storage()

    def _verify_storage(self) -> None:
        for name, content in self.network_files.items():
            result = self.exec(["cat", f"/etc/{name}"])
            if result.returncode or result.stdout.decode() != content:
                raise RuntimeError(f"Sandbox did not receive its owned /etc/{name}")
        paths = ["/", "/tmp", "/var/tmp", "/root", self.pwd]
        resolved = self.exec(["readlink", "-f", "--", *paths])
        mounts = self.exec(["cat", "/proc/self/mountinfo"])
        stats = self.exec(["stat", "-f", "-c", "%S %b %a", "/"])
        if any(item.returncode for item in (resolved, mounts, stats)):
            raise RuntimeError("Cannot verify disk overlay mounts/capacity")
        resolved_paths = resolved.stdout.decode().splitlines()
        if len(resolved_paths) != len(paths):
            raise RuntimeError("Invalid resolved writable paths")
        overlay_mounts(mounts.stdout.decode(), resolved_paths)
        block, total, available = map(int, stats.stdout.split())
        self.storage_status = {
            "mode": "disk-overlay",
            "storage_mb": self.storage_mb,
            "policy": "nominal-image-mib",
            "image_sha256": self.image_sha256,
            "total_bytes": block * total,
            "available_bytes_at_start": block * available,
            "writable_paths_verified": True,
        }

    def close(self) -> None:
        # The instance is being torn down and kill-processes is serialized out by
        # the worker's lifecycle lock, so releasing the pinned namespaces here is
        # safe and idempotent.
        self._close_namespace_fds()
        if self.name is not None:
            if not self._instance_stopped:
                if self.overlay is None or any(
                    item["instance"] == self.name for item in self.exec_instance_list()
                ):
                    self._stop_instance()
                self._instance_stopped = True
            if self.network is not None:
                self.network.close()
                self.network = None
        if self.seccomp_path is not None and self._instance_stopped:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.seccomp_path)
            self.seccomp_path = None
        # Stopping the instance closes its descriptors, so remaining drains reach
        # EOF. If any are still alive, keep the overlay and state for a retry.
        alive = self.join_drain_threads()
        if alive:
            raise RuntimeError(
                f"{len(alive)} capture drain thread(s) still running after join; "
                "sandbox state retained so cleanup can be retried"
            )
        if self.overlay is not None:
            self.overlay.remove()
        self.name = None
        self.network_metadata = {}
        self.storage_status = None

    def join_drain_threads(self, timeout: float = DRAIN_GRACE_SEC) -> list:
        """Join capture drains and return any that are still alive."""
        deadline = time.monotonic() + timeout
        for thread in self._drain_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        alive = [thread for thread in self._drain_threads if thread.is_alive()]
        self._drain_threads = alive
        return alive

    def exec_instance_list(self, *, timeout: float = 30) -> list[dict[str, Any]]:
        result = subprocess.run(
            [self.apptainer, "instance", "list", "--json"],
            capture_output=True,
            env=self.env(),
            timeout=timeout,
            check=True,
        )
        instances = json.loads(result.stdout)["instances"]
        if not isinstance(instances, list) or any(
            not isinstance(item, dict)
            or not isinstance(item.get("instance"), str)
            or not item["instance"]
            for item in instances
        ):
            raise RuntimeError("Malformed Apptainer instance listing")
        return instances

    def _stop_instance(self) -> None:
        result = subprocess.run(
            [self.apptainer, "instance", "stop", self.name],
            capture_output=True,
            env=self.env(),
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace").strip()
            error = RuntimeError("apptainer instance stop failed: " + stderr)
            if "no instance found" not in stderr.lower():
                raise error
            # The instance may disappear between the initial list and stop.
            # Only an authoritative listing in the same environment proves
            # cleanup can proceed; stderr alone is not sufficient evidence.
            try:
                instances = self.exec_instance_list(timeout=2)
                present = any(item["instance"] == self.name for item in instances)
            except Exception as listing_error:
                error.add_note(f"Could not verify instance absence: {listing_error}")
                raise error from listing_error
            if present:
                raise error
            return
        deadline = time.monotonic() + 5
        while self.overlay is not None and any(
            item["instance"] == self.name for item in self.exec_instance_list(timeout=2)
        ):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Instance is still listed after stop; retaining disk"
                )
            time.sleep(0.1)

    def exec(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        pwd: str | None = None,
        max_capture: int | None = None,
        command_env: dict[str, str] | None = None,
        on_start: Callable[[subprocess.Popen], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> subprocess.CompletedProcess:
        if self.name is None:
            raise RuntimeError("Sandbox has not been started")
        if self._instance_stopped:
            raise RuntimeError("Sandbox has stopped; network cleanup is still pending")
        if self.network is not None:
            self.network.ensure_alive()
        result = self._exec_internal(
            argv,
            stdin=stdin,
            pwd=pwd,
            max_capture=max_capture,
            command_env=command_env,
            on_start=on_start,
            on_complete=on_complete,
        )
        if self.network is not None:
            self.network.ensure_alive()
        return result

    def _exec_internal(
        self,
        argv: list[str],
        *,
        stdin: bytes | None = None,
        pwd: str | None = None,
        max_capture: int | None = None,
        command_env: dict[str, str] | None = None,
        on_start: Callable[[subprocess.Popen], None] | None = None,
        on_complete: Callable[[], None] | None = None,
    ) -> subprocess.CompletedProcess:
        process_env = dict(self.env(pwd=pwd))
        flags = []
        if command_env is not None:
            process_env.update(
                {
                    f"APPTAINERENV_{key}": value
                    for key, value in validate_command_env(command_env).items()
                }
            )
            # Literal overrides: never evaluate shell substitutions in values.
            flags = ["--no-eval"]
        # A new session lets cancellation signal only this command's group.
        process = subprocess.Popen(
            [
                self.apptainer,
                "exec",
                *self.exec_flags,
                *flags,
                f"instance://{self.name}",
                *argv,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=process_env,
            start_new_session=True,
        )
        if on_start is not None:
            on_start(process)
        return capture_process(
            process,
            stdin=stdin,
            max_capture=max_capture,
            track=self._drain_threads,
            on_complete=on_complete,
        )

    @property
    def exec_flags(self) -> tuple[str, ...]:
        caps = ()
        if env_enabled("SANDFLEET_SANDBOX_DROP_SETUID_CAPS"):
            caps += DROP_SETUID_CAPS
        if env_enabled("SANDFLEET_SANDBOX_DROP_NET_CAPS"):
            caps += DROP_NET_CAPS
        caps = tuple(dict.fromkeys(caps))
        return ("--drop-caps", ",".join(caps)) if caps else ()

    @property
    def uses_pid_namespace(self) -> bool:
        return any(
            flag in {"--containall", "-C", "--pid", "-p"} for flag in self.start_flags
        )

    def _command_pidfile(self, command_id: str) -> str:
        # Scoped to this instance generation and command id. The digest keeps
        # distinct ids distinct after sanitization.
        digest = hashlib.sha256(command_id.encode("utf-8")).hexdigest()[:16]
        readable = re.sub(r"[^A-Za-z0-9_.-]", "_", command_id)[:40]
        return f"/tmp/sandfleet-cancel-{self.name}-{readable}-{digest}.pgid"

    def _command_cancelfile(self, command_id: str) -> str:
        return self._command_pidfile(command_id) + ".stop"

    def _wrap_for_in_instance_cancel(self, command_id: str, wrapped: str) -> str:
        # Publish the in-container pgid atomically, then honour a .stop sentinel
        # left by a cancel that arrived before publication.
        pidfile = self._command_pidfile(command_id)
        cancelfile = self._command_cancelfile(command_id)
        inner = (
            f"echo $$ > {shlex.quote(pidfile + '.tmp')} && "
            f"mv -f {shlex.quote(pidfile + '.tmp')} {shlex.quote(pidfile)} && "
            f"if [ -e {shlex.quote(cancelfile)} ]; then "
            f"exit {CANCELLED_SIGNALLED_EXIT}; fi && "
            f"exec {wrapped}"
        )
        # setsid -w makes bash the group leader ($$ == pgid) and propagates its
        # exit status.
        return f"exec setsid -w bash -c {shlex.quote(inner)}"

    def cancel_command_group(
        self, command_id: str, timeout: float | None = None
    ) -> tuple[bool, str | None]:
        """Kill a command's process group from inside its PID namespace.

        Returns (group_gone, error). Never stops the instance. ``timeout`` caps
        the whole exec (kill-processes' fallback passes its remaining budget).
        """
        if not self.uses_pid_namespace:
            return True, None
        pidfile = self._command_pidfile(command_id)
        cancelfile = self._command_cancelfile(command_id)
        term_polls = max(1, int(CANCEL_TERM_GRACE_SECONDS / IN_INSTANCE_POLL_SECONDS))
        kill_polls = max(1, int(CANCEL_KILL_GRACE_SECONDS / IN_INSTANCE_POLL_SECONDS))
        pidfile_polls = max(1, int(CANCEL_PIDFILE_WAIT_SECONDS / 0.05))
        interval = IN_INSTANCE_POLL_SECONDS
        snippet = self._in_instance_kill_snippet(
            pidfile, cancelfile, pidfile_polls, term_polls, kill_polls, interval
        )
        overall = (
            CANCEL_PIDFILE_WAIT_SECONDS
            + CANCEL_TERM_GRACE_SECONDS
            + CANCEL_KILL_GRACE_SECONDS
            + 15.0
        )
        if timeout is not None:
            overall = min(overall, timeout)
        try:
            result = subprocess.run(
                [
                    self.apptainer,
                    "exec",
                    *self.exec_flags,
                    f"instance://{self.name}",
                    "bash",
                    "-c",
                    snippet,
                ],
                capture_output=True,
                env=self.env(),
                timeout=overall,
            )
        except subprocess.TimeoutExpired:
            return False, "in-instance cancel exec timed out"
        except OSError as error:
            return False, f"in-instance cancel exec failed: {error}"
        if result.returncode == 0:
            return True, None
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        reason = detail or f"in-instance kill exited {result.returncode}"
        return False, f"in-instance command group not reaped: {reason}"

    @staticmethod
    def _in_instance_kill_snippet(
        pidfile: str,
        cancelfile: str,
        pidfile_polls: int,
        term_polls: int,
        kill_polls: int,
        interval: float,
    ) -> str:
        # Exit codes: 0 group confirmed gone (the only success); 1 group survived
        # TERM and KILL; 2 no pidfile; 3 invalid pidfile; 4 no pidfile and the
        # .stop sentinel could not be created.
        return (
            "set -u; "
            f"cf={shlex.quote(cancelfile)}; pf={shlex.quote(pidfile)}; "
            'sentinel_ok=1; if ! : > "$cf" 2>/dev/null; then '
            "sentinel_ok=0; echo SENTINEL_CREATE_FAILED >&2; fi; "
            f'i=0; while [ ! -s "$pf" ] && [ $i -lt {pidfile_polls} ]; do '
            "sleep 0.05; i=$((i+1)); done; "
            'if [ ! -s "$pf" ]; then '
            'if [ "$sentinel_ok" = 0 ]; then echo NO_SENTINEL_NO_PIDFILE >&2; '
            "exit 4; fi; "
            "echo NO_PIDFILE >&2; exit 2; fi; "
            'pgid=$(cat "$pf" 2>/dev/null); '
            'case "$pgid" in ""|*[!0-9]*) echo BAD_PIDFILE >&2; exit 3;; esac; '
            'if [ "$pgid" -le 1 ]; then echo UNSAFE_PGID >&2; exit 3; fi; '
            # Scan /proc rather than `kill -0`, which conflates EPERM and ESRCH.
            # After the last ") " the fields are state, ppid, pgrp.
            "group_alive() { "
            "for s in /proc/[0-9]*/stat; do "
            'st=$(cat "$s" 2>/dev/null) || continue; '
            "rest=${st##*) }; set -- $rest; "
            '[ "${3:-}" = "$pgid" ] && return 0; '
            "done; return 1; }; "
            'kill -TERM -"$pgid" 2>/dev/null || true; '
            f"i=0; while [ $i -lt {term_polls} ]; do "
            f"group_alive || exit 0; sleep {interval}; i=$((i+1)); done; "
            'kill -KILL -"$pgid" 2>/dev/null || true; '
            f"i=0; while [ $i -lt {kill_polls} ]; do "
            f"group_alive || exit 0; sleep {interval}; i=$((i+1)); done; "
            "echo SURVIVOR >&2; exit 1"
        )

    def kill_processes(self, deadline: float | None = None) -> int:
        """SIGKILL every process in the sandbox except its init; return the count.

        ``deadline`` (``time.monotonic()``) caps the PID lookup and the kill loop.

        Host-side and independent of the image: the worker spawns an isolated
        Python helper that joins the instance's pinned user and PID namespaces
        (no mount namespace, so only the worker's own interpreter runs, never
        image code) and runs the kernel's atomic ``kill(-1)``, then verifies with
        a host-side ``/proc`` scan and repeats until nothing remains. Nothing in
        the sandbox runs the kill or decides success, so a sandbox root that
        shadows ``bash``/sets ``BASH_ENV``/``ld.so.preload`` cannot fake it.
        ``kill(-1)`` is scoped by the joined PID namespace, so it reaches
        processes forking concurrently but never the worker, the namespace's
        init, or other sandboxes of the shared host UID. Zombie leaders with live
        threads count as alive; a plain zombie does not.
        """
        if self.name is None:
            raise RuntimeError("Sandbox has not been started")
        if self.pid_ns_fd is None or self.user_ns_fd is None:
            raise RuntimeError(
                "Sandbox namespaces are not pinned; refusing to kill processes"
            )
        if deadline is None:
            deadline = time.monotonic() + KILL_PROCESSES_BUDGET_SECONDS
        init_pid = self._instance_pid(timeout=_budget(deadline, 10))
        return kill_namespace_processes(
            init_pid,
            self.user_ns_fd,
            self.pid_ns_fd,
            _budget(deadline, KILL_PROCESSES_TIMEOUT_SECONDS),
        )

    @property
    def readonly_mounts_status(self) -> list[dict[str, Any]]:
        return [mount.describe() for mount in self.readonly_mounts]

    def add_readonly_mounts(self, mounts: list[ReadonlyMount]) -> int:
        """Restart the instance from its disk overlay with extra read-only binds.

        Returns how many sandbox processes were stopped. Apptainer cannot add a
        mount to a running instance, so this SIGKILLs every sandbox process
        (host-side, as kill_processes does), creates the mount points in the
        overlay, stops the instance, and starts it again from the same overlay
        with every read-only bind so far. The files persist; processes do not.
        The new instance passes the same network, seccomp, PID-namespace and
        storage checks as a fresh start, and each mount is then verified from
        the host. If the restart fails, the sandbox and its overlay are closed.
        """
        if self.name is None or self._instance_stopped:
            raise RuntimeError("Sandbox has not been started")
        if self.overlay is None:
            raise ValueError(
                "Read-only mounts need a disk-overlay sandbox (storage_mb): a "
                "writable-tmpfs sandbox would lose its files when it restarts"
            )
        self._check_new_mounts(mounts)
        stopped = self.kill_processes()
        # Existing targets too: the agent may have replaced a start-time
        # target's parent directory with a symlink.
        self._prepare_mount_points(
            [mount.target for mount in [*self.readonly_mounts, *mounts]]
        )
        self._stop_keeping_storage()
        self.readonly_mounts = [*self.readonly_mounts, *mounts]
        self._launch(reuse_overlay=True)
        # Instance start runs the overlay's /.singularity.d/startscript, which the
        # agent could have rewritten to relaunch a process; stop those too.
        try:
            return stopped + self.kill_processes()
        except Exception:
            self.close()
            raise

    def _checked_start_mounts(self, mounts: list[ReadonlyMount]) -> list[ReadonlyMount]:
        self.readonly_mounts: list[ReadonlyMount] = []
        if mounts:
            self._check_new_mounts(mounts)
        return mounts

    def _check_new_mounts(self, mounts: list[ReadonlyMount]) -> None:
        """Checks shared by start-time and added mounts, before any change."""
        if not seccomp_denies_mounts(self.seccomp, default_profile()):
            raise RuntimeError(
                "Read-only mounts need a seccomp profile that denies the mount "
                "syscalls; without it sandbox root could remount them read-write"
            )
        targets = [mount.target for mount in mounts]
        check_target_overlap([mount.target for mount in self.readonly_mounts] + targets)
        check_protected_paths(targets, ["/tmp", "/var/tmp", "/root", self.pwd])

    def _prepare_mount_points(self, targets: list[str]) -> None:
        """Make each target a real directory reached through real directories.

        Apptainer needs bind destinations to exist when it starts from a
        persistent overlay and follows symlinks in them, so a symlink or
        non-directory on the way to a target is removed first. Existing
        directories are kept; the mount hides a target's contents. A sandbox
        that defeats this only makes the restart fail: the mounts are verified
        from the host afterwards.
        """
        paths: list[str] = []
        for target in targets:
            parts = target.strip("/").split("/")
            for index in range(1, len(parts) + 1):
                path = "/" + "/".join(parts[:index])
                if path not in paths:
                    paths.append(path)
        script = (
            'for p do if [ -L "$p" ] || { [ -e "$p" ] && [ ! -d "$p" ]; }; then '
            'rm -f -- "$p" || exit 1; fi; [ -d "$p" ] || mkdir -- "$p" || exit 1; done'
        )
        result = self.exec(
            ["sh", "-c", script, "sandfleet-mount-points", *paths], pwd="/"
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Could not create the read-only mount points: "
                + result.stderr.decode("utf-8", "replace").strip()
            )

    def _stop_keeping_storage(self) -> None:
        """Stop the instance and its relay but keep the disk overlay for a restart."""
        self._close_namespace_fds()
        if not self._instance_stopped:
            self._stop_instance()
            self._instance_stopped = True
        if self.network is not None:
            self.network.close()
            self.network = None
        alive = self.join_drain_threads()
        if alive:
            raise RuntimeError(
                f"{len(alive)} capture drain thread(s) still running after stop; "
                "refusing to restart the sandbox"
            )
        self.network_metadata = {}

    def _verify_readonly_mounts(self) -> None:
        """Check each bind with the worker's own helper, never a sandbox program."""
        pid = self._instance_pid()
        pidfd = os.pidfd_open(pid)
        try:
            root = os.open(
                f"/proc/{pid}/root", os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC
            )
            try:
                namespace = os.stat(f"/proc/{pid}/ns/pid").st_ino
                if namespace != os.fstat(self.pid_ns_fd).st_ino:
                    raise RuntimeError(
                        "Sandbox init is not in the pinned PID namespace"
                    )
                mountinfo = read_mountinfo(pid)
                # The root and mount table belong to this live init.
                signal.pidfd_send_signal(pidfd, 0)
                reports = report_mounts(
                    self.user_ns_fd,
                    root,
                    [mount.target for mount in self.readonly_mounts],
                )
            finally:
                os.close(root)
        finally:
            os.close(pidfd)
        # Read from the host, not via a sandbox program the agent could replace.
        check_instance_mounts(mountinfo, [m.target for m in self.readonly_mounts])
        leaked = network_filesystem_mounts(
            mountinfo, frozenset(mount.target for mount in self.readonly_mounts)
        )
        if leaked:
            raise RuntimeError(f"Sandbox has network or FUSE mounts: {leaked}")
        for report, mount in zip(reports, self.readonly_mounts, strict=True):
            check_mount_report(report, mount)

    def run_command(
        self,
        command: str,
        timeout: int | None,
        *,
        env: dict[str, str] | None = None,
        on_start: Callable[[subprocess.Popen], None] | None = None,
        on_complete: Callable[[], None] | None = None,
        command_id: str | None = None,
        worker_deadline: bool = False,
    ) -> dict[str, Any]:
        effective_timeout = self.timeout if timeout is None else timeout
        if worker_deadline:
            # The worker-side _CommandDeadline enforces the limit and reports
            # timed_out independently of the exit code; BusyBox `timeout` lacks
            # the long options used below.
            wrapped = f"bash -c {shlex.quote(command)}"
        else:
            wrapped = (
                f"timeout --signal=TERM --kill-after=10 {shlex.quote(str(effective_timeout))} "
                f"bash -c {shlex.quote(command)}"
            )
        if command_id is not None and self.uses_pid_namespace:
            wrapped = self._wrap_for_in_instance_cancel(str(command_id), wrapped)
        # Only command output is capped; +1 detects truncation.
        if self.pids_limit:
            # Inside the instance, so only this sandbox's user namespace is counted.
            wrapped = f"ulimit -u {self.pids_limit} && {wrapped}"
        options = {} if env is None else {"command_env": validate_command_env(env)}
        if on_complete is not None:
            options["on_complete"] = on_complete
        result = self.exec(
            ["bash", "-c", wrapped],
            max_capture=MAX_OUTPUT_BYTES + 1,
            on_start=on_start,
            **options,
        )
        stderr = result.stderr[:MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
        if not worker_deadline and result.returncode == 124:
            stderr = f"Command timed out after {effective_timeout}s.\n{stderr}"
        return {
            "stdout": result.stdout[:MAX_OUTPUT_BYTES].decode(
                "utf-8", errors="replace"
            ),
            "stderr": stderr,
            "exit_code": result.returncode,
            "output_truncated": len(result.stdout) > MAX_OUTPUT_BYTES
            or len(result.stderr) > MAX_OUTPUT_BYTES,
        }

    def write_file(
        self,
        path: str,
        content: bytes,
        *,
        offset: int | None = None,
        truncate: bool = False,
    ) -> None:
        directory = os.path.dirname(path) or "/"
        if offset is not None:
            self._validate_chunk(offset, len(content))
            if type(truncate) is not bool or (truncate and offset != 0):
                raise ValueError("truncate is only valid for the first chunk")
            prefix = f": > {shlex.quote(path)} && " if truncate else ""
            command = f"mkdir -p {shlex.quote(directory)} && {prefix}dd of={shlex.quote(path)} bs=262144 oflag=seek_bytes seek={offset} conv=notrunc status=none"
            result = self.exec(["sh", "-c", command], stdin=content)
            if result.returncode:
                raise RuntimeError(result.stderr.decode("utf-8", "replace"))
            return
        result = self.exec(
            [
                "sh",
                "-c",
                f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(path)}",
            ],
            stdin=content,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))

    @staticmethod
    def _validate_chunk(offset: int, size: int) -> None:
        if (
            type(offset) is not int
            or offset < 0
            or type(size) is not int
            or not 0 <= size <= 1048576
        ):
            raise ValueError("Invalid file chunk offset/size")

    def read_file(
        self, path: str, *, offset: int | None = None, size: int = 262144
    ) -> bytes:
        if offset is not None:
            self._validate_chunk(offset, size)
        exists = self.exec(["sh", "-c", f"test -e {shlex.quote(path)}"])
        if exists.returncode != 0:
            raise FileNotFoundError(path)
        is_directory = self.exec(["sh", "-c", f"test -d {shlex.quote(path)}"])
        if is_directory.returncode == 0:
            raise IsADirectoryError(path)
        command = (
            ["cat", path]
            if offset is None
            else [
                "dd",
                f"if={path}",
                "bs=262144",
                "iflag=skip_bytes,count_bytes",
                f"skip={offset}",
                f"count={size}",
                "status=none",
            ]
        )
        result = self.exec(command)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return result.stdout

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        # Absolute destinations do not depend on the episode's mutable cwd.
        result = self.exec(
            ["tar", "-xf", "-", "-C", root],
            stdin=tar_bytes,
            pwd="/" if os.path.isabs(root) else None,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


class _CommandDeadline:
    """Serialize foreground completion with deadline ownership and cleanup.

    Cleanup runs outside the lock, but finish joins it before publishing. Errors
    cross the timer-thread boundary and are surfaced by the exec caller.
    """

    def __init__(self, timeout, is_active, killer):
        self.timeout = timeout
        self.is_active = is_active
        self.killer = killer
        self.lock = threading.Lock()
        self.winner: str | None = None
        self.timer: threading.Timer | None = None
        self.error: str | None = None
        self.exception: Exception | None = None

    def start(self):
        if self.timeout and self.timeout > 0:
            self.timer = threading.Timer(float(self.timeout), self._fire)
            self.timer.daemon = True
            self.timer.start()

    def complete(self):
        with self.lock:
            if self.winner is None:
                self.winner = "foreground"

    def _fire(self):
        with self.lock:
            if self.winner is not None or not self.is_active():
                return
            self.winner = "deadline"
        try:
            interrupted, error = self.killer()
            if not interrupted:
                self.error = error or "deadline reap failed"
        except Exception as error:  # Deferred intact to the calling exec thread.
            self.exception = error

    def finish(self):
        if self.timer is not None:
            self.timer.cancel()
            self.timer.join()

    @property
    def timed_out(self):
        return self.winner == "deadline"


class _CancelOp:
    """Barrier for an in-flight cancellation of a running command.

    cancel_command registers one under the registry lock, then signals/waits
    WITHOUT holding the lock; the exec's finally blocks on ``done`` before it
    publishes its result, so the exec can never report a successful cancellation
    the signalling did not actually achieve (the completion race).
    """

    __slots__ = ("done", "interrupted", "error")

    def __init__(self) -> None:
        self.done = threading.Event()
        self.interrupted = False
        self.error: str | None = None


class WorkerSession:
    """Dispatch step RPCs against exactly one replaceable Apptainer instance."""

    _OPERATIONS = {
        "start": "_start",
        "restart": "_start",
        "run_command": "_run_command",
        "run_command_env": "_run_command_env",
        "write_file": "_write_file",
        "read_file": "_read_file",
        "put_archive": "_put_archive",
        "add_readonly_mounts": "_add_readonly_mounts",
        "status": "_status",
        "stop": "_stop",
        "shutdown": "_shutdown",
    }

    def __init__(
        self,
        storage_root: str | None = None,
        max_storage_mb: int = 0,
        *,
        gpus: int = 0,
        readonly_mount_roots: tuple[str, ...] = (),
        agent_readonly_mount_roots: tuple[str, ...] = (),
    ) -> None:
        self.gpus = gpus
        self.instance: ApptainerInstance | None = None
        self.storage_root = storage_root
        self.max_storage_mb = max_storage_mb
        # Operator-allowlisted host directories: verifier-phase roots for
        # add_readonly_mounts, agent-phase roots for start-time mounts. They are
        # resolved on each request, so a missing root fails only that request.
        self.agent_readonly_mount_roots, self.readonly_mount_roots = (
            validate_phase_roots(agent_readonly_mount_roots, readonly_mount_roots)
        )
        # Cancellation registry, shared with the control-channel thread. All
        # dicts below are keyed by command_id.
        self._cmd_lock = threading.Lock()
        # Process-group id captured at start; the leader may be reaped first.
        self._active: dict[str, int] = {}
        self._cancelled: collections.OrderedDict[str, None] = collections.OrderedDict()
        self._finished: collections.OrderedDict[str, None] = collections.OrderedDict()
        # Failures from signalling a cancel that raced command start.
        self._signal_failures: dict[str, str] = {}
        # In-instance killers for PID-namespace instances.
        self._instance_killers: dict[
            str, Callable[[], tuple[bool, str | None]] | None
        ] = {}
        # Barriers the exec waits on so it never reports an unachieved cancel.
        self._cancel_inflight: dict[str, _CancelOp] = {}
        # Serializes kill-processes against start/restart/stop of the instance
        # (reentrant: restart stops then starts). It is deliberately NOT held by
        # run_command, so a kill never waits on a hung exec.
        self._lifecycle_lock = threading.RLock()

    @staticmethod
    def _remember(
        store: collections.OrderedDict[str, None], key: str, cap: int
    ) -> None:
        store[key] = None
        store.move_to_end(key)
        while len(store) > cap:
            store.popitem(last=False)

    def _mark_finished(self, command_id: str) -> None:
        # Caller holds _cmd_lock.
        self._cancelled.pop(command_id, None)
        self._remember(self._finished, command_id, FINISHED_CACHE_SIZE)

    def _require_instance(self) -> ApptainerInstance:
        if self.instance is None:
            raise RuntimeError("Sandbox has not been started")
        return self.instance

    def _status(self, _request: dict[str, Any]) -> dict[str, Any]:
        return {
            "instance_name": None if self.instance is None else self.instance.name,
            "cgroup": current_cgroup(include_gpu=True),
            "storage": None if self.instance is None else self.instance.storage_status,
            "max_storage_mb": self.max_storage_mb,
            "network": {} if self.instance is None else self.instance.network_metadata,
            "readonly_mounts": (
                [] if self.instance is None else self.instance.readonly_mounts_status
            ),
            "readonly_mount_roots": list(self.readonly_mount_roots),
            "agent_readonly_mount_roots": list(self.agent_readonly_mount_roots),
        }

    def _start(self, request: dict[str, Any]) -> dict[str, Any]:
        config = dict(request["config"])
        if "network_dns" in config:
            raise RuntimeError(
                "Start request comes from a pre-0.9 agent (network_dns); this "
                "worker's source tree is newer. Replace the worker allocation"
            )
        validate_storage(config.get("storage_mb"), self.max_storage_mb)
        if config.get("storage_mb") is not None:
            config["storage_root"] = self.storage_root
        if "readonly_mounts" in config:
            config["readonly_mounts"] = self._agent_mounts(config["readonly_mounts"])
        with self._lifecycle_lock:
            self._stop(request)
            if bool(config.get("gpu_enabled")) != bool(self.gpus):
                raise ValueError(
                    "Container GPU mode does not match the Slurm step allocation"
                )
            self.instance = ApptainerInstance(**config)
            self.instance.start()
            return self._status(request)

    def _run_command(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_run(request, env=None)

    def _run_command_env(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._dispatch_run(request, env=validate_command_env(request["env"]))

    def _dispatch_run(
        self, request: dict[str, Any], *, env: dict[str, str] | None
    ) -> dict[str, Any]:
        instance = self._require_instance()
        command_id = request.get("command_id")
        if command_id is None:
            return instance.run_command(
                request["command"], request.get("timeout"), env=env
            )
        return self._run_cancellable(
            instance, str(command_id), request["command"], request.get("timeout"), env
        )

    def _begin_cancellable(self, command_id: str) -> dict[str, Any] | None:
        # Returns a terminal before-start result, or None to proceed with the run.
        with self._cmd_lock:
            if command_id in self._finished or command_id in self._active:
                raise ValueError(f"command_id {command_id!r} has already been used")
            if command_id not in self._cancelled:
                return None
            self._mark_finished(command_id)
        return {
            "stdout": "",
            "stderr": "Command cancelled before start.",
            "exit_code": CANCELLED_BEFORE_START_EXIT,
            "output_truncated": False,
            "cancelled": True,
        }

    def _await_command_outcome(self, command_id: str) -> tuple[bool, str | None]:
        # Drop the command from the active set, then wait (unlocked) for any
        # in-flight cancel/kill. Returns (was_cancelled, signal_failure).
        with self._cmd_lock:
            self._active.pop(command_id, None)
            self._instance_killers.pop(command_id, None)
            was_cancelled = command_id in self._cancelled
            signal_failure = self._signal_failures.pop(command_id, None)
            inflight = self._cancel_inflight.get(command_id)
        if inflight is not None:
            inflight.done.wait()
            if signal_failure is None and not inflight.interrupted:
                signal_failure = inflight.error or "interruption failed"
        with self._cmd_lock:
            if inflight is not None:
                # Re-read after the barrier: a cancel or kill marks the command
                # cancelled while resolving its barrier, which can land after the
                # first read (e.g. a slow kill reaping the command's processes).
                was_cancelled = command_id in self._cancelled
                self._cancel_inflight.pop(command_id, None)
            self._mark_finished(command_id)
        return was_cancelled, signal_failure

    def _command_is_active(self, command_id: str) -> bool:
        with self._cmd_lock:
            return command_id in self._active and command_id not in self._cancelled

    def _run_cancellable(
        self,
        instance: ApptainerInstance,
        command_id: str,
        command: str,
        timeout: int | None,
        env: dict[str, str] | None,
    ) -> dict[str, Any]:
        before_start = self._begin_cancellable(command_id)
        if before_start is not None:
            return before_start

        killer: Callable[..., tuple[bool, str | None]] | None = None
        if getattr(instance, "uses_pid_namespace", False):
            # Optional timeout= is passed only by kill-processes' bounded fallback.
            killer = lambda **options: instance.cancel_command_group(  # noqa: E731
                command_id, **options
            )

        deadline = None
        if killer is not None:
            effective_timeout = instance.timeout if timeout is None else timeout
            deadline = _CommandDeadline(
                effective_timeout, lambda: self._command_is_active(command_id), killer
            )

        def on_start(process: subprocess.Popen) -> None:
            self._register_command(process.pid, command_id, killer, deadline)

        deadline_options = (
            {"on_complete": deadline.complete, "worker_deadline": True}
            if deadline is not None
            else {}
        )
        try:
            result = instance.run_command(
                command,
                timeout,
                env=env,
                on_start=on_start,
                command_id=command_id,
                **deadline_options,
            )
            if deadline is not None:
                deadline.complete()
        finally:
            if deadline is not None:
                deadline.finish()
            was_cancelled, signal_failure = self._await_command_outcome(command_id)

        return self._finish_cancellable(
            command_id, result, deadline, was_cancelled, signal_failure
        )

    def _register_command(self, pgid, command_id, killer, deadline):
        with self._cmd_lock:
            self._active[command_id] = pgid
            self._instance_killers[command_id] = killer
            pending_cancel = command_id in self._cancelled
        if pending_cancel:
            interrupted, error = self._interrupt(pgid, killer)
            if not interrupted:
                with self._cmd_lock:
                    self._signal_failures[command_id] = error or "interruption failed"
        elif deadline is not None:
            deadline.start()

    @staticmethod
    def _finish_cancellable(
        command_id, result, deadline, was_cancelled, signal_failure
    ):
        if deadline is not None:
            if deadline.exception is not None:
                raise deadline.exception
            signal_failure = signal_failure or deadline.error
        if signal_failure is not None:
            raise SandfleetError(
                f"Could not interrupt command {command_id!r}: {signal_failure}",
                error_type="SandfleetError",
            )
        if was_cancelled:
            result["cancelled"] = True
            result["exit_code"] = CANCELLED_SIGNALLED_EXIT
            # User cancel takes precedence over a racing deadline.
            if deadline is not None:
                result["timed_out"] = False
        else:
            result["cancelled"] = False
            if deadline is not None:
                result["timed_out"] = deadline.timed_out
                if deadline.timed_out:
                    marker = f"Command timed out after {deadline.timeout}s."
                    existing = result.get("stderr", "")
                    result["stderr"] = f"{marker}\n{existing}" if existing else marker
        return result

    def cancel_command(self, command_id: str) -> dict[str, Any]:
        """Interrupt one running or not-yet-started command by identity."""
        command_id = str(command_id)
        with self._cmd_lock:
            if command_id in self._finished:
                return {
                    "cancelled": False,
                    "interrupted": False,
                    "state": "already_finished",
                }
            if (
                command_id not in self._cancelled
                and len(self._cancelled) >= CANCELLED_CACHE_SIZE
            ):
                # Evicting an acknowledged cancel could let its command start.
                raise SandfleetError(
                    "Cancellation registry is full; cannot register a new "
                    "cancellation without dropping an acknowledged one",
                    error_type="SandfleetError",
                )
            self._cancelled[command_id] = None
            pgid = self._active.get(command_id)
            killer = self._instance_killers.get(command_id)
            if pgid is None:
                return {"cancelled": True, "interrupted": False, "state": "pending"}
            # Register the barrier atomically with reading pgid; concurrent
            # cancels share one.
            op = self._cancel_inflight.get(command_id)
            i_own_op = op is None
            if i_own_op:
                op = _CancelOp()
                self._cancel_inflight[command_id] = op
        if not i_own_op:
            op.done.wait()
            interrupted, error = op.interrupted, op.error
        else:
            interrupted, error = False, None
            try:
                interrupted, error = self._interrupt(pgid, killer)
            finally:
                with self._cmd_lock:
                    op.interrupted = interrupted
                    op.error = (
                        error if interrupted else (error or "interruption failed")
                    )
                    op.done.set()
        response: dict[str, Any] = {
            "cancelled": True,
            "interrupted": interrupted,
            "state": "running",
        }
        if error is not None:
            response["error"] = error
        return response

    def kill_processes(self) -> dict[str, Any]:
        """Kill every sandbox process except init, reconciling in-flight commands.

        A command is marked cancelled only after the kill succeeds, so a failed
        kill never leaves it cancelled-without-a-deadline. On failure the owned
        commands fall back to the normal interrupt path, which is then surfaced.
        Background processes are not tracked by the worker; the namespace-wide
        kill is what reaches them. Serialized against start/restart/stop by the
        lifecycle lock (never against a running exec). One overall deadline
        (KILL_PROCESSES_BUDGET_SECONDS) bounds the lock wait, the kill, and any
        failure-path fallback interrupts.
        """
        deadline = time.monotonic() + KILL_PROCESSES_BUDGET_SECONDS
        lock_wait = _budget(deadline, LIFECYCLE_LOCK_TIMEOUT_SECONDS)
        if not self._lifecycle_lock.acquire(timeout=lock_wait):
            raise SandfleetError(
                "Sandbox is starting or stopping; kill-processes must be retried",
                error_type="SandfleetError",
            )
        try:
            instance = self._require_instance()
            with self._cmd_lock:
                running = dict(self._active)
                killers = dict(self._instance_killers)
                # Own the cancellation barrier for any command without one yet.
                owned = {
                    command_id: self._cancel_inflight.setdefault(
                        command_id, _CancelOp()
                    )
                    for command_id in running
                    if command_id not in self._cancel_inflight
                }
            try:
                killed = instance.kill_processes(deadline=deadline)
            except Exception:
                # The kill did not happen: interrupt owned commands the normal
                # way rather than leaving them cancelled with no deadline.
                for command_id, op in owned.items():
                    self._resolve_killed_command(
                        command_id,
                        op,
                        running[command_id],
                        killers.get(command_id),
                        deadline,
                    )
                raise
            for command_id, op in owned.items():
                # The namespace kill already reaped the command; just stand its
                # deadline down and release the barrier as interrupted.
                self._resolve_killed_command(command_id, op, None, None, deadline)
            return {"killed": killed, "cancelled_commands": sorted(running)}
        finally:
            self._lifecycle_lock.release()

    def _resolve_killed_command(
        self,
        command_id: str,
        op: _CancelOp,
        pgid: int | None,
        killer: Callable[..., tuple[bool, str | None]] | None,
        deadline: float,
    ) -> None:
        # Mark cancelled first so the exec reports it and its deadline stands
        # down. pgid is None after a successful kill (processes already gone);
        # otherwise the kill failed and we interrupt the group ourselves, within
        # the remaining budget, so the command is never left cancelled with
        # nothing having stopped it.
        with self._cmd_lock:
            self._remember(self._cancelled, command_id, CANCELLED_CACHE_SIZE)
        interrupted, error = (
            (True, None)
            if pgid is None
            else self._interrupt(pgid, killer, deadline=deadline)
        )
        with self._cmd_lock:
            op.interrupted = interrupted
            op.error = None if interrupted else (error or "interruption failed")
            op.done.set()

    def _interrupt(
        self,
        pgid: int,
        killer: Callable[..., tuple[bool, str | None]] | None,
        *,
        deadline: float | None = None,
    ) -> tuple[bool, str | None]:
        # With a killer, both the in-instance group and the host exec client
        # must be stopped. A deadline caps both (kill-processes' fallback).
        # Only the bounded path passes a deadline down.
        bound = {} if deadline is None else {"deadline": deadline}
        if killer is None:
            return self._signal_process(pgid, **bound)
        if deadline is None:
            in_ok, in_error = killer()
        else:
            remaining = deadline - time.monotonic()
            # Leave half for the host-side group stop below.
            in_ok, in_error = (
                killer(timeout=remaining / 2)
                if remaining > 0
                else (False, "kill-processes budget exhausted")
            )
        host_ok, host_error = self._signal_process(pgid, **bound)
        if in_ok and host_ok:
            return True, None
        reasons = []
        if not in_ok:
            reasons.append(in_error or "in-instance reap failed")
        if not host_ok:
            reasons.append(host_error or "host client stop failed")
        return False, "; ".join(reasons)

    @staticmethod
    def _signal_graces(deadline: float | None) -> tuple[float, float]:
        """TERM and KILL grace periods, scaled down to fit a deadline."""
        if deadline is None:
            return CANCEL_TERM_GRACE_SECONDS, CANCEL_KILL_GRACE_SECONDS
        remaining = max(0.0, deadline - time.monotonic())
        term = min(CANCEL_TERM_GRACE_SECONDS, remaining / 2)
        return term, min(CANCEL_KILL_GRACE_SECONDS, remaining - term)

    def _signal_process(
        self, pgid: int, *, deadline: float | None = None
    ) -> tuple[bool, str | None]:
        if pgid <= 1:
            return False, f"refusing to signal unsafe process group {pgid}"
        term_grace, kill_grace = self._signal_graces(deadline)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        if self._wait_group_gone(pgid, term_grace):
            return True, None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        if self._wait_group_gone(pgid, kill_grace):
            return True, None
        return False, "command process group did not exit after SIGKILL"

    @staticmethod
    def _wait_group_gone(pgid: int, grace_seconds: float) -> bool:
        deadline = time.monotonic() + grace_seconds
        while True:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return True  # no processes remain in the group
            except PermissionError:
                return False  # group exists but is not ours to signal
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    def _write_file(self, request: dict[str, Any]) -> None:
        self._require_instance().write_file(
            request["path"],
            request["content"],
            offset=request.get("offset"),
            truncate=request.get("truncate", False),
        )

    def _read_file(self, request: dict[str, Any]) -> bytes:
        return self._require_instance().read_file(
            request["path"],
            offset=request.get("offset"),
            size=request.get("size", 262144),
        )

    def _put_archive(self, request: dict[str, Any]) -> None:
        self._require_instance().put_archive(request["root"], request["tar_bytes"])

    def _agent_mounts(self, requested: Any) -> list[ReadonlyMount]:
        """Resolve a start request's mounts against the agent-phase roots only."""
        agent_roots = resolve_mount_roots(self.agent_readonly_mount_roots)
        # Symlinked roots may resolve into each other; recheck the resolved paths.
        check_phase_roots(agent_roots, resolve_mount_roots(self.readonly_mount_roots))
        return resolve_mounts(
            validate_mount_request(requested), agent_roots, phase=AGENT_PHASE
        )

    def _add_readonly_mounts(self, request: dict[str, Any]) -> dict[str, Any]:
        """Restart the instance with extra operator-allowlisted read-only mounts.

        Runs on the main channel, so no command of this lease is in flight; the
        lifecycle lock keeps kill-processes out while the instance is replaced.
        """
        requested = validate_mount_request(request["mounts"])
        with self._lifecycle_lock:
            instance = self._require_instance()
            mounts = resolve_mounts(
                requested, resolve_mount_roots(self.readonly_mount_roots)
            )
            stopped = instance.add_readonly_mounts(mounts)
            return {
                **self._status(request),
                "added_mounts": [mount.describe() for mount in mounts],
                "stopped_processes": stopped,
            }

    def _stop(self, _request: dict[str, Any]) -> dict[str, Any]:
        with self._lifecycle_lock:
            if self.instance is not None:
                self.instance.close()
                self.instance = None
            return self._status({})

    def _shutdown(self, request: dict[str, Any]) -> None:
        self._stop(request)

    def dispatch(self, request: dict[str, Any]) -> tuple[Any, bool]:
        operation = request["operation"]
        try:
            handler = getattr(self, self._OPERATIONS[operation])
        except KeyError as error:
            raise ValueError(f"Unknown worker operation {operation!r}") from error

        before = memory_events()
        result = handler(request)
        after = memory_events()
        if any(after[key] > before[key] for key in ("oom", "oom_kill")):
            raise SandboxOOMError("Sandbox exceeded its Slurm step memory limit")
        return result, operation == "shutdown"

    def close(self) -> None:
        self._stop({})


def error_response(error: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }


def _handle_control(session: WorkerSession, request: dict[str, Any]) -> Any:
    operation = request.get("operation")
    if operation == "cancel":
        return session.cancel_command(str(request["command_id"]))
    if operation == "kill_processes":
        return session.kill_processes()
    if operation == "status":
        return session._status(request)
    raise ValueError(f"Unknown control operation {operation!r}")


def _serve_control(session: WorkerSession, listener: Listener) -> None:
    # Serves cancels and process kills while the main channel is blocked in an exec.
    while True:
        try:
            connection = listener.accept()
        except OSError:
            return  # listener closed at shutdown
        try:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                try:
                    result = _handle_control(session, request)
                    connection.send({"ok": True, "result": result})
                except Exception as error:
                    connection.send(error_response(error))
        except OSError:
            pass
        finally:
            connection.close()


def serve(
    socket_path: str,
    authkey: bytes,
    *,
    control_socket_path: str | None = None,
    storage_root: str | None = None,
    max_storage_mb: int = 0,
    gpus: int = 0,
    readonly_mount_roots: tuple[str, ...] = (),
    agent_readonly_mount_roots: tuple[str, ...] = (),
) -> int:
    session = WorkerSession(
        storage_root,
        max_storage_mb,
        gpus=gpus,
        readonly_mount_roots=readonly_mount_roots,
        agent_readonly_mount_roots=agent_readonly_mount_roots,
    )
    listener = Listener(socket_path, family="AF_UNIX", authkey=authkey)
    os.chmod(socket_path, 0o600)
    control_listener: Listener | None = None
    if control_socket_path is not None:
        control_listener = Listener(
            control_socket_path, family="AF_UNIX", authkey=authkey
        )
        os.chmod(control_socket_path, 0o600)
        threading.Thread(
            target=_serve_control,
            args=(session, control_listener),
            name="sfworker-control",
            daemon=True,
        ).start()
    try:
        connection = listener.accept()
        try:
            while True:
                try:
                    request = connection.recv()
                except EOFError:
                    break
                try:
                    result, should_exit = session.dispatch(request)
                    connection.send({"ok": True, "result": result})
                except Exception as error:
                    should_exit = False
                    connection.send(error_response(error))
                if should_exit:
                    break
        finally:
            connection.close()
    finally:
        session.close()
        listener.close()
        if control_listener is not None:
            control_listener.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(socket_path)
        if control_socket_path is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(control_socket_path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--control-socket")
    parser.add_argument("--storage-root")
    parser.add_argument("--max-storage-mb", type=int, default=0)
    parser.add_argument(
        "--readonly-mount-root",
        action="append",
        default=[],
        dest="readonly_mount_roots",
    )
    parser.add_argument(
        "--agent-readonly-mount-root",
        action="append",
        default=[],
        dest="agent_readonly_mount_roots",
    )
    args = parser.parse_args()
    return serve(
        args.socket,
        bytes.fromhex(args.authkey),
        control_socket_path=args.control_socket,
        storage_root=args.storage_root,
        max_storage_mb=args.max_storage_mb,
        gpus=args.gpus,
        readonly_mount_roots=tuple(args.readonly_mount_roots),
        agent_readonly_mount_roots=tuple(args.agent_readonly_mount_roots),
    )


if __name__ == "__main__":
    raise SystemExit(main())
