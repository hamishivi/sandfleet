"""Operator-allowlisted read-only host mounts for a lease's sandbox.

Mounts belong to one of two phases, each with its own operator roots:

- ``agent``: given when the sandbox starts (``start``/``restart``), for inputs
  the agent may read (for example model weights or data).
- ``verifier``: added after the agent phase and before verification
  (``add-mounts``), for inputs only a verifier may see (for example a hidden
  model). Apptainer cannot add a mount to a running instance (``apptainer exec
  --bind instance://...`` ignores the bind), so the step worker stops the
  instance, keeps its disk overlay, and starts it again with every bind.

No agent root may contain a verifier root, so a bad start request cannot show
verifier-only files to the agent. Everything here is pure validation and
parsing; the lifecycle lives in ``step_worker``.

Read-only depends on the seccomp profile: fakeroot sandbox root keeps
CAP_SYS_ADMIN in its user namespace, so without a filter on the mount syscalls
it could remount an Apptainer ``:ro`` bind read-write and write to the host
directory as the worker's UID. Workers therefore refuse the operation unless the
active profile denies every syscall in ``MOUNT_SYSCALLS``.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import stat
from dataclasses import dataclass
from typing import Any

MAX_READONLY_MOUNTS = 16
AGENT_PHASE = "agent"
VERIFIER_PHASE = "verifier"
MAX_PATH_LENGTH = 1024
# Every syscall that can change a mount's flags or attach/detach mounts.
MOUNT_SYSCALLS = (
    "mount",
    "umount",
    "umount2",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "move_mount",
    "open_tree",
    "mount_setattr",
)
# Targets that would shadow Apptainer's or the kernel's own mounts.
RESERVED_TARGET_ROOTS = ("/proc", "/sys", "/dev", "/etc", "/.singularity.d")
# ":" and "," separate fields in Apptainer's --bind/--mount syntax.
_UNSAFE_PATH = re.compile(r"[:,\x00-\x1f\x7f]")
_DENY_ACTIONS = frozenset(
    {
        "SCMP_ACT_ERRNO",
        "SCMP_ACT_KILL",
        "SCMP_ACT_KILL_THREAD",
        "SCMP_ACT_KILL_PROCESS",
        "SCMP_ACT_TRAP",
    }
)


@dataclass(frozen=True)
class ReadonlyMount:
    """One validated mount: ``host_path`` is the resolved ``source``.

    ``identity`` is the source directory's (st_dev, st_ino) when it was
    resolved; the sandbox's mount must be a bind of that very directory, so a
    path swapped on the host after the check fails verification.
    """

    source: str
    target: str
    host_path: str
    phase: str = VERIFIER_PHASE
    identity: tuple[int, int] | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "read_only": True,
            "phase": self.phase,
        }

    def bind_flag(self) -> tuple[str, str]:
        return ("--bind", f"{self.host_path}:{self.target}:ro")


def _clean_absolute(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    if len(value) > MAX_PATH_LENGTH:
        raise ValueError(f"{what} is longer than {MAX_PATH_LENGTH} characters")
    if _UNSAFE_PATH.search(value):
        raise ValueError(f"{what} {value!r} contains ':', ',' or a control character")
    if not value.startswith("/") or posixpath.normpath(value) != value:
        raise ValueError(f"{what} {value!r} must be an absolute, normalized path")
    if value.startswith("//"):
        raise ValueError(f"{what} {value!r} must be an absolute, normalized path")
    if value == "/":
        raise ValueError(f"{what} must not be /")
    return value


def is_within(path: str, root: str) -> bool:
    """True when ``path`` equals ``root`` or is below it (both normalized)."""
    return path == root or path.startswith(root.rstrip("/") + "/")


def validate_mount_roots(roots: Any) -> tuple[str, ...]:
    """Validate operator-configured roots without touching the filesystem."""
    if not isinstance(roots, (list, tuple)):
        raise ValueError("Read-only mount roots must be a list of paths")
    cleaned = tuple(_clean_absolute(root, "Read-only mount root") for root in roots)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("Read-only mount roots must be unique")
    return cleaned


def check_phase_roots(
    agent_roots: tuple[str, ...], verifier_roots: tuple[str, ...]
) -> None:
    """Refuse an agent root at or above a verifier root.

    An agent root inside a verifier root is fine (verifiers may also read what
    the agent reads); the reverse would let a start-time mount show the agent
    files meant only for verification.
    """
    for agent_root in agent_roots:
        for verifier_root in verifier_roots:
            if is_within(verifier_root, agent_root):
                raise ValueError(
                    f"Agent read-only mount root {agent_root!r} contains the "
                    f"verifier-only root {verifier_root!r}"
                )


def validate_phase_roots(
    agent_roots: Any, verifier_roots: Any
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate both phases' configured roots without touching the filesystem."""
    agent = validate_mount_roots(agent_roots)
    verifier = validate_mount_roots(verifier_roots)
    check_phase_roots(agent, verifier)
    return agent, verifier


def resolve_mount_roots(roots: Any) -> tuple[str, ...]:
    """Resolve symlinks in each root on this node; every root must be a directory."""
    resolved = []
    for root in validate_mount_roots(roots):
        real = os.path.realpath(root)
        if not os.path.isdir(real):
            raise ValueError(f"Read-only mount root {root!r} is not a directory")
        resolved.append(real)
    return tuple(resolved)


def validate_mount_request(mounts: Any) -> list[dict[str, str]]:
    """Check the wire shape of a mount request; no filesystem access."""
    if not isinstance(mounts, list) or not mounts:
        raise ValueError("mounts must be a non-empty list")
    if len(mounts) > MAX_READONLY_MOUNTS:
        raise ValueError(f"At most {MAX_READONLY_MOUNTS} read-only mounts are allowed")
    cleaned = []
    for item in mounts:
        if not isinstance(item, dict) or set(item) != {"source", "target"}:
            raise ValueError("Each mount must be an object with source and target")
        source = _clean_absolute(item["source"], "Mount source")
        target = _clean_absolute(item["target"], "Mount target")
        if any(is_within(target, root) for root in RESERVED_TARGET_ROOTS):
            raise ValueError(f"Mount target {target!r} is under a reserved path")
        cleaned.append({"source": source, "target": target})
    check_target_overlap([item["target"] for item in cleaned])
    return cleaned


def check_target_overlap(targets: list[str]) -> None:
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            if is_within(target, other) or is_within(other, target):
                raise ValueError(
                    f"Mount targets {target!r} and {other!r} are equal or nested"
                )


def check_protected_paths(targets: list[str], protected: list[str]) -> None:
    """Refuse targets at or above a path that must stay on the writable overlay."""
    for target in targets:
        for path in protected:
            if is_within(path, target):
                raise ValueError(
                    f"Mount target {target!r} would cover the writable path {path!r}"
                )


def mountinfo_entries(mountinfo: str) -> list[tuple[str, frozenset[str]]]:
    """(mount point, per-mount options) for each line of a mountinfo table."""
    entries = []
    for line in mountinfo.splitlines():
        fields = line.partition(" - ")[0].split()
        if len(fields) < 6:
            raise RuntimeError(f"Malformed mountinfo line: {line!r}")
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        entries.append((point, frozenset(fields[5].split(","))))
    return entries


def mounts_below(mountinfo: str, path: str) -> list[str]:
    """Mount points strictly below ``path``."""
    return [
        point
        for point, _options in mountinfo_entries(mountinfo)
        if point != path and is_within(point, path)
    ]


def check_instance_mounts(mountinfo: str, targets: list[str]) -> None:
    """Check the sandbox's mount table: each target is mounted read-only, with
    nothing mounted below it.

    Apptainer binds recursively but makes only the top mount read-only, so a
    mount below a source would appear writable inside the sandbox.
    """
    entries = mountinfo_entries(mountinfo)
    for target in targets:
        at_target = [options for point, options in entries if point == target]
        if not at_target or any("ro" not in options for options in at_target):
            raise RuntimeError(f"Sandbox mount at {target!r} is not read-only")
        below = [p for p, _o in entries if p != target and is_within(p, target)]
        if below:
            raise RuntimeError(
                f"Sandbox has mounts below the read-only mount {target!r}: {below}"
            )


def resolve_mounts(
    mounts: list[dict[str, str]],
    resolved_roots: tuple[str, ...],
    host_mountinfo: str | None = None,
    phase: str = VERIFIER_PHASE,
) -> list[ReadonlyMount]:
    """Resolve each source on this node and require it inside one of the
    phase's allowlisted roots.

    A source with a mount below it is refused: Apptainer binds recursively and
    makes only the top mount read-only, so the sandbox could write to the lower
    mount.
    """
    if not resolved_roots:
        raise ValueError(
            f"Read-only {phase}-phase mounts are not enabled on this worker"
        )
    if host_mountinfo is None:
        with open("/proc/self/mountinfo", encoding="utf-8") as handle:
            host_mountinfo = handle.read()
    result = []
    for item in mounts:
        real = os.path.realpath(item["source"])
        if not any(is_within(real, root) for root in resolved_roots):
            raise PermissionError(
                f"Mount source {item['source']!r} is not inside an operator-allowlisted "
                f"read-only mount root for the {phase} phase"
            )
        try:
            seen = os.stat(real, follow_symlinks=False)
        except FileNotFoundError:
            seen = None
        if seen is None or not stat.S_ISDIR(seen.st_mode):
            raise ValueError(f"Mount source {item['source']!r} is not a directory")
        if os.path.realpath(real) != real:
            raise PermissionError(
                f"Mount source {item['source']!r} changed while it was resolved"
            )
        if _UNSAFE_PATH.search(real):
            raise ValueError(
                f"Mount source {item['source']!r} resolves to an unsafe path"
            )
        below = mounts_below(host_mountinfo, real)
        if below:
            raise PermissionError(
                f"Mount source {item['source']!r} has mounts below it ({below}); "
                "they would not be read-only in the sandbox"
            )
        result.append(
            ReadonlyMount(
                item["source"],
                item["target"],
                real,
                phase,
                (seen.st_dev, seen.st_ino),
            )
        )
    return result


def profile_denies_mounts(profile: dict[str, Any]) -> bool:
    """True when an OCI seccomp profile denies every MOUNT_SYSCALLS entry outright.

    A syscall counts as denied when an argument-free rule names it with a deny
    action, or when the default action denies and no rule names it at all.
    """
    rules = profile.get("syscalls") or []
    default_denies = profile.get("defaultAction") in _DENY_ACTIONS
    for name in MOUNT_SYSCALLS:
        named = [rule for rule in rules if name in (rule.get("names") or [])]
        if any(
            rule.get("action") in _DENY_ACTIONS and not rule.get("args")
            for rule in named
        ):
            continue
        if default_denies and not named:
            continue
        return False
    return True


def seccomp_denies_mounts(profile: str | None, default_profile: dict[str, Any]) -> bool:
    """Check the worker's active profile ("default", a path, or None when off)."""
    if profile is None:
        return False
    if profile == "default":
        return profile_denies_mounts(default_profile)
    with open(profile, encoding="utf-8") as handle:
        return profile_denies_mounts(json.load(handle))


def check_mount_report(report: dict[str, Any], mount: ReadonlyMount) -> None:
    """Check one helper report: ``mount.target`` is a read-only bind of its source.

    A bind mount shares its source's filesystem and inode, so the target must
    have the source's (st_dev, st_ino) as resolved (``mount.identity``), and
    the mount it lies on must be read-only. The report comes from the worker's own helper, never from a
    program inside the sandbox.
    """
    if report.get("target") != mount.target:
        raise RuntimeError(f"Malformed mount report for {mount.target!r}: {report!r}")
    if "error" in report:
        raise RuntimeError(
            f"Sandbox has no directory at {mount.target!r}: {report['error']}"
        )
    identity = mount.identity
    if identity is None:
        source = os.stat(mount.host_path)
        identity = (source.st_dev, source.st_ino)
    if (report["dev"], report["ino"]) != identity:
        raise RuntimeError(
            f"Sandbox directory {mount.target!r} is not a bind of {mount.source!r}"
        )
    if report["read_only"] is not True:
        raise RuntimeError(f"Sandbox mount at {mount.target!r} is not read-only")
