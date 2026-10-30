"""Default seccomp profile for sandboxes (Apptainer `--security seccomp:`).

Everything is allowed except syscalls that sandboxed workloads (compilers,
package managers, interpreters, servers) do not need and that mostly serve
kernel exploits or escapes. Apptainer applies the filter to every instance
process; `apptainer exec` into the instance inherits it.

Apptainer (1.5) returns EPERM for every SCMP_ACT_ERRNO rule and ignores
`errnoRet`. clone3 cannot be filtered on its flags (they are in a struct), and
glibc >= 2.34 falls back from clone3 to clone only on ENOSYS: EPERM would
break thread creation. clone3 therefore uses SCMP_ACT_TRACE, which the kernel
answers with ENOSYS unless a tracer has enabled PTRACE_O_TRACESECCOMP; the
profile refuses that ptrace option, so no sandbox process can become such a
tracer. clone and unshare are then refused only with CLONE_NEWUSER.

The profile lists only the native architecture. Apptainer always filters the
native ABI and adds only the non-native architectures a profile lists, so no
other ABI gets rules; libseccomp's default bad-architecture action then kills
any thread that makes a syscall through another ABI (for example i386 or x32 on
x86_64, or AArch32 on arm64). Only x86_64 and aarch64 are supported: argument
positions such as clone's flags differ on other architectures (s390x).
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from typing import Any

CLONE_NEWUSER = 0x10000000
PTRACE_SETOPTIONS = 0x4200
PTRACE_SEIZE = 0x4206
PTRACE_O_TRACESECCOMP = 0x80
NATIVE_ARCHITECTURES = {"x86_64": "SCMP_ARCH_X86_64", "aarch64": "SCMP_ARCH_AARCH64"}

DENIED_SYSCALLS = (
    # Mount table changes, including the new mount API.
    "mount",
    "umount",
    "umount2",
    "pivot_root",
    "fsopen",
    "fsconfig",
    "fsmount",
    "fspick",
    "move_mount",
    "open_tree",
    "mount_setattr",
    # Kernel attack surface without a sandbox use.
    "bpf",
    "perf_event_open",
    "userfaultfd",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "keyctl",
    "add_key",
    "request_key",
    "kexec_load",
    "kexec_file_load",
    "init_module",
    "finit_module",
    "delete_module",
    "open_by_handle_at",
)


def _masked(index: int, mask: int) -> dict[str, Any]:
    return {"index": index, "value": mask, "valueTwo": mask, "op": "SCMP_CMP_MASKED_EQ"}


def native_architecture(machine: str | None = None) -> str:
    machine = platform.machine() if machine is None else machine
    try:
        return NATIVE_ARCHITECTURES[machine]
    except KeyError:
        raise ValueError(
            f"The default seccomp profile supports {sorted(NATIVE_ARCHITECTURES)}, "
            f"not {machine!r}; set SANDFLEET_SANDBOX_SECCOMP to a profile path or 0"
        ) from None


def default_profile(machine: str | None = None) -> dict[str, Any]:
    deny = "SCMP_ACT_ERRNO"
    syscalls: list[dict[str, Any]] = [
        {"names": list(DENIED_SYSCALLS), "action": deny},
        {
            "names": ["clone", "unshare"],
            "action": deny,
            "args": [_masked(0, CLONE_NEWUSER)],
        },
        {"names": ["clone3"], "action": "SCMP_ACT_TRACE"},
    ]
    for request in (PTRACE_SETOPTIONS, PTRACE_SEIZE):
        syscalls.append(
            {
                "names": ["ptrace"],
                "action": deny,
                "args": [
                    {"index": 0, "value": request, "op": "SCMP_CMP_EQ"},
                    _masked(3, PTRACE_O_TRACESECCOMP),
                ],
            }
        )
    return {
        "defaultAction": "SCMP_ACT_ALLOW",
        "architectures": [native_architecture(machine)],
        "syscalls": syscalls,
    }


def write_default_profile() -> str:
    """Write the profile to a private temporary file and return its path."""
    descriptor, path = tempfile.mkstemp(prefix="sandfleet-seccomp-", suffix=".json")
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(default_profile(), handle)
    return path
