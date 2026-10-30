from __future__ import annotations

import json
import os

import pytest

from sandfleet import seccomp


def rules(profile, name):
    return [rule for rule in profile["syscalls"] if name in rule["names"]]


def test_profile_denies_exploit_surface_and_allows_the_rest():
    profile = seccomp.default_profile()
    assert profile["defaultAction"] == "SCMP_ACT_ALLOW"
    # Native ABI only: libseccomp kills syscalls from any other architecture.
    assert profile["architectures"] == [seccomp.native_architecture()]
    requested = {
        "mount",
        "umount2",
        "pivot_root",
        "bpf",
        "perf_event_open",
        "userfaultfd",
        "keyctl",
        "add_key",
        "request_key",
        "kexec_load",
        "kexec_file_load",
        "init_module",
        "finit_module",
        "delete_module",
        "open_by_handle_at",
    }
    (unconditional,) = [
        rule
        for rule in profile["syscalls"]
        if "args" not in rule and rule["action"] == "SCMP_ACT_ERRNO"
    ]
    assert requested <= set(unconditional["names"])
    for name in ("execve", "clone", "unshare", "ptrace", "setns", "socket"):
        assert name not in unconditional["names"]


def test_namespace_creation_is_refused_only_with_clone_newuser():
    profile = seccomp.default_profile()
    for name in ("clone", "unshare"):
        (rule,) = rules(profile, name)
        assert rule["action"] == "SCMP_ACT_ERRNO"
        assert rule["args"] == [
            {
                "index": 0,
                "value": seccomp.CLONE_NEWUSER,
                "valueTwo": seccomp.CLONE_NEWUSER,
                "op": "SCMP_CMP_MASKED_EQ",
            }
        ]


def test_clone3_gets_enosys_and_no_sandbox_tracer_can_intercept_it():
    profile = seccomp.default_profile()
    # Apptainer forces EPERM for ERRNO rules; TRACE without a seccomp tracer is ENOSYS,
    # which glibc answers by falling back to clone (filtered above).
    (clone3,) = rules(profile, "clone3")
    assert clone3 == {"names": ["clone3"], "action": "SCMP_ACT_TRACE"}
    ptrace = rules(profile, "ptrace")
    assert sorted(rule["args"][0]["value"] for rule in ptrace) == [
        seccomp.PTRACE_SETOPTIONS,
        seccomp.PTRACE_SEIZE,
    ]
    for rule in ptrace:
        assert rule["action"] == "SCMP_ACT_ERRNO"
        assert rule["args"][0]["op"] == "SCMP_CMP_EQ"
        assert rule["args"][1] == {
            "index": 3,
            "value": seccomp.PTRACE_O_TRACESECCOMP,
            "valueTwo": seccomp.PTRACE_O_TRACESECCOMP,
            "op": "SCMP_CMP_MASKED_EQ",
        }


def test_profile_file_is_private_json():
    path = seccomp.write_default_profile()
    try:
        assert os.stat(path).st_mode & 0o777 == 0o600
        with open(path, encoding="utf-8") as handle:
            assert json.load(handle) == seccomp.default_profile()
    finally:
        os.unlink(path)


@pytest.mark.parametrize(
    ("machine", "expected"),
    [("x86_64", "SCMP_ARCH_X86_64"), ("aarch64", "SCMP_ARCH_AARCH64")],
)
def test_profile_lists_only_the_native_architecture(machine, expected):
    assert seccomp.default_profile(machine)["architectures"] == [expected]


@pytest.mark.parametrize("machine", ["s390x", "ppc64le", "armv7l", ""])
def test_unsupported_architecture_fails_loudly(machine):
    with pytest.raises(ValueError, match="SANDFLEET_SANDBOX_SECCOMP"):
        seccomp.default_profile(machine)
