"""Operator-allowlisted read-only mounts added to a started disk sandbox."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sandfleet import mounts, step_worker
from sandfleet.agent import AgentHandler, AgentState, build_parser, validate_args
from sandfleet.audit import AuditLog
from sandfleet.client import ADD_MOUNTS_TIMEOUT
from sandfleet.http import request_json
from sandfleet.mounts import ReadonlyMount
from sandfleet.seccomp import default_profile
from sandfleet.slurm import SlurmLauncher
from sandfleet.step_backend import ADD_READONLY_MOUNTS_TIMEOUT_SECONDS, SlurmStepSandbox
from sandfleet.step_worker import ApptainerInstance, WorkerSession

from test_agent import FakeSandbox
from test_client import lease  # noqa: F401 - fixture
from test_slurm import settings

# --- request and root validation -------------------------------------------


def test_valid_request_is_returned_cleaned():
    request = [{"source": "/models/example", "target": "/verifier/models/example"}]
    assert mounts.validate_mount_request(request) == request


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "non-empty list"),
        ([], "non-empty list"),
        ([{"source": "/a", "target": f"/t{i}"} for i in range(17)], "At most 16"),
        ([{"source": "/a"}], "source and target"),
        ([{"source": "/a", "target": "/t", "mode": "rw"}], "source and target"),
        (["/a:/t"], "source and target"),
        ([{"source": "a", "target": "/t"}], "absolute, normalized"),
        ([{"source": "/a/../b", "target": "/t"}], "absolute, normalized"),
        ([{"source": "/a/", "target": "/t"}], "absolute, normalized"),
        ([{"source": "//a", "target": "/t"}], "absolute, normalized"),
        ([{"source": "/a", "target": "/t/./x"}], "absolute, normalized"),
        ([{"source": "/a:b", "target": "/t"}], "':', ','"),
        ([{"source": "/a", "target": "/t,ro"}], "':', ','"),
        ([{"source": "/a", "target": "/t\nx"}], "control character"),
        ([{"source": "/a", "target": "/"}], "must not be /"),
        ([{"source": 3, "target": "/t"}], "non-empty string"),
        ([{"source": "/a", "target": "/etc/hosts"}], "reserved"),
        ([{"source": "/a", "target": "/proc"}], "reserved"),
        ([{"source": "/a", "target": "/dev/shm/x"}], "reserved"),
        ([{"source": "/a", "target": "/.singularity.d/x"}], "reserved"),
        (
            [{"source": "/a", "target": "/v"}, {"source": "/b", "target": "/v/m"}],
            "equal or nested",
        ),
        (
            [{"source": "/a", "target": "/v"}, {"source": "/b", "target": "/v"}],
            "equal or nested",
        ),
    ],
)
def test_malformed_requests_are_refused(value, message):
    with pytest.raises(ValueError, match=message):
        mounts.validate_mount_request(value)


def test_sibling_targets_with_a_common_prefix_are_not_nested():
    mounts.validate_mount_request(
        [{"source": "/a", "target": "/v/m"}, {"source": "/b", "target": "/v/mm"}]
    )


@pytest.mark.parametrize("roots", ["/models", ["relative"], ["/a", "/a"], ["/"]])
def test_malformed_roots_are_refused(roots):
    with pytest.raises(ValueError):
        mounts.validate_mount_roots(roots)


def test_protected_writable_paths_cannot_be_covered():
    with pytest.raises(ValueError, match="writable path '/workspace'"):
        mounts.check_protected_paths(["/workspace"], ["/tmp", "/workspace"])
    with pytest.raises(ValueError, match="writable path '/var/tmp'"):
        mounts.check_protected_paths(["/var"], ["/var/tmp"])
    mounts.check_protected_paths(["/workspace/models"], ["/tmp", "/workspace"])


# --- source resolution -------------------------------------------------------


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "allowed"
    (root / "model").mkdir(parents=True)
    (root / "model" / "weights").write_text("w")
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside)
    (root / "file").write_text("f")
    return SimpleNamespace(root=str(root), outside=str(outside))


def test_source_inside_a_root_resolves(tree):
    [mount] = mounts.resolve_mounts(
        [{"source": f"{tree.root}/model", "target": "/verifier/m"}],
        mounts.resolve_mount_roots([tree.root]),
    )
    assert mount.host_path == os.path.realpath(f"{tree.root}/model")
    assert mount.bind_flag() == ("--bind", f"{mount.host_path}:/verifier/m:ro")
    assert mount.describe() == {
        "source": f"{tree.root}/model",
        "target": "/verifier/m",
        "read_only": True,
        "phase": "verifier",
    }


@pytest.mark.parametrize(
    ("source", "error", "message"),
    [
        ("{root}/escape", PermissionError, "allowlisted"),
        ("{outside}", PermissionError, "allowlisted"),
        ("{root}/file", ValueError, "not a directory"),
        ("{root}/missing", ValueError, "not a directory"),
    ],
)
def test_sources_outside_roots_or_not_directories_are_refused(
    tree, source, error, message
):
    path = source.format(root=tree.root, outside=tree.outside)
    with pytest.raises(error, match=message):
        mounts.resolve_mounts(
            [{"source": path, "target": "/v"}], mounts.resolve_mount_roots([tree.root])
        )


def test_a_root_prefix_is_not_a_root(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models-secret").mkdir()
    with pytest.raises(PermissionError):
        mounts.resolve_mounts(
            [{"source": str(tmp_path / "models-secret"), "target": "/v"}],
            mounts.resolve_mount_roots([str(tmp_path / "models")]),
        )


def test_a_source_with_a_mount_below_it_is_refused(tree):
    # Apptainer binds recursively and makes only the top mount read-only.
    model = os.path.realpath(f"{tree.root}/model")
    request = [{"source": model, "target": "/v"}]
    roots = mounts.resolve_mount_roots([tree.root])
    escaped = model.replace(" ", "\\040")
    table = f"1 0 0:1 / / rw - xfs d rw\n2 1 0:2 / {escaped} rw - nfs s rw\n"
    assert mounts.resolve_mounts(request, roots, table)[0].host_path == model
    with pytest.raises(PermissionError, match="mounts below it"):
        mounts.resolve_mounts(
            request, roots, table + f"3 2 0:3 / {escaped}/sub rw - fuse.x f rw\n"
        )


def test_no_roots_means_disabled(tree):
    with pytest.raises(ValueError, match="not enabled"):
        mounts.resolve_mounts([{"source": f"{tree.root}/model", "target": "/v"}], ())


def test_symlinked_root_is_resolved(tree, tmp_path):
    link = tmp_path / "root-link"
    link.symlink_to(tree.root)
    [mount] = mounts.resolve_mounts(
        [{"source": f"{link}/model", "target": "/v"}],
        mounts.resolve_mount_roots([str(link)]),
    )
    assert mount.host_path == os.path.realpath(f"{tree.root}/model")


def test_missing_root_fails_on_resolution(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        mounts.resolve_mount_roots([str(tmp_path / "gone")])


# --- seccomp requirement -----------------------------------------------------


def test_default_profile_denies_every_mount_syscall():
    assert mounts.profile_denies_mounts(default_profile("aarch64"))
    assert mounts.profile_denies_mounts(default_profile("x86_64"))


def test_profile_missing_one_mount_syscall_does_not_count():
    profile = default_profile("aarch64")
    profile["syscalls"][0]["names"].remove("mount_setattr")
    assert not mounts.profile_denies_mounts(profile)


def test_conditional_rules_do_not_count_as_denial():
    profile = {
        "defaultAction": "SCMP_ACT_ALLOW",
        "syscalls": [
            {
                "names": list(mounts.MOUNT_SYSCALLS),
                "action": "SCMP_ACT_ERRNO",
                "args": [{"index": 0, "value": 1, "op": "SCMP_CMP_EQ"}],
            }
        ],
    }
    assert not mounts.profile_denies_mounts(profile)


def test_deny_by_default_counts_unless_a_rule_names_the_syscall():
    profile = {"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []}
    assert mounts.profile_denies_mounts(profile)
    profile["syscalls"].append({"names": ["mount"], "action": "SCMP_ACT_ALLOW"})
    assert not mounts.profile_denies_mounts(profile)


def test_disabled_and_custom_profiles(tmp_path):
    assert not mounts.seccomp_denies_mounts(None, default_profile("aarch64"))
    assert mounts.seccomp_denies_mounts("default", default_profile("aarch64"))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"defaultAction": "SCMP_ACT_ALLOW", "syscalls": []}))
    assert not mounts.seccomp_denies_mounts(str(path), default_profile("aarch64"))


# --- host-side mount verification -------------------------------------------

MOUNT = ReadonlyMount("/shared/models/q", "/verifier/models/q", "/shared/models/q")


@pytest.fixture
def sandbox_root(tmp_path):
    """A directory standing in for /proc/<init>/root, with the target in it."""
    root = tmp_path / "root"
    (root / "verifier" / "models" / "q").mkdir(parents=True)
    descriptor = os.open(root, os.O_PATH | os.O_DIRECTORY)
    yield SimpleNamespace(path=root, fd=descriptor)
    os.close(descriptor)


def run_report_program(root_fd, targets):
    """The helper exactly as shipped, minus joining a user namespace."""
    program = step_worker._MOUNT_REPORT_PROGRAM
    join = "os.setns(int(sys.argv[1]), CLONE_NEWUSER)\n"
    assert program.count(join) == 1
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", program.replace(join, ""), "0", str(root_fd)]
        + targets,
        capture_output=True,
        pass_fds=(root_fd,),
        check=True,
    )
    return [json.loads(line) for line in result.stdout.decode().splitlines()]


def test_report_program_reports_identity_and_mount_flags(sandbox_root):
    [report] = run_report_program(sandbox_root.fd, [MOUNT.target])
    seen = os.stat(sandbox_root.path / "verifier/models/q")
    assert report == {
        "target": MOUNT.target,
        "dev": seen.st_dev,
        "ino": seen.st_ino,
        "read_only": bool(os.statvfs(sandbox_root.path).f_flag & os.ST_RDONLY),
    }


@pytest.mark.parametrize("swap", ["verifier", "verifier/models", "verifier/models/q"])
def test_report_program_never_follows_symlinks(sandbox_root, tmp_path, swap):
    """An absolute symlink must not resolve against the worker's own root."""
    elsewhere = tmp_path / "elsewhere" / "models" / "q"
    elsewhere.mkdir(parents=True)
    link = sandbox_root.path / swap
    shutil.rmtree(link)
    link.symlink_to(tmp_path / "elsewhere" / Path(swap).relative_to("verifier"))
    [report] = run_report_program(sandbox_root.fd, [MOUNT.target])
    assert report["target"] == MOUNT.target and "error" in report


def test_report_program_reports_missing_targets(sandbox_root):
    reports = run_report_program(sandbox_root.fd, ["/verifier/absent", MOUNT.target])
    assert "error" in reports[0] and "ino" in reports[1]


def report_for(path, **extra):
    seen = os.stat(path)
    return {
        "target": MOUNT.target,
        "dev": seen.st_dev,
        "ino": seen.st_ino,
        "read_only": True,
        **extra,
    }


def test_read_only_bind_of_the_source_passes(tmp_path):
    mounts.check_mount_report(
        report_for(tmp_path), ReadonlyMount("/s", MOUNT.target, str(tmp_path))
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"read_only": False}, "not read-only"),
        ({"ino": -1}, "not a bind of"),
        ({"dev": -1}, "not a bind of"),
        ({"error": "Permission denied"}, "no directory at"),
        ({"target": "/elsewhere"}, "Malformed"),
    ],
)
def test_report_mismatches_are_errors(tmp_path, change, message):
    with pytest.raises(RuntimeError, match=message):
        mounts.check_mount_report(
            report_for(tmp_path, **change),
            ReadonlyMount("/s", MOUNT.target, str(tmp_path)),
        )


def test_a_source_swapped_on_the_host_after_resolution_fails(tree):
    [mount] = mounts.resolve_mounts(
        [{"source": f"{tree.root}/model", "target": MOUNT.target}],
        mounts.resolve_mount_roots([tree.root]),
    )
    os.rename(f"{tree.root}/model", f"{tree.root}/old")
    os.symlink(tree.outside, f"{tree.root}/model")
    with pytest.raises(RuntimeError, match="not a bind of"):
        mounts.check_mount_report(report_for(tree.outside), mount)
    mounts.check_mount_report(report_for(f"{tree.root}/old"), mount)


def test_report_runner_joins_the_pinned_user_namespace(monkeypatch):
    seen = {}

    class Process:
        pid = 4242
        returncode = 0

        def communicate(self, timeout=None):
            line = json.dumps({"target": "/v", "dev": 1, "ino": 2, "read_only": True})
            return (line + "\n").encode(), b""

    def popen(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return Process()

    monkeypatch.setattr(step_worker.subprocess, "Popen", popen)
    assert step_worker.report_mounts(10, 11, ["/v"]) == [
        {"target": "/v", "dev": 1, "ino": 2, "read_only": True}
    ]
    assert seen["argv"][:3] == [sys.executable, "-I", "-S"]
    assert seen["argv"][5:] == ["10", "11", "/v"]
    assert seen["kwargs"]["pass_fds"] == (10, 11)
    assert seen["kwargs"]["start_new_session"] is True
    assert "CLONE_NEWPID" not in seen["argv"][4]  # user namespace only


@pytest.mark.parametrize(
    ("returncode", "stdout", "message"),
    [(1, b"", "helper failed"), (0, b"", "wrong number of reports")],
)
def test_report_runner_failures_are_errors(monkeypatch, returncode, stdout, message):
    class Process:
        pid = 4242

        def __init__(self):
            self.returncode = returncode

        def communicate(self, timeout=None):
            return stdout, b"setns: EPERM"

    monkeypatch.setattr(step_worker.subprocess, "Popen", lambda *a, **k: Process())
    with pytest.raises(RuntimeError, match=message):
        step_worker.report_mounts(10, 11, ["/v"])


def test_requested_network_mounts_are_exempt_only_at_their_targets():
    info = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:61 /export/q /verifier/models/q ro - nfs srv:/export rw\n"
        "3 1 0:61 /export /mnt/leak ro - nfs srv:/export rw\n"
    )
    assert step_worker.network_filesystem_mounts(info) == [
        "/verifier/models/q (nfs)",
        "/mnt/leak (nfs)",
    ]
    assert step_worker.network_filesystem_mounts(
        info, frozenset({"/verifier/models/q"})
    ) == ["/mnt/leak (nfs)"]


# --- ApptainerInstance lifecycle ----------------------------------------------


def disk_instance(monkeypatch, tmp_path):
    monkeypatch.setattr("sandfleet.step_worker.image_digest", lambda *_a: "a" * 64)
    monkeypatch.delenv("SANDFLEET_SANDBOX_SECCOMP", raising=False)
    instance = ApptainerInstance(
        image="/image.sif",
        timeout=60,
        pwd="/workspace",
        cache_dir=None,
        tmp_dir=None,
        extra_start_flags=[],
        apptainer_binary="apptainer",
        storage_mb=64,
        image_sha256="a" * 64,
        storage_root=str(tmp_path),
    )
    instance.overlay = Mock()
    instance.overlay.env.return_value = {}
    instance.name = "only-this-instance"
    return instance


def record_lifecycle(monkeypatch, instance, calls, *, killed=3):
    monkeypatch.setattr(
        instance, "kill_processes", lambda: calls.append(("kill",)) or killed
    )
    monkeypatch.setattr(
        instance,
        "_prepare_mount_points",
        lambda targets: calls.append(("prepare", tuple(targets))),
    )
    monkeypatch.setattr(
        instance, "_stop_keeping_storage", lambda: calls.append(("stop",))
    )
    monkeypatch.setattr(
        instance,
        "_launch",
        lambda *, reuse_overlay: calls.append(
            ("launch", reuse_overlay, list(instance.readonly_mounts))
        ),
    )


def test_add_mounts_kills_prepares_stops_and_relaunches_from_the_overlay(
    monkeypatch, tmp_path
):
    instance = disk_instance(monkeypatch, tmp_path)
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    # The second kill stops anything the overlay's startscript relaunched.
    assert instance.add_readonly_mounts([MOUNT]) == 6
    assert calls == [
        ("kill",),
        ("prepare", ("/verifier/models/q",)),
        ("stop",),
        ("launch", True, [MOUNT]),
        ("kill",),
    ]
    assert instance.readonly_mounts_status == [MOUNT.describe()]
    instance.overlay.remove.assert_not_called()
    instance.overlay.create.assert_not_called()


def test_later_mounts_accumulate_and_may_not_overlap(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    instance.add_readonly_mounts([MOUNT])
    other = ReadonlyMount("/shared/data", "/verifier/data", "/shared/data")
    instance.add_readonly_mounts([other])
    assert calls[-2] == ("launch", True, [MOUNT, other])
    # Earlier targets are prepared again: the sandbox may have swapped a parent.
    assert ("prepare", (MOUNT.target, other.target)) in calls
    nested = ReadonlyMount("/shared/x", "/verifier/models/q/x", "/shared/x")
    calls.clear()
    with pytest.raises(ValueError, match="equal or nested"):
        instance.add_readonly_mounts([nested])
    assert calls == []


def test_writable_tmpfs_sandboxes_are_refused(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    instance.overlay = None
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    with pytest.raises(ValueError, match="disk-overlay"):
        instance.add_readonly_mounts([MOUNT])
    assert calls == []


@pytest.mark.parametrize("setting", ["0", "custom"])
def test_mounts_need_a_seccomp_profile_that_denies_mounting(
    monkeypatch, tmp_path, setting
):
    instance = disk_instance(monkeypatch, tmp_path)
    if setting == "0":
        instance.seccomp = None
    else:
        path = tmp_path / "allow-all.json"
        path.write_text(json.dumps({"defaultAction": "SCMP_ACT_ALLOW"}))
        instance.seccomp = str(path)
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    with pytest.raises(RuntimeError, match="seccomp profile that denies"):
        instance.add_readonly_mounts([MOUNT])
    assert calls == []


def test_targets_may_not_cover_the_working_directory(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    with pytest.raises(ValueError, match="writable path '/workspace'"):
        instance.add_readonly_mounts([ReadonlyMount("/s", "/workspace", "/s")])
    assert calls == []


def test_stopped_sandbox_is_refused(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    instance._instance_stopped = True
    with pytest.raises(RuntimeError, match="not been started"):
        instance.add_readonly_mounts([MOUNT])


def test_failed_kill_leaves_the_sandbox_running(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    monkeypatch.setattr(
        instance, "kill_processes", Mock(side_effect=RuntimeError("SURVIVORS"))
    )
    with pytest.raises(RuntimeError, match="SURVIVORS"):
        instance.add_readonly_mounts([MOUNT])
    assert calls == [] and instance.readonly_mounts == []


def test_failed_kill_after_the_restart_closes_the_sandbox(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    calls = []
    record_lifecycle(monkeypatch, instance, calls)
    kills = iter([3, RuntimeError("SURVIVORS")])

    def kill():
        result = next(kills)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(instance, "kill_processes", kill)
    monkeypatch.setattr(instance, "close", lambda: calls.append(("close",)))
    with pytest.raises(RuntimeError, match="SURVIVORS"):
        instance.add_readonly_mounts([MOUNT])
    assert calls[-1] == ("close",)


def test_restart_reuses_the_overlay_and_binds_every_mount(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    image = tmp_path / "disk" / "writable.img"
    image.parent.mkdir()
    image.write_bytes(b"")
    instance.overlay.image = image
    instance.readonly_mounts = [MOUNT]
    flags = instance._disk_start_flags(reuse=True)
    instance.overlay.create.assert_not_called()
    assert flags[flags.index("--overlay") + 1] == str(image)
    assert "/shared/models/q:/verifier/models/q:ro" in flags
    assert flags.index("/shared/models/q:/verifier/models/q:ro") < flags.index("--net")
    image.unlink()
    with pytest.raises(RuntimeError, match="overlay is missing"):
        instance._disk_start_flags(reuse=True)


def test_fresh_start_creates_a_new_overlay(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    image = tmp_path / "new" / "writable.img"
    image.parent.mkdir()
    instance.overlay.create.return_value = image
    flags = instance._disk_start_flags()
    instance.overlay.create.assert_called_once_with(create_dirs=[])
    assert not any(":ro" in flag and "verifier" in flag for flag in flags)


def test_mount_points_are_prepared_parents_first_without_duplicates(
    monkeypatch, tmp_path
):
    instance = disk_instance(monkeypatch, tmp_path)
    seen = {}

    def fake_exec(argv, *, pwd=None, **_kw):
        seen.update(argv=argv, pwd=pwd)
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(instance, "exec", fake_exec)
    instance._prepare_mount_points(["/verifier/models/a", "/verifier/data"])
    assert seen["pwd"] == "/"
    assert seen["argv"][:2] == ["sh", "-c"]
    assert seen["argv"][4:] == [
        "/verifier",
        "/verifier/models",
        "/verifier/models/a",
        "/verifier/data",
    ]
    script = seen["argv"][2]
    assert '[ -L "$p" ]' in script and 'rm -f -- "$p"' in script
    assert "rm -rf" not in script


def test_failed_mount_point_preparation_is_an_error(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance,
        "exec",
        lambda *_a, **_kw: SimpleNamespace(returncode=1, stderr=b"read-only"),
    )
    with pytest.raises(RuntimeError, match="mount points: read-only"):
        instance._prepare_mount_points(["/verifier/x"])


def test_stop_for_restart_keeps_the_overlay(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    stopped = []
    network = Mock()
    instance.network = network
    monkeypatch.setattr(instance, "_stop_instance", lambda: stopped.append(True))
    instance._stop_keeping_storage()
    assert stopped == [True] and instance._instance_stopped
    network.close.assert_called_once_with()
    assert instance.network is None
    instance.overlay.remove.assert_not_called()
    assert instance.name == "only-this-instance"


def test_stop_for_restart_refuses_with_live_drain_threads(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    monkeypatch.setattr(instance, "_stop_instance", lambda: None)
    monkeypatch.setattr(instance, "join_drain_threads", lambda: [object()])
    with pytest.raises(RuntimeError, match="refusing to restart"):
        instance._stop_keeping_storage()


def test_restart_failure_closes_the_sandbox(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    instance._instance_stopped = True
    instance.readonly_mounts = [MOUNT]
    monkeypatch.setattr(instance, "_disk_start_flags", lambda reuse: ("--x",))
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stderr=b""),
    )
    monkeypatch.setattr(instance, "_start_network", lambda: None)
    monkeypatch.setattr(instance, "_bootstrap", lambda: None)
    monkeypatch.setattr(
        instance,
        "_verify_readonly_mounts",
        Mock(side_effect=RuntimeError("not read-only")),
    )
    closed = []
    monkeypatch.setattr(instance, "close", lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="not read-only"):
        instance._launch(reuse_overlay=True)
    assert closed == [True]


def test_mounts_are_verified_through_the_inits_root(monkeypatch, tmp_path):
    instance = disk_instance(monkeypatch, tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    mount = ReadonlyMount(str(source), "/verifier/models/q", str(source))
    instance.readonly_mounts = [mount]
    monkeypatch.setattr(instance, "_instance_pid", lambda: os.getpid())
    instance.pid_ns_fd = os.open("/proc/self/ns/pid", os.O_RDONLY)
    instance.user_ns_fd = 77
    seen = []

    def report(user_ns_fd, root, targets):
        seen.append((user_ns_fd, os.fstat(root).st_ino, targets))
        return [report_for(source, target=mount.target)]

    monkeypatch.setattr(step_worker, "report_mounts", report)
    table = (
        "1 0 0:1 / / rw - overlay o rw\n2 1 0:2 / /verifier/models/q ro - nfs s rw\n"
    )
    monkeypatch.setattr(step_worker, "read_mountinfo", lambda pid: table)
    try:
        instance._verify_readonly_mounts()
        assert seen == [(77, os.stat("/").st_ino, ["/verifier/models/q"])]
        for bad, message in (
            (table + "3 2 0:3 / /verifier/models/q/sub rw - xfs d rw\n", "below"),
            (table.replace(" ro - nfs", " rw - nfs"), "not read-only"),
            (table + "3 1 0:3 / /mnt rw - nfs s rw\n", "network or FUSE"),
        ):
            monkeypatch.setattr(step_worker, "read_mountinfo", lambda pid, t=bad: t)
            with pytest.raises(RuntimeError, match=message):
                instance._verify_readonly_mounts()
        monkeypatch.setattr(step_worker, "read_mountinfo", lambda pid: table)
        monkeypatch.setattr(
            step_worker,
            "report_mounts",
            lambda *_a: [report_for(source, target=mount.target, read_only=False)],
        )
        with pytest.raises(RuntimeError, match="not read-only"):
            instance._verify_readonly_mounts()
    finally:
        os.close(instance.pid_ns_fd)


def test_verification_refuses_an_init_outside_the_pinned_namespace(
    monkeypatch, tmp_path
):
    instance = disk_instance(monkeypatch, tmp_path)
    instance.readonly_mounts = [MOUNT]
    monkeypatch.setattr(instance, "_instance_pid", lambda: os.getpid())
    monkeypatch.setattr(step_worker, "report_mounts", Mock(side_effect=AssertionError))
    instance.pid_ns_fd = os.open("/proc/self/ns/net", os.O_RDONLY)
    try:
        with pytest.raises(RuntimeError, match="pinned PID namespace"):
            instance._verify_readonly_mounts()
    finally:
        os.close(instance.pid_ns_fd)


# --- worker session ---------------------------------------------------------


class FakeMountInstance:
    def __init__(self):
        self.name = "inst"
        self.storage_status = None
        self.network_metadata = {}
        self.readonly_mounts = []
        self.added = []

    @property
    def readonly_mounts_status(self):
        return [mount.describe() for mount in self.readonly_mounts]

    def add_readonly_mounts(self, items):
        self.added.append(items)
        self.readonly_mounts += items
        return 2


@pytest.fixture
def session(monkeypatch, tree):
    monkeypatch.setattr(step_worker, "current_cgroup", lambda **_kw: {})
    monkeypatch.setattr(step_worker, "memory_events", lambda: {"oom": 0, "oom_kill": 0})
    value = WorkerSession(readonly_mount_roots=(tree.root,))
    value.instance = FakeMountInstance()
    return value


def test_session_adds_resolved_mounts_and_reports_them(session, tree):
    result, done = session.dispatch(
        {
            "operation": "add_readonly_mounts",
            "mounts": [{"source": f"{tree.root}/model", "target": "/verifier/m"}],
        }
    )
    assert not done
    [[mount]] = session.instance.added
    assert mount.host_path == os.path.realpath(f"{tree.root}/model")
    assert result["stopped_processes"] == 2
    assert result["added_mounts"] == [mount.describe()]
    assert result["readonly_mounts"] == [mount.describe()]
    assert result["readonly_mount_roots"] == [tree.root]


def test_session_refuses_sources_outside_its_roots(session, tree):
    with pytest.raises(PermissionError, match="allowlisted"):
        session.dispatch(
            {
                "operation": "add_readonly_mounts",
                "mounts": [{"source": tree.outside, "target": "/verifier/m"}],
            }
        )
    assert session.instance.added == []


def test_session_without_roots_refuses(monkeypatch, tree):
    monkeypatch.setattr(step_worker, "current_cgroup", lambda **_kw: {})
    monkeypatch.setattr(step_worker, "memory_events", lambda: {"oom": 0, "oom_kill": 0})
    session = WorkerSession()
    session.instance = FakeMountInstance()
    with pytest.raises(ValueError, match="not enabled"):
        session.dispatch(
            {
                "operation": "add_readonly_mounts",
                "mounts": [{"source": f"{tree.root}/model", "target": "/v"}],
            }
        )
    assert session._status({})["readonly_mount_roots"] == []


def test_session_requires_a_started_sandbox(session, tree):
    session.instance = None
    with pytest.raises(RuntimeError, match="not been started"):
        session.dispatch(
            {
                "operation": "add_readonly_mounts",
                "mounts": [{"source": f"{tree.root}/model", "target": "/v"}],
            }
        )


def test_session_rejects_malformed_roots():
    with pytest.raises(ValueError):
        WorkerSession(readonly_mount_roots=("relative",))


# --- step backend -----------------------------------------------------------


def status_result(**extra):
    return {
        "instance_name": "inst-2",
        "network": {},
        "cgroup": {
            "path": "/slurm/job/step/task",
            "memory.max.effective": str(512 * 1024 * 1024),
            "cpuset.cpus.effective": "7",
        },
        "readonly_mounts": [],
        "readonly_mount_roots": ["/models"],
        "agent_readonly_mount_roots": [],
        **extra,
    }


def test_backend_add_mounts_records_the_new_status(monkeypatch):
    backend = SlurmStepSandbox(cpus=1, memory_mb=512, readonly_mount_roots=("/models",))
    described = [{"source": "/models/q", "target": "/v/q", "read_only": True}]
    seen = {}

    def rpc(operation, *, response_timeout, **payload):
        seen.update(operation=operation, timeout=response_timeout, payload=payload)
        return status_result(
            readonly_mounts=described, added_mounts=described, stopped_processes=4
        )

    monkeypatch.setattr(backend, "_rpc", rpc)
    request = [{"source": "/models/q", "target": "/v/q"}]
    result = backend.add_readonly_mounts(request)
    assert seen == {
        "operation": "add_readonly_mounts",
        "timeout": ADD_READONLY_MOUNTS_TIMEOUT_SECONDS,
        "payload": {"mounts": request},
    }
    assert result["instance_name"] == "inst-2"
    assert result["readonly_mounts"] == described
    assert result["added_mounts"] == described and result["stopped_processes"] == 4
    assert backend.status()["readonly_mount_roots"] == ["/models"]


def test_backend_rejects_a_worker_with_another_mount_policy():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512, readonly_mount_roots=("/models",))
    with pytest.raises(RuntimeError, match="mount policy/version mismatch"):
        backend._record_status(status_result(readonly_mount_roots=[]))


def test_backend_passes_roots_to_the_step_worker(monkeypatch):
    backend = SlurmStepSandbox(
        cpus=1,
        memory_mb=512,
        readonly_mount_roots=("/models", "/data"),
        start_timeout=0,
    )
    launched = {}

    class Process:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

        def kill(self):
            pass

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr("sandfleet.step_backend.shutil.which", lambda _name: "/srun")
    monkeypatch.setattr(
        "sandfleet.step_backend.subprocess.Popen",
        lambda command, **_kw: launched.update(command=command) or Process(),
    )
    with pytest.raises(TimeoutError):
        backend._launch_worker()
    command = launched["command"]
    assert " ".join(command).endswith(
        "--readonly-mount-root /models --readonly-mount-root /data"
    )


# --- agent ------------------------------------------------------------------


@pytest.fixture
def agent_server(monkeypatch, tmp_path):
    log = AuditLog(tmp_path / "w.jsonl", max_bytes=1 << 20, context={"pool": "p"})
    state = AgentState(
        service_token="service-secret",
        slots=1,
        cpus=1,
        memory_mb=4096,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
        audit=log,
        readonly_mount_roots=("/models",),
    )
    sandbox = FakeSandbox()
    sandbox.add_readonly_mounts = Mock(
        return_value={"instance_name": "fake-2", "stopped_processes": 3}
    )
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    granted = request_json(
        url, "/v1/control/leases", method="POST", token="service-secret", payload={}
    )

    def add(payload):
        return request_json(
            url,
            f"/v1/leases/{granted['lease_id']}/add-mounts",
            method="POST",
            token=granted["lease_token"],
            payload=payload,
            timeout=5,
        )

    try:
        yield SimpleNamespace(
            add=add,
            sandbox=sandbox,
            state=state,
            lease=granted,
            log=tmp_path / "w.jsonl",
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def audit_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_agent_add_mounts_is_validated_forwarded_and_audited(agent_server):
    request = [{"source": "/models/q", "target": "/verifier/q"}]
    assert agent_server.add({"mounts": request}) == {
        "instance_name": "fake-2",
        "stopped_processes": 3,
    }
    agent_server.sandbox.add_readonly_mounts.assert_called_once_with(request)
    [record] = audit_records(agent_server.log)
    assert record["op"] == "add-mounts"
    assert record["mounts"] == request and record["stopped_processes"] == 3
    assert record["lease_id"] == agent_server.lease["lease_id"]
    assert "error" not in record


def test_agent_refuses_malformed_mounts_and_audits_the_refusal(agent_server):
    with pytest.raises(Exception, match="reserved"):
        agent_server.add({"mounts": [{"source": "/models/q", "target": "/etc/q"}]})
    agent_server.sandbox.add_readonly_mounts.assert_not_called()
    [record] = audit_records(agent_server.log)
    assert record["op"] == "add-mounts" and "reserved" in record["error"]
    assert isinstance(record["mounts"], str) and "/etc/q" in record["mounts"]


def test_agent_add_mounts_waits_for_the_exec_lock(agent_server):
    lock = agent_server.state.leases[agent_server.lease["lease_id"]].lock
    outcome = {}
    with lock:
        thread = threading.Thread(
            target=lambda: outcome.update(
                result=agent_server.add(
                    {"mounts": [{"source": "/models/q", "target": "/v/q"}]}
                )
            )
        )
        thread.start()
        thread.join(timeout=0.5)
        assert thread.is_alive()  # an active exec holds the lease lock
        agent_server.sandbox.add_readonly_mounts.assert_not_called()
    thread.join(timeout=5)
    assert outcome["result"]["stopped_processes"] == 3


def test_agent_passes_roots_to_new_steps(monkeypatch):
    state = AgentState(
        service_token="s",
        slots=1,
        cpus=1,
        memory_mb=1024,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
        readonly_mount_roots=("/models",),
    )
    assert state.new_sandbox().readonly_mount_roots == ("/models",)


def test_agent_cli_rejects_malformed_roots():
    parser = build_parser()
    base = ["--controller-url", "http://c:1", "--pool", "p", "--slots", "1"]
    base += ["--cpus", "1", "--memory-mb", "64"]
    args = parser.parse_args(base + ["--readonly-mount-root", "/models"])
    validate_args(parser, args)
    assert args.readonly_mount_roots == ["/models"]
    args = parser.parse_args(base + ["--readonly-mount-root", "models"])
    with pytest.raises(SystemExit):
        validate_args(parser, args)


# --- controller and client --------------------------------------------------


def test_worker_submission_passes_each_root(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_: seen.append(command)
        or SimpleNamespace(returncode=0, stdout="1\n", stderr=""),
    )
    configured = replace(settings(tmp_path), readonly_mount_roots=("/models", "/d"))
    for value in (settings(tmp_path), configured):
        SlurmLauncher(
            value, service_token_file="/t", controller_url="http://c:1"
        ).submit_worker_batch(pool="p", worker_count=1, slots=1, cpus=1, memory_mb=64)
    assert "--readonly-mount-root" not in seen[0][-1]
    assert "--readonly-mount-root /models --readonly-mount-root /d" in seen[1][-1]


def test_controller_settings_reject_malformed_roots(tmp_path):
    with pytest.raises(ValueError):
        replace(settings(tmp_path), readonly_mount_roots=("relative",))


def test_controller_parser_collects_roots():
    from sandfleet.controller import build_parser as controller_parser

    args = controller_parser().parse_args(
        [
            *("--partition", "p", "--account", "a", "--source-root", "/s"),
            *("--log-dir", "/l", "--sandbox-readonly-mount-root", "/models"),
            *("--sandbox-readonly-mount-root", "/data"),
        ]
    )
    assert args.sandbox_readonly_mount_roots == ["/models", "/data"]


def test_client_add_mounts_hits_the_route(monkeypatch, lease):  # noqa: F811
    seen = {}

    def request(path, *, payload, timeout):
        seen.update(path=path, payload=payload, timeout=timeout)
        return {"stopped_processes": 1}

    monkeypatch.setattr(lease, "_request", request)
    mounts_request = [{"source": "/models/q", "target": "/v/q"}]
    assert lease.add_readonly_mounts(mounts_request) == {"stopped_processes": 1}
    assert seen == {
        "path": "/add-mounts",
        "payload": {"mounts": mounts_request},
        "timeout": ADD_MOUNTS_TIMEOUT,
    }
    assert ADD_MOUNTS_TIMEOUT > ADD_READONLY_MOUNTS_TIMEOUT_SECONDS
