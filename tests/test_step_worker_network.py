"""Regression coverage for network ownership inside a persistent sandbox step."""

import io
import json
import os
import socket
import subprocess
from unittest.mock import Mock

import pytest

from sandfleet import step_worker


class _FakeStat:
    def __init__(self, inode, dev=7):
        self.st_ino = inode
        self.st_dev = dev


HOST_NAMESPACE = "net:[100]"
SANDBOX_NAMESPACE = "net:[200]"
HOST_PID_NAMESPACE = "pid:[100]"
SANDBOX_PID_NAMESPACE = "pid:[200]"
HOST_PID_INODE = 100
HOST_USER_INODE = 101
SANDBOX_PID_INODE = 200
SANDBOX_USER_INODE = 300
ROUTE_HEADER = (
    "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"
)
PROXY_URL = "http://lease:secret@127.0.0.1:4567"
SANDBOX_PROXY_URL = "http://lease:secret@127.0.0.1:3128"
# The /32 forwarder route 0.9 internet sandboxes had; 0.11 allows no route at all.
FORWARDER_ROUTE = "sfnet0\t0302010A\t0202000A\t0007\t0\t0\t0\tFFFFFFFF\t0\t0\t0\n"
# Mount table and status of a confined sandbox, as the startup probe reads them.
SANDBOX_MOUNTINFO = (
    "1 0 0:1 / / rw - fuse.fuse-overlayfs fuse-overlayfs rw\n"
    "2 1 0:2 / /proc rw - proc proc rw\n"
    "3 1 9:127 /run/resolv.conf /etc/resolv.conf ro - xfs /dev/md127 ro\n"
)
HOST_RESOLVER = "nameserver 192.0.2.53\nsearch example.test\n"
CONFIG = {
    "image": "/shared/image.sif",
    "timeout": 30,
    "pwd": "/workspace",
    "cache_dir": None,
    "tmp_dir": None,
    "extra_start_flags": [],
    "apptainer_binary": "apptainer",
}
INTERNET_CONFIG = CONFIG | {"proxy_url": PROXY_URL}


class FakeNetwork:
    def __init__(self, harness):
        self.harness = harness
        self.pid = 9000 + len(harness.helpers)
        self.target = None
        self.start_error = None
        self.health_error = None
        self.close_error = None

    def start(self, instance_pid, namespace):
        self.harness.events.append("network:start")
        self.target = instance_pid
        self.namespace = namespace
        if self.start_error is not None:
            raise self.start_error

    def ensure_alive(self):
        self.harness.events.append("network:health")
        if self.health_error is not None:
            raise self.health_error

    def close(self):
        self.harness.events.append("network:close")
        if self.close_error is not None:
            raise self.close_error


class _FakePopen:
    """Minimal Popen stand-in for the drained exec path."""

    def __init__(self, args, returncode, stdout, stderr):
        self.args = args
        self.returncode = returncode
        self.pid = 4242
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.stdin = io.BytesIO()

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        pass


class InstanceHarness:
    def __init__(self):
        self.calls = []
        self.events = []
        self.helpers = []
        self.name = None
        self.instance_pid = 8000
        self.namespace = SANDBOX_NAMESPACE
        self.pid_inode = SANDBOX_PID_INODE
        self.user_inode = SANDBOX_USER_INODE
        # PID namespace the verification exec reports (normally the pinned one).
        self.exec_pid_namespace = SANDBOX_PID_NAMESPACE
        self.exec_namespace = SANDBOX_NAMESPACE
        # Fake namespace descriptors the pinning code opens/fstats/closes.
        self._ns_fds = {}
        self._next_ns_fd = 5000
        # None reports the routes the last namespace configuration created.
        self.routes = None
        self.failures = {}
        # Start failure for the next relay the instance creates.
        self.next_start_error = None
        # Seccomp mode the probe reports; None reports 2 iff --security was used.
        self.seccomp = None
        self.started_with_seccomp = False
        self.mountinfo = SANDBOX_MOUNTINFO

    def network(self, upstream, *, max_pending):
        helper = FakeNetwork(self)
        helper.upstream = upstream
        helper.max_pending = max_pending
        helper.start_error = self.next_start_error
        self.helpers.append(helper)
        return helper

    def readlink(self, path):
        if str(path) == "/proc/self/ns/net":
            return HOST_NAMESPACE
        assert str(path) == f"/proc/{self.instance_pid}/ns/net"
        return self.namespace

    def _ns_inode(self, text):
        if text == "/proc/self/ns/pid":
            return HOST_PID_INODE
        if text == "/proc/self/ns/user":
            return HOST_USER_INODE
        if text == f"/proc/{self.instance_pid}/ns/pid":
            return self.pid_inode
        if text == f"/proc/{self.instance_pid}/ns/user":
            return self.user_inode
        return None

    def ns_open(self, path, *args, **kwargs):
        inode = self._ns_inode(str(path))
        if inode is None:
            return self._orig_open(path, *args, **kwargs)
        fd = self._next_ns_fd
        self._next_ns_fd += 1
        self._ns_fds[fd] = inode
        return fd

    def ns_fstat(self, fd, *args, **kwargs):
        if fd in self._ns_fds:
            return _FakeStat(self._ns_fds[fd])
        return self._orig_fstat(fd, *args, **kwargs)

    def ns_pidfd_open(self, _pid, *args, **kwargs):
        fd = self._next_ns_fd
        self._next_ns_fd += 1
        self._ns_fds[fd] = None  # tracked so ns_close ignores the real descriptor
        return fd

    def ns_close(self, fd, *args, **kwargs):
        if fd in self._ns_fds:
            self._ns_fds.pop(fd, None)
            return None
        return self._orig_close(fd, *args, **kwargs)

    def run(self, command, **kwargs):
        self.calls.append((command, kwargs))
        operation, output = self._result(command)
        self.events.append(operation)
        failure = self.failures.get(operation)
        if isinstance(failure, Exception):
            raise failure
        return subprocess.CompletedProcess(
            command,
            returncode=1 if failure else 0,
            stdout=output,
            stderr=str(failure).encode() if failure else b"",
        )

    def popen(self, command, **kwargs):
        """Popen counterpart of run().

        _exec_internal drains stdout/stderr itself instead of calling
        subprocess.run(capture_output=True), because communicate() waits for
        pipe EOF and a command that backgrounds a process therefore blocks
        forever. The harness has to model Popen too, or exec escapes the mock
        and invokes a real apptainer. Same call recording, same _result()
        dispatch and same failure injection as run(), so every existing
        assertion keeps working.
        """
        self.calls.append((command, kwargs))
        operation, output = self._result(command)
        self.events.append(operation)
        failure = self.failures.get(operation)
        if isinstance(failure, Exception):
            raise failure
        return _FakePopen(
            command,
            1 if failure else 0,
            output,
            str(failure).encode() if failure else b"",
        )

    def _result(self, command):
        if command[1:3] == ["instance", "start"]:
            self.name = command[-1]
            self.started_with_seccomp = "--security" in command
            return "instance:start", b""
        if command[1:3] == ["instance", "list"]:
            payload = {"instances": [{"instance": self.name, "pid": self.instance_pid}]}
            return "instance:list", json.dumps(payload).encode()
        if command[1:3] == ["instance", "stop"]:
            return "instance:stop", b""
        assert command[1] == "exec"
        if "readlink /proc/self/ns/net" in " ".join(command):
            routes = ROUTE_HEADER if self.routes is None else self.routes
            seccomp = self.seccomp
            if seccomp is None:
                seccomp = 2 if self.started_with_seccomp else 0
            split = "--sandfleet-verify--\n"
            status = f"Name:\tsh\nSeccomp:\t{seccomp}\nSeccomp_filters:\t1\n"
            mounts = (
                self.mountinfo
                if self.mountinfo.endswith("\n")
                else (self.mountinfo + "\n")
            )
            output = (
                f"{self.exec_namespace}\n{routes}{split}{status}{split}{mounts}"
                f"{split}{self.exec_pid_namespace}\n"
            )
            return "network:probe", output.encode()
        apt = command[-1].startswith(step_worker.APT_CONFIG_SCRIPT)
        if "mkdir" in command or apt:
            return "instance:apt" if apt else "instance:mkdir", b""
        return "instance:exec", b"command result\n"


@pytest.fixture
def harness(monkeypatch):
    value = InstanceHarness()
    monkeypatch.setattr(step_worker, "LoopbackProxy", value.network)
    monkeypatch.setattr(step_worker.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(step_worker.subprocess, "run", value.run)
    monkeypatch.setattr(step_worker.subprocess, "Popen", value.popen)
    original_readlink = os.readlink

    def readlink(path, **kwargs):
        if str(path).startswith("/proc/"):
            return value.readlink(path)
        return original_readlink(path, **kwargs)

    monkeypatch.setattr(step_worker.os, "readlink", readlink)
    value._orig_open = os.open
    value._orig_fstat = os.fstat
    value._orig_close = os.close
    monkeypatch.setattr(step_worker.os, "open", value.ns_open)
    monkeypatch.setattr(step_worker.os, "fstat", value.ns_fstat)
    monkeypatch.setattr(step_worker.os, "close", value.ns_close)
    monkeypatch.setattr(step_worker.os, "pidfd_open", value.ns_pidfd_open)
    monkeypatch.setattr(step_worker.signal, "pidfd_send_signal", lambda *_a, **_k: None)
    monkeypatch.setattr(
        step_worker, "current_cgroup", lambda **_kwargs: {"path": "/test/cgroup"}
    )
    monkeypatch.setattr(step_worker, "memory_events", lambda: {"oom": 0, "oom_kill": 0})
    return value


def make_instance(**overrides):
    return step_worker.ApptainerInstance(**(CONFIG | overrides))


def make_internet_instance(**overrides):
    return make_instance(proxy_url=PROXY_URL, **overrides)


def test_offline_start_has_only_loopback_and_no_helper(harness):
    instance = make_instance(extra_start_flags=["--bind", "/shared:/shared"])
    instance.start()

    command = harness.calls[0][0]
    assert "--net" in command
    assert "--network=none" in command
    assert not any(flag.startswith("--dns") for flag in command)
    assert "/shared:/shared" in command
    assert instance.name == harness.name
    assert harness.helpers == []
    assert instance.network is None
    assert harness.events.index("network:probe") < harness.events.index(
        "instance:mkdir"
    )
    assert instance.network_metadata == {
        "mode": "none",
        "namespace": SANDBOX_NAMESPACE,
        "host_namespace": HOST_NAMESPACE,
        "helper_pid": None,
        "internet": False,
    }


def test_internet_start_has_no_routes_and_a_loopback_proxy(harness):
    instance = make_internet_instance()
    instance.start()

    command = harness.calls[0][0]
    assert "--network=none" in command
    assert not any(flag.startswith("--dns") for flag in command)
    assert harness.helpers[0].target == harness.instance_pid
    assert harness.helpers[0].upstream == ("127.0.0.1", 4567)
    assert harness.helpers[0].namespace == SANDBOX_NAMESPACE
    assert harness.helpers[0].max_pending == 1
    assert (
        harness.events.index("instance:start")
        < harness.events.index("network:start")
        < harness.events.index("network:probe")
        < harness.events.index("instance:mkdir")
    )
    assert instance.network_metadata == {
        "mode": "loopback",
        "namespace": SANDBOX_NAMESPACE,
        "host_namespace": HOST_NAMESPACE,
        "helper_pid": None,
        "internet": True,
    }


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("value", [None, "0"])
def test_instance_start_drops_network_caps(harness, monkeypatch, tmp_path, disk, value):
    if value is None:
        monkeypatch.delenv("SANDFLEET_SANDBOX_DROP_NET_CAPS", raising=False)
    else:
        monkeypatch.setenv("SANDFLEET_SANDBOX_DROP_NET_CAPS", value)
    if disk:
        disk_instance(harness, monkeypatch, tmp_path)
    else:
        make_instance().start()
    command = next(
        command for command, _ in harness.calls if command[1:3] == ["instance", "start"]
    )
    if value is None:
        assert (
            command[command.index("--drop-caps") + 1]
            == "CAP_NET_ADMIN,CAP_NET_RAW,CAP_SYS_PTRACE"
        )
    else:
        assert "--drop-caps" not in command


def test_internet_execs_drop_ptrace_without_setuid_caps(harness, monkeypatch):
    monkeypatch.setenv("SANDFLEET_SANDBOX_DROP_SETUID_CAPS", "0")
    monkeypatch.delenv("SANDFLEET_SANDBOX_DROP_NET_CAPS", raising=False)
    instance = make_internet_instance()
    instance.start()
    instance.exec(["true"])
    start = next(
        command for command, _ in harness.calls if command[1:3] == ["instance", "start"]
    )
    execs = [command for command, _ in harness.calls if command[1] == "exec"]
    for command in [start, *execs]:
        caps = command[command.index("--drop-caps") + 1].split(",")
        assert "CAP_SYS_PTRACE" in caps
        assert "CAP_SETUID" not in caps


def test_start_request_from_a_pre_0_9_agent_is_explained(harness):
    session = step_worker.WorkerSession()
    with pytest.raises(RuntimeError, match="pre-0.9 agent"):
        session._start({"config": CONFIG | {"network_dns": None}})
    assert harness.calls == []


def test_internet_lease_requires_dropped_network_capabilities(harness, monkeypatch):
    monkeypatch.setenv("SANDFLEET_SANDBOX_DROP_NET_CAPS", "0")
    with pytest.raises(ValueError, match="SANDFLEET_SANDBOX_DROP_NET_CAPS"):
        make_internet_instance()
    make_instance().start()
    assert harness.calls


def test_proxy_listener_failure_stops_instance(harness):
    instance = make_internet_instance()
    harness.next_start_error = RuntimeError("Address already in use")

    with pytest.raises(RuntimeError, match="Address already in use"):
        instance.start()

    assert "network:probe" not in harness.events
    assert harness.events.count("instance:stop") == 1
    assert "network:close" in harness.events
    assert instance.name is None


def test_start_keeps_the_images_tmp(harness):
    make_instance().start()

    command = harness.calls[0][0]
    assert "--writable-tmpfs" in command
    assert command[command.index("--no-mount") + 1] == "tmp"


def disk_instance(harness, monkeypatch, tmp_path, **overrides):
    monkeypatch.setattr(step_worker, "image_digest", lambda *_: "a" * 64)
    original = step_worker.Path.read_text

    def read_text(path, *args, **kwargs):
        if str(path) == "/etc/resolv.conf":
            return HOST_RESOLVER
        return original(path, *args, **kwargs)

    monkeypatch.setattr(step_worker.Path, "read_text", read_text)
    instance = make_instance(
        storage_mb=64, image_sha256="a" * 64, storage_root=str(tmp_path), **overrides
    )
    instance.overlay = Mock()
    instance.overlay.env.return_value = {}
    instance.overlay.create.return_value = tmp_path / "owned.img"
    monkeypatch.setattr(instance, "_verify_storage", Mock())
    instance.start()
    instance.overlay.remove.reset_mock()
    monkeypatch.setattr(
        instance,
        "exec_instance_list",
        Mock(side_effect=[[{"instance": instance.name}], []]),
    )
    return instance


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("block", [None, "0"])
@pytest.mark.parametrize("block_user_switch", [None, "0"])
def test_start_configures_apt_to_stay_root(
    harness, monkeypatch, tmp_path, disk, block, block_user_switch
):
    for name, value in (
        ("SANDFLEET_SANDBOX_BLOCK_APT", block),
        ("SANDFLEET_SANDBOX_BLOCK_USER_SWITCH", block_user_switch),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    if disk:
        disk_instance(harness, monkeypatch, tmp_path)
    else:
        make_instance().start()
    command, options = next(
        call
        for call in harness.calls
        if call[0][-1].startswith(step_worker.APT_CONFIG_SCRIPT)
    )
    script = step_worker.APT_CONFIG_SCRIPT + " || exit"
    if block is None:
        script += step_worker.APT_BLOCK_SCRIPT
    if block_user_switch is None:
        script += step_worker.USER_SWITCH_BLOCK_SCRIPT
    assert command[-3:] == ["sh", "-c", script]
    assert options["env"]["APPTAINER_PWD"] == "/"
    assert harness.events.index("network:probe") < harness.events.index("instance:apt")


@pytest.mark.parametrize("disk", [False, True])
@pytest.mark.parametrize("setuid", [None, "0"])
@pytest.mark.parametrize("net", [None, "0"])
def test_every_exec_drops_setuid_and_network_caps_by_default(
    harness, monkeypatch, tmp_path, disk, setuid, net
):
    for name, value in (
        ("SANDFLEET_SANDBOX_DROP_SETUID_CAPS", setuid),
        ("SANDFLEET_SANDBOX_DROP_NET_CAPS", net),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    expected = []
    if setuid is None:
        expected += ["CAP_SETUID", "CAP_SETGID", "CAP_SYS_PTRACE"]
    if net is None:
        expected += ["CAP_NET_ADMIN", "CAP_NET_RAW"]
        if setuid is not None:
            expected.append("CAP_SYS_PTRACE")
    if disk:
        instance = disk_instance(harness, monkeypatch, tmp_path)
    else:
        instance = make_instance()
        instance.start()
    instance.exec(["true"], command_env={"A": "1"})
    instance.cancel_command_group("command-1")
    execs = [command for command, _ in harness.calls if command[1] == "exec"]
    assert len(execs) >= 4
    for command in execs:
        flags = command[2 : command.index(f"instance://{instance.name}")]
        if expected:
            assert flags[:2] == ["--drop-caps", ",".join(expected)]
        else:
            assert "--drop-caps" not in flags


def run_apt_block_script(bin_dir, prefix="true"):
    script = prefix + " || exit" + step_worker.APT_BLOCK_SCRIPT
    return subprocess.run(
        ["sh", "-c", script.replace("/usr/bin", str(bin_dir))], capture_output=True
    )


def test_apt_block_script_writes_stubs(tmp_path):
    assert run_apt_block_script(tmp_path).returncode == 0
    for name in ("apt", "apt-get", "aptitude"):
        stub = tmp_path / name
        assert stub.stat().st_mode & 0o777 == 0o755
        result = subprocess.run([stub, "update"], capture_output=True)
        assert (result.returncode, result.stdout) == (100, b"")
        assert result.stderr == b"E: apt is disabled in this sandbox.\n"


def test_apt_block_script_never_fails(tmp_path):
    missing = tmp_path / "missing"
    assert run_apt_block_script(missing).returncode == 0
    assert not missing.exists()
    tmp_path.chmod(0o500)
    try:
        assert run_apt_block_script(tmp_path).returncode == 0
        assert list(tmp_path.iterdir()) == []
    finally:
        tmp_path.chmod(0o700)
    # A failed apt.conf write still fails bootstrap.
    assert run_apt_block_script(tmp_path, prefix="false").returncode == 1


def run_user_switch_block_script(dirs):
    script = step_worker.USER_SWITCH_BLOCK_SCRIPT.replace(
        "/usr/bin /usr/sbin /bin /sbin", " ".join(map(str, dirs))
    )
    return subprocess.run(["sh", "-c", script], capture_output=True)


def test_user_switch_block_script_replaces_binaries(tmp_path):
    shell = tmp_path / "busybox"
    shell.write_text("#!/bin/sh\necho busybox\n")
    bin_dir, sbin_dir = tmp_path / "bin", tmp_path / "sbin"
    bin_dir.mkdir()
    sbin_dir.mkdir()
    (bin_dir / "su").symlink_to(shell)
    dirs = [bin_dir, sbin_dir, tmp_path / "missing"]
    assert run_user_switch_block_script(dirs).returncode == 0
    # The symlink is replaced, not written through.
    assert shell.read_text() == "#!/bin/sh\necho busybox\n"
    assert not (tmp_path / "missing").exists()
    for directory in (bin_dir, sbin_dir):
        for name in ("su", "sudo", "runuser", "setpriv", "chroot"):
            stub = directory / name
            assert not stub.is_symlink()
            assert stub.stat().st_mode & 0o777 == 0o755
            result = subprocess.run([stub, "user", "-c", "id"], capture_output=True)
            assert (result.returncode, result.stdout) == (1, b"")
            assert result.stderr == b"E: switching users is disabled in this sandbox.\n"


def test_user_switch_block_script_never_fails(tmp_path):
    tmp_path.chmod(0o500)
    try:
        assert run_user_switch_block_script([tmp_path]).returncode == 0
        assert list(tmp_path.iterdir()) == []
    finally:
        tmp_path.chmod(0o700)


def test_apt_config_failure_fails_start_and_cleans_up(harness):
    harness.failures["instance:apt"] = "read-only file system"
    instance = make_internet_instance()

    with pytest.raises(RuntimeError, match="could not configure apt: read-only"):
        instance.start()

    assert harness.events.count("instance:stop") == 1
    assert "network:close" in harness.events
    assert instance.name is None


@pytest.mark.parametrize("internet", [False, True])
def test_disk_start_keeps_network_flags_and_checks(
    harness, monkeypatch, tmp_path, internet
):
    instance = disk_instance(
        harness, monkeypatch, tmp_path, proxy_url=PROXY_URL if internet else None
    )
    command = next(
        command for command, _ in harness.calls if command[1:3] == ["instance", "start"]
    )
    assert "--net" in command
    assert "--network=none" in command
    assert not any(flag.startswith("--dns") for flag in command)
    assert command[command.index("--overlay") + 1] == str(tmp_path / "owned.img")
    assert (
        command[command.index("--bind") + 1]
        == f"{tmp_path}/resolv.conf:/etc/resolv.conf:ro"
    )
    # A snapshot of the host resolver, which the sandbox has no route to.
    assert (tmp_path / "resolv.conf").read_text() == HOST_RESOLVER
    assert (tmp_path / "resolv.conf").stat().st_mode & 0o777 == 0o644
    assert f"{tmp_path}/hosts:/etc/hosts:ro" in command
    assert (tmp_path / "hosts").read_text() == "127.0.0.1 localhost\n::1 localhost\n"
    assert (tmp_path / "hosts").stat().st_mode & 0o777 == 0o644
    assert "--writable-tmpfs" not in command
    assert "tmp,home,cwd,hostfs,bind-paths" in command
    assert "network:probe" in harness.events
    assert len(harness.helpers) == int(internet)
    instance._verify_storage.assert_called_once()
    instance.close()
    instance.overlay.remove.assert_called_once()


@pytest.mark.parametrize("exit_code,contents", [(0, b""), (1, HOST_RESOLVER.encode())])
def test_disk_readiness_rejects_missing_resolver(
    harness, monkeypatch, tmp_path, exit_code, contents
):
    instance = disk_instance(harness, monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance,
        "exec",
        Mock(return_value=subprocess.CompletedProcess([], exit_code, stdout=contents)),
    )
    with pytest.raises(RuntimeError, match="owned /etc/resolv.conf"):
        step_worker.ApptainerInstance._verify_storage(instance)
    instance.close()


def test_disk_readiness_rejects_missing_localhost_file(harness, monkeypatch, tmp_path):
    instance = disk_instance(harness, monkeypatch, tmp_path)
    monkeypatch.setattr(
        instance,
        "exec",
        Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout=HOST_RESOLVER.encode()),
                subprocess.CompletedProcess([], 0, stdout=b""),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="owned /etc/hosts"):
        step_worker.ApptainerInstance._verify_storage(instance)
    instance.close()


def test_disk_is_retained_until_network_helper_cleanup_succeeds(
    harness, monkeypatch, tmp_path
):
    instance = disk_instance(harness, monkeypatch, tmp_path, proxy_url=PROXY_URL)
    helper = harness.helpers[0]
    helper.close_error = TimeoutError("helper remains alive")
    name = instance.name
    with pytest.raises(TimeoutError, match="helper remains alive"):
        instance.close()
    assert instance.name == name
    assert instance._instance_stopped
    instance.overlay.remove.assert_not_called()
    with pytest.raises(RuntimeError, match="cleanup is still pending"):
        instance.exec(["must-not-run"])
    helper.close_error = None
    instance.close()
    assert harness.events.count("instance:stop") == 1
    instance.overlay.remove.assert_called_once()
    assert instance.name is None


def test_disk_stop_failure_retains_network_and_disk(harness, monkeypatch, tmp_path):
    instance = disk_instance(harness, monkeypatch, tmp_path, proxy_url=PROXY_URL)
    harness.failures["instance:stop"] = "busy"
    harness.events.clear()
    with pytest.raises(RuntimeError, match="busy"):
        instance.close()
    assert not instance._instance_stopped
    assert "network:close" not in harness.events
    instance.overlay.remove.assert_not_called()


@pytest.mark.parametrize("flag", ["--network=host", "--net=false", "--dns=127.0.0.1"])
def test_instance_rejects_network_override_before_start(harness, flag):
    with pytest.raises(ValueError):
        make_instance(extra_start_flags=[flag])
    assert harness.calls == []


def test_same_host_network_is_rejected_before_starting_helper(harness):
    harness.namespace = HOST_NAMESPACE
    instance = make_internet_instance()

    with pytest.raises(RuntimeError):
        instance.start()

    assert "network:start" not in harness.events
    assert harness.events.count("instance:stop") == 1
    assert instance.name is None


def test_shared_host_pid_namespace_is_rejected_at_start(harness):
    harness.pid_inode = HOST_PID_INODE  # same PID namespace as the worker
    instance = make_instance()

    with pytest.raises(RuntimeError, match="sharing the worker PID namespace"):
        instance.start()

    assert instance.name is None
    assert instance.pid_ns_fd is None and instance.user_ns_fd is None


def test_shared_host_user_namespace_is_rejected_at_start(harness):
    harness.user_inode = HOST_USER_INODE  # same user namespace as the worker
    instance = make_instance()

    with pytest.raises(RuntimeError, match="sharing the worker user namespace"):
        instance.start()

    assert instance.name is None
    assert instance.pid_ns_fd is None and instance.user_ns_fd is None


def test_exec_outside_the_pinned_pid_namespace_is_rejected_at_start(harness):
    # The verification exec reports a PID namespace other than the pinned one.
    harness.exec_pid_namespace = "pid:[999]"
    instance = make_instance()

    with pytest.raises(RuntimeError, match="not the pinned"):
        instance.start()

    assert instance.name is None
    assert instance.pid_ns_fd is None and instance.user_ns_fd is None


def test_start_pins_the_instance_namespaces(harness):
    instance = make_instance()
    instance.start()
    assert instance.pid_ns_fd is not None and instance.user_ns_fd is not None
    # The pinned PID namespace fd reports the instance's inode, not the worker's.
    assert harness.ns_fstat(instance.pid_ns_fd).st_ino == SANDBOX_PID_INODE
    instance.close()
    assert instance.pid_ns_fd is None and instance.user_ns_fd is None


@pytest.mark.parametrize(
    "override",
    [{"routes": ROUTE_HEADER + FORWARDER_ROUTE}, {"exec_namespace": HOST_NAMESPACE}],
)
def test_unusable_network_is_rejected_and_cleaned(harness, override):
    for key, value in override.items():
        setattr(harness, key, value)
    instance = make_internet_instance()

    with pytest.raises(RuntimeError):
        instance.start()

    assert "network:start" in harness.events
    assert "instance:mkdir" not in harness.events
    assert harness.events.count("instance:stop") == 1
    assert "network:close" in harness.events
    assert instance.name is None


def test_helper_start_failure_stops_instance_and_preserves_error(harness):
    instance = make_internet_instance()
    primary = TimeoutError("network helper never became ready")
    harness.next_start_error = primary

    with pytest.raises(TimeoutError) as caught:
        instance.start()

    assert caught.value is primary
    assert harness.events.count("instance:stop") == 1
    assert "network:probe" not in harness.events
    assert instance.name is None


def test_start_cleanup_failure_retains_identity_and_primary_error(harness):
    instance = make_internet_instance()
    primary = TimeoutError("network helper never became ready")
    harness.next_start_error = primary
    harness.failures["instance:stop"] = "instance stop refused"

    with pytest.raises(TimeoutError) as caught:
        instance.start()

    assert caught.value is primary
    assert instance.name == harness.name
    assert any("instance stop refused" in note for note in caught.value.__notes__)
    harness.failures.clear()
    instance.close()
    assert instance.name is None


def test_bootstrap_failure_cleans_network_and_instance(harness):
    harness.failures["instance:mkdir"] = "working directory is unavailable"
    instance = make_internet_instance()

    with pytest.raises(RuntimeError, match="working directory"):
        instance.start()

    assert harness.events.count("instance:stop") == 1
    assert "network:close" in harness.events
    assert instance.name is None


def test_close_stop_failure_keeps_network_and_instance_for_retry(harness):
    instance = make_internet_instance()
    instance.start()
    harness.events.clear()
    harness.failures["instance:stop"] = "instance stop refused"

    with pytest.raises(RuntimeError, match="instance stop refused"):
        instance.close()

    assert instance.name == harness.name
    assert "network:close" not in harness.events
    harness.failures.clear()
    instance.close()
    assert instance.name is None


def test_close_helper_failure_retains_cleanup_identity(harness):
    instance = make_internet_instance()
    instance.start()
    helper = harness.helpers[0]
    helper.close_error = TimeoutError("network helper will not exit")

    with pytest.raises(TimeoutError, match="network helper will not exit"):
        instance.close()

    assert instance.name == harness.name
    helper.close_error = None
    harness.failures["instance:stop"] = "instance already stopped"
    instance.close()
    assert instance.name is None
    assert harness.events.count("instance:stop") == 1


def test_dead_network_helper_prevents_command_execution(harness):
    instance = make_internet_instance()
    instance.start()
    harness.helpers[0].health_error = RuntimeError("network helper exited unexpectedly")
    harness.calls.clear()

    with pytest.raises(RuntimeError, match="network helper exited unexpectedly"):
        instance.exec(["echo", "must not run"])

    assert harness.calls == []


def test_status_is_read_only_and_preserves_required_fields(harness):
    session = step_worker.WorkerSession()
    session._start({"config": CONFIG})
    harness.calls.clear()
    harness.events.clear()

    result, should_exit = session.dispatch({"operation": "status"})

    assert result["instance_name"] == harness.name
    assert result["cgroup"] == {"path": "/test/cgroup"}
    assert not should_exit
    assert harness.calls == []
    assert "network:start" not in harness.events
    assert "network:close" not in harness.events


def test_restart_replaces_instance_network_inside_same_worker(harness):
    session = step_worker.WorkerSession()
    worker_pid = os.getpid()
    first, _ = session.dispatch({"operation": "start", "config": INTERNET_CONFIG})
    first_helper = harness.helpers[0]
    harness.instance_pid += 1
    harness.namespace = "net:[201]"
    harness.exec_namespace = harness.namespace

    second, _ = session.dispatch({"operation": "restart", "config": INTERNET_CONFIG})

    assert os.getpid() == worker_pid
    assert first["instance_name"] != second["instance_name"]
    assert len(harness.helpers) == 2
    assert first_helper.target != harness.helpers[1].target
    assert "network:close" in harness.events
    assert session.instance.network_metadata["namespace"] == "net:[201]"
    assert session.instance.network_metadata["internet"] is True


def test_restart_keeps_an_offline_lease_offline(harness):
    session = step_worker.WorkerSession()
    session.dispatch({"operation": "start", "config": CONFIG})

    session.dispatch({"operation": "restart", "config": CONFIG})

    assert harness.helpers == []
    assert session.instance.network_metadata["internet"] is False


def test_session_stop_failure_retains_instance_for_release_retry(harness):
    session = step_worker.WorkerSession()
    session._start({"config": CONFIG})
    original = session.instance
    harness.failures["instance:stop"] = "instance stop refused"

    with pytest.raises(RuntimeError, match="instance stop refused"):
        session._stop({})

    assert session.instance is original
    harness.failures.clear()
    session._stop({})
    assert session.instance is None


@pytest.mark.parametrize("mode", ["host", "fakeroot"])
def test_unsupported_network_mode_is_rejected_before_start(harness, mode):
    with pytest.raises(ValueError, match="network_mode"):
        make_instance(network_mode=mode)
    assert harness.calls == []


def test_instance_discovery_failure_stops_unready_instance(harness):
    harness.failures["instance:list"] = "cannot inspect instance PID"
    instance = make_instance()

    with pytest.raises(RuntimeError, match="cannot inspect instance PID"):
        instance.start()

    assert "network:start" not in harness.events
    assert harness.events.count("instance:stop") == 1
    assert instance.name is None


def test_ambient_network_overrides_do_not_reach_start_or_exec(harness, monkeypatch):
    monkeypatch.setenv("APPTAINER_NET", "false")
    monkeypatch.setenv("SINGULARITY_NETWORK", "host")
    monkeypatch.setenv("APPTAINER_DNS", "127.0.0.1")
    monkeypatch.setenv("APPTAINER_SECURITY", "seccomp:/tmp/allow-all.json")
    instance = make_instance()

    instance.start()
    instance.exec(["true"])

    for command, options in harness.calls:
        if command[1:3] == ["instance", "start"] or command[1] == "exec":
            env = options["env"]
            assert "APPTAINER_NET" not in env
            assert "SINGULARITY_NETWORK" not in env
            assert "APPTAINER_DNS" not in env
            assert "APPTAINER_SECURITY" not in env


def test_network_probe_uses_existing_root_before_working_directory_bootstrap(harness):
    instance = make_instance(pwd="/new/task/workspace")

    instance.start()

    probe = next(
        options
        for command, options in harness.calls
        if "readlink /proc/self/ns/net" in " ".join(command)
    )
    assert probe["env"]["APPTAINER_PWD"] == "/"


def test_missing_requested_directory_is_created_before_using_it(harness, monkeypatch):
    requested_directory = "/new/task/workspace"
    directory_exists = False

    def run(command, **kwargs):
        nonlocal directory_exists
        if command[1:3] == ["instance", "start"] or command[1] == "exec":
            cwd = kwargs["env"]["APPTAINER_PWD"]
            if cwd != "/" and not directory_exists:
                return subprocess.CompletedProcess(
                    command, 1, stdout=b"", stderr=b"requested directory does not exist"
                )
        if "mkdir" in command:
            assert command[-1] == requested_directory
            directory_exists = True
        return harness.run(command, **kwargs)

    monkeypatch.setattr(step_worker.subprocess, "run", run)
    instance = make_instance(pwd=requested_directory)

    instance.start()
    instance.exec(["pwd"])

    assert harness.calls[0][1]["env"]["APPTAINER_PWD"] == "/"
    assert directory_exists
    assert harness.calls[-1][1]["env"]["APPTAINER_PWD"] == requested_directory


DEFAULT_ROUTE = "sfnet0\t00000000\t0202000A\t0003\t0\t0\t0\t00000000\t0\t0\t0\n"
SUBNET_ROUTE = "sfnet0\t0000020A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n"


@pytest.mark.parametrize(
    ("internet", "routes"),
    [
        (False, DEFAULT_ROUTE),
        (False, FORWARDER_ROUTE),
        (False, SUBNET_ROUTE),
        (True, DEFAULT_ROUTE),
        (True, FORWARDER_ROUTE),
        (True, SUBNET_ROUTE),
    ],
)
def test_routes_beyond_the_owned_policy_are_rejected(harness, internet, routes):
    harness.routes = ROUTE_HEADER + routes
    instance = make_instance(proxy_url=PROXY_URL if internet else None)

    with pytest.raises(RuntimeError, match="the policy allows none"):
        instance.start()

    assert instance.name is None
    assert "instance:mkdir" not in harness.events


def test_helper_failure_during_command_is_reported_without_replay(harness, monkeypatch):
    instance = make_internet_instance()
    instance.start()
    harness.calls.clear()

    def fail_helper_after(call):
        def wrapper(command, **kwargs):
            result = call(command, **kwargs)
            harness.helpers[0].health_error = RuntimeError(
                "network helper died during command"
            )
            return result

        return wrapper

    # exec() drains its own pipes via Popen, so the helper must be failed on
    # that path; run() is still wrapped for any non-exec call this makes.
    monkeypatch.setattr(step_worker.subprocess, "run", fail_helper_after(harness.run))
    monkeypatch.setattr(
        step_worker.subprocess, "Popen", fail_helper_after(harness.popen)
    )

    with pytest.raises(RuntimeError, match="network helper died during command"):
        instance.exec(["touch", "/task/side-effect"])

    assert len(harness.calls) == 1


@pytest.mark.parametrize("gpu_enabled", [False, True])
def test_session_preserves_storage_and_gpu_configuration(
    harness, monkeypatch, tmp_path, gpu_enabled
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setattr(step_worker, "image_digest", lambda *_: "a" * 64)
    instance = Mock(name="instance")
    factory = Mock(return_value=instance)
    monkeypatch.setattr(step_worker, "ApptainerInstance", factory)
    session = step_worker.WorkerSession(str(tmp_path), 128, gpus=int(gpu_enabled))
    config = CONFIG | {
        "gpu_enabled": gpu_enabled,
        "storage_mb": 64,
        "image_sha256": "a" * 64,
    }
    session._start({"config": config})
    factory.assert_called_once_with(**(config | {"storage_root": str(tmp_path)}))
    assert session._status({})["max_storage_mb"] == 128


@pytest.mark.parametrize("gpu_container_mode", ["nv", "nvccli"])
def test_gpu_disk_start_retains_allocation_and_overlay(
    harness, monkeypatch, tmp_path, gpu_container_mode
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    instance = disk_instance(
        harness,
        monkeypatch,
        tmp_path,
        gpu_enabled=True,
        gpu_container_mode=gpu_container_mode,
    )
    command = next(
        command for command, _ in harness.calls if command[1:3] == ["instance", "start"]
    )
    assert [flag for flag in command if flag in ("--nv", "--nvccli")] == [
        f"--{gpu_container_mode}"
    ]
    assert "--overlay" in command
    assert instance.env()["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "3"
    instance.close()


@pytest.mark.parametrize("proxy_url", [None, PROXY_URL])
def test_proxy_env_is_set_only_for_internet_leases(harness, proxy_url):
    instance = make_instance(proxy_url=proxy_url)
    instance.start()

    env = harness.calls[0][1]["env"]
    names = [f"APPTAINERENV_{name}" for name in step_worker.PROXY_ENV_NAMES]
    if proxy_url is None:
        assert not any(
            key.startswith("APPTAINERENV_") and "proxy" in key.lower() for key in env
        )
    else:
        assert all(env[name] == SANDBOX_PROXY_URL for name in names)
        no_proxy = env["APPTAINERENV_NO_PROXY"].split(",")
        assert {
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0",
        } <= set(no_proxy)
        assert not any(item.startswith("10.0.2.") for item in no_proxy)
        assert socket.gethostname() in no_proxy


def start_command(harness):
    return next(
        command for command, _ in harness.calls if command[1:3] == ["instance", "start"]
    )


def test_default_seccomp_profile_is_applied_verified_and_removed(harness, monkeypatch):
    monkeypatch.delenv("SANDFLEET_SANDBOX_SECCOMP", raising=False)
    instance = make_instance()
    instance.start()
    command = start_command(harness)
    flag = command[command.index("--security") + 1]
    path = flag.removeprefix("seccomp:")
    assert flag.startswith("seccomp:") and os.path.isfile(path)
    assert command.index("--security") < command.index(CONFIG["image"])
    instance.close()
    assert not os.path.exists(path)


def test_operator_seccomp_profile_path_is_used_and_kept(harness, monkeypatch, tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    monkeypatch.setenv("SANDFLEET_SANDBOX_SECCOMP", str(profile))
    instance = make_instance()
    instance.start()
    assert f"seccomp:{profile}" in start_command(harness)
    instance.close()
    assert profile.exists()


def test_missing_operator_seccomp_profile_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDFLEET_SANDBOX_SECCOMP", str(tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="does not exist"):
        make_instance()


def test_seccomp_can_be_disabled(harness, monkeypatch):
    monkeypatch.setenv("SANDFLEET_SANDBOX_SECCOMP", "0")
    make_instance().start()
    assert "--security" not in start_command(harness)


def test_start_refuses_a_sandbox_without_the_seccomp_filter(harness, monkeypatch):
    monkeypatch.delenv("SANDFLEET_SANDBOX_SECCOMP", raising=False)
    harness.seccomp = 0
    instance = make_instance()
    with pytest.raises(RuntimeError, match="seccomp"):
        instance.start()
    assert "instance:stop" in harness.events
    assert instance.name is None


def test_clients_cannot_replace_the_seccomp_profile(monkeypatch):
    monkeypatch.delenv("SANDFLEET_SANDBOX_SECCOMP", raising=False)
    with pytest.raises(ValueError, match="seccomp"):
        make_instance(extra_start_flags=["--security", "seccomp:/tmp/allow.json"])


def test_start_refuses_host_network_filesystems(harness):
    harness.mountinfo = SANDBOX_MOUNTINFO + (
        "9 1 0:50 /exports/example /shared/example rw - nfs4 server:/exports/example rw\n"
    )
    instance = make_instance()
    with pytest.raises(RuntimeError, match=r"/shared/example \(nfs4\)"):
        instance.start()
    assert "instance:stop" in harness.events


def test_network_filesystem_mounts_parses_optional_fields():
    mountinfo = (
        "1 0 0:1 / / rw shared:1 master:2 - fuse.fuse-overlayfs o rw\n"
        "2 1 0:2 / /data rw shared:3 - nfs host:/x rw\n"
        "3 1 0:3 / /scratch rw - lustre x rw\n"
    )
    assert step_worker.network_filesystem_mounts(mountinfo) == [
        "/data (nfs)",
        "/scratch (lustre)",
    ]
    with pytest.raises(RuntimeError, match="Malformed"):
        step_worker.network_filesystem_mounts("garbage\n")


@pytest.mark.parametrize(("value", "expected"), [(None, 4096), ("64", 64), ("0", 0)])
def test_commands_lower_nproc_inside_the_sandbox(harness, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SANDFLEET_SANDBOX_PIDS_LIMIT", raising=False)
    else:
        monkeypatch.setenv("SANDFLEET_SANDBOX_PIDS_LIMIT", value)
    instance = make_instance()
    instance.start()
    instance.run_command("echo hi", 10)
    command = harness.calls[-1][0]
    instance_index = command.index(f"instance://{instance.name}")
    script = command[instance_index + 3]
    assert command[instance_index + 1 : instance_index + 3] == ["bash", "-c"]
    if expected:
        assert script.startswith(f"ulimit -u {expected} && timeout ")
    else:
        assert "ulimit" not in script


@pytest.mark.parametrize("value", ["-1", "many"])
def test_invalid_pids_limit_fails(monkeypatch, value):
    monkeypatch.setenv("SANDFLEET_SANDBOX_PIDS_LIMIT", value)
    with pytest.raises(ValueError, match="PIDS_LIMIT"):
        make_instance()


@pytest.mark.parametrize(
    ("fstype", "refused"),
    [
        ("fuse.fuse-overlayfs", False),
        ("fuse.squashfuse", False),
        ("fuse.squashfuse_ll", False),
        ("fuse.ext4", False),
        ("fuse.fuse2fs", True),
        ("fuse.sshfs", True),
        ("fuse.s3fs", True),
        ("fuse.gcsfuse", True),
        ("fuse.glusterfs", True),
        ("fuse", True),
        ("fuseblk", True),
    ],
)
def test_unrecognized_fuse_mounts_are_refused(fstype, refused):
    mountinfo = f"7 1 0:60 / /data rw - {fstype} source rw\n"
    expected = [f"/data ({fstype})"] if refused else []
    assert step_worker.network_filesystem_mounts(mountinfo) == expected


def test_default_seccomp_profile_fails_on_unsupported_hosts(monkeypatch):
    monkeypatch.delenv("SANDFLEET_SANDBOX_SECCOMP", raising=False)
    monkeypatch.setattr("sandfleet.seccomp.platform.machine", lambda: "s390x")
    with pytest.raises(ValueError, match="s390x"):
        make_instance()
    monkeypatch.setenv("SANDFLEET_SANDBOX_SECCOMP", "0")
    make_instance()


def test_each_start_of_an_instance_gets_a_fresh_relay(harness):
    instance = make_internet_instance()
    instance.start()
    first = instance.network
    instance.start()
    assert len(harness.helpers) == 2
    assert instance.network is harness.helpers[1] is not first
    assert harness.events.count("network:close") == 1
    instance.close()
    assert instance.network is None


def test_relay_pending_cap_comes_from_the_start_config(harness):
    instance = make_internet_instance(relay_max_pending=4)
    instance.start()
    assert harness.helpers[0].max_pending == 4
    with pytest.raises(ValueError, match="relay_max_pending"):
        make_internet_instance(relay_max_pending=0)
