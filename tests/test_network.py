from __future__ import annotations

import errno
import os
import select
import socket
import subprocess
import threading
import time
from unittest.mock import Mock

import pytest

from sandfleet import network
from sandfleet.network import (
    SANDBOX_PROXY_PORT,
    LoopbackProxy,
    forwarder_endpoint,
    sandbox_listener,
    sandbox_proxy_url,
    sanitized_apptainer_env,
    validate_network_config,
    validate_network_flags,
)


@pytest.mark.parametrize(
    "argument",
    [
        "--net",
        "--net=false",
        "-n",
        "-n=false",
        "-Cen",
        "--network",
        "--network=host",
        "--network-args=portmap=8080:80/tcp",
        "--netns-path=/proc/1/ns/net",
        "--sharens",
        "--dns=127.0.0.1",
        "--fakeroot=false",
        "-f=false",
        "--userns",
        "-u",
        "--ignore-userns",
        "--ignore-subuid",
        "--",
    ],
)
def test_reject_network_override(argument):
    with pytest.raises(ValueError, match="Sandfleet manages network isolation and DNS"):
        validate_network_flags([argument])


def test_allow_unrelated_flags_and_attached_short_option_values():
    validate_network_flags(
        ["--bind", "/tmp/network:/data", "-B/tmp/network:/data", "--env", "THING=value"]
    )


def test_validate_operator_network_config():
    validate_network_config("loopback")


@pytest.mark.parametrize("mode", ["host", "none", "", "bridge"])
def test_reject_network_modes_without_managed_isolation(mode):
    with pytest.raises(ValueError, match="network_mode"):
        validate_network_config(mode)


def test_removed_fakeroot_mode_is_rejected():
    with pytest.raises(ValueError, match="network_mode"):
        validate_network_config("fakeroot")


def test_removed_slirp4netns_mode_is_rejected_with_its_replacement():
    with pytest.raises(ValueError, match="slirp4netns' was removed.*'loopback'"):
        validate_network_config("slirp4netns")


def test_forwarder_endpoint_comes_from_the_lease_proxy_url():
    assert forwarder_endpoint("http://lease:secret@127.0.0.1:4567") == (
        "127.0.0.1",
        4567,
    )


@pytest.mark.parametrize(
    "proxy_url", ["https://lease:secret@127.0.0.1:1", "http://lease:secret@host", ""]
)
def test_forwarder_endpoint_must_be_http_with_a_port(proxy_url):
    with pytest.raises(ValueError, match="forwarder URL"):
        forwarder_endpoint(proxy_url)


def test_sandbox_proxy_url_keeps_credentials_and_uses_the_local_listener():
    assert (
        sandbox_proxy_url("http://lease:s%2Fecret@127.0.0.1:4567")
        == f"http://lease:s%2Fecret@127.0.0.1:{SANDBOX_PROXY_PORT}"
    )


OWN_NAMESPACE = f"net:[{os.stat('/proc/self/ns/net').st_ino}]"


def test_listener_joins_the_captured_namespaces_and_is_close_on_exec(monkeypatch):
    calls = []
    monkeypatch.setattr("sandfleet.network.shutil.which", lambda name: name)
    real_run = subprocess.run

    def run(command, **kwargs):
        calls.append((command, kwargs))
        # Run the real helper here instead of inside nsenter's namespaces.
        helper = command[command.index("--") + 1 :]
        return real_run(helper[:-1] + ["0"], **kwargs)

    monkeypatch.setattr("sandfleet.network.subprocess.run", run)
    monkeypatch.setattr("sandfleet.network._verify_listener", lambda *_: None)
    listener = sandbox_listener(os.getpid(), OWN_NAMESPACE)
    try:
        command, options = calls[0]
        user_fd, net_fd = options["pass_fds"][1:]
        assert command[:6] == [
            "nsenter",
            f"--user=/proc/self/fd/{user_fd}",
            f"--net=/proc/self/fd/{net_fd}",
            "--preserve-credentials",
            "--",
            network.sys.executable,
        ]
        assert command[-1] == str(SANDBOX_PROXY_PORT)
        assert not listener.get_inheritable()
        assert os.get_blocking(listener.fileno())
        host, port = listener.getsockname()
        assert host == "127.0.0.1" and port > 0
        with socket.create_connection((host, port), timeout=5):
            pass
    finally:
        listener.close()
    for descriptor in (user_fd, net_fd):
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_listener_refuses_a_pid_whose_namespace_changed(monkeypatch):
    monkeypatch.setattr("sandfleet.network.shutil.which", lambda name: name)
    run = Mock()
    monkeypatch.setattr("sandfleet.network.subprocess.run", run)
    with pytest.raises(RuntimeError, match="no longer has network namespace"):
        sandbox_listener(os.getpid(), "net:[1]")
    run.assert_not_called()


@pytest.mark.parametrize(
    "namespace", ["", "net:", "net:[x]", "4026531840", "user:[4026531837]"]
)
def test_listener_rejects_malformed_namespace(namespace):
    with pytest.raises(ValueError, match="Malformed net namespace"):
        sandbox_listener(os.getpid(), namespace)


def test_listener_failure_is_visible(monkeypatch):
    monkeypatch.setattr("sandfleet.network.shutil.which", lambda name: name)
    monkeypatch.setattr(
        "sandfleet.network.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, b"", b"OSError: [Errno 98] Address already in use"
        ),
    )
    with pytest.raises(RuntimeError, match="exit=1.*Address already in use"):
        sandbox_listener(os.getpid(), OWN_NAMESPACE)


def test_listener_requires_nsenter(monkeypatch):
    monkeypatch.setattr("sandfleet.network.shutil.which", lambda name: None)
    monkeypatch.setattr("sandfleet.network.subprocess.run", Mock())
    with pytest.raises(RuntimeError, match="'nsenter' is unavailable"):
        sandbox_listener(os.getpid(), OWN_NAMESPACE)


@pytest.mark.parametrize("pid", [0, -1, True, "1"])
def test_listener_rejects_invalid_instance_pid(pid):
    with pytest.raises(ValueError, match="positive integer"):
        sandbox_listener(pid, OWN_NAMESPACE)


def test_listener_in_the_expected_namespace_is_accepted(monkeypatch):
    listener = socket.create_server(("127.0.0.1", 0))
    own = os.open("/proc/self/ns/net", os.O_RDONLY)
    monkeypatch.setattr("sandfleet.network.fcntl.ioctl", lambda *_: os.dup(own))
    inode = network.namespace_inode(OWN_NAMESPACE)
    try:
        port = listener.getsockname()[1]
        network._verify_listener(listener, inode, port)
        with pytest.raises(RuntimeError, match="bound to"):
            network._verify_listener(listener, inode, port + 1)
        with pytest.raises(RuntimeError, match="not in the sandbox network namespace"):
            network._verify_listener(listener, inode + 1, port)
    finally:
        os.close(own)
        listener.close()


@pytest.fixture
def upstream():
    """An echo server standing in for the egress forwarder."""
    server = socket.create_server(("127.0.0.1", 0))
    connections = []

    def serve():
        while True:
            try:
                connection, _ = server.accept()
            except OSError:
                return
            connections.append(connection)

            def echo(item=connection):
                with item:
                    while block := item.recv(65536):
                        item.sendall(b"echo:" + block)

            threading.Thread(target=echo, daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname()
    server.close()


def local_proxy(endpoint, **options):
    proxy = LoopbackProxy(
        endpoint,
        listener_factory=lambda _pid, _namespace: socket.create_server(
            ("127.0.0.1", 0)
        ),
        **options,
    )
    proxy.start(4321, OWN_NAMESPACE)
    return proxy


def test_relay_carries_both_directions_and_closes_cleanly(upstream):
    proxy = local_proxy(upstream)
    try:
        address = proxy.listener.getsockname()
        for _ in range(3):
            with socket.create_connection(address, timeout=5) as client:
                client.sendall(b"CONNECT pypi.org:443 HTTP/1.1\r\n\r\n")
                assert client.recv(100).startswith(b"echo:CONNECT")
        proxy.ensure_alive()
        assert proxy.pid is None
    finally:
        proxy.close()
    assert proxy.listener is None
    with pytest.raises(RuntimeError, match="not been started"):
        proxy.ensure_alive()


def test_close_interrupts_open_relays(upstream):
    proxy = local_proxy(upstream)
    client = socket.create_connection(proxy.listener.getsockname(), timeout=5)
    client.sendall(b"x")
    assert client.recv(10) == b"echo:x"
    proxy.close()
    assert client.recv(10) == b""
    client.close()


def test_unreachable_forwarder_closes_the_sandbox_connection():
    unused = socket.create_server(("127.0.0.1", 0))
    endpoint = unused.getsockname()
    unused.close()
    proxy = local_proxy(endpoint)
    try:
        with socket.create_connection(
            proxy.listener.getsockname(), timeout=5
        ) as client:
            assert client.recv(10) == b""
        proxy.ensure_alive()
    finally:
        proxy.close()


def test_dead_accept_loop_is_reported(upstream):
    proxy = local_proxy(upstream)
    proxy.listener.shutdown(socket.SHUT_RDWR)
    proxy._accept_thread.join(timeout=5)
    with pytest.raises(RuntimeError, match="relay stopped"):
        proxy.ensure_alive()
    proxy.close()


def test_connection_cap_refuses_extra_sandbox_connections(upstream):
    proxy = local_proxy(upstream, max_connections=1)
    try:
        address = proxy.listener.getsockname()
        first = socket.create_connection(address, timeout=5)
        first.sendall(b"a")
        assert first.recv(10) == b"echo:a"
        with socket.create_connection(address, timeout=5) as second:
            assert second.recv(10) == b""
        first.close()
    finally:
        proxy.close()


def test_double_start_keeps_the_owned_listener(upstream):
    proxy = local_proxy(upstream)
    try:
        listener = proxy.listener
        with pytest.raises(RuntimeError, match="already been started"):
            proxy.start(4321, OWN_NAMESPACE)
        assert proxy.listener is listener
    finally:
        proxy.close()


@pytest.mark.parametrize("prefix", ["APPTAINER_", "SINGULARITY_"])
def test_remove_inherited_namespace_overrides_and_preserve_other_environment(prefix):
    original = {
        "PATH": "/usr/bin:/bin",
        "HTTPS_PROXY": "http://proxy.example:3128",
        prefix + "CACHEDIR": "/scratch/cache",
        prefix + "NET": "false",
        prefix + "UNSHARE_NET": "false",
        prefix + "NETWORK": "none",
        prefix + "NETWORK_ARGS": "portmap=8080:80/tcp",
        prefix + "NETNS_PATH": "/proc/1/ns/net",
        prefix + "SHARENS": "true",
        prefix + "DNS": "127.0.0.1",
        prefix + "FAKEROOT": "false",
        prefix + "USERNS": "true",
        prefix + "UNSHARE_USERNS": "true",
        prefix + "IGNORE_USERNS": "true",
        prefix + "IGNORE_SUBUID": "true",
        prefix + "SECURITY": "seccomp:/tmp/allow.json",
    }
    sanitized = sanitized_apptainer_env(original)
    assert sanitized == {
        "PATH": "/usr/bin:/bin",
        "HTTPS_PROXY": "http://proxy.example:3128",
        prefix + "CACHEDIR": "/scratch/cache",
    }
    assert prefix + "NETWORK" in original


def test_closed_proxy_cannot_be_restarted(upstream):
    proxy = local_proxy(upstream)
    proxy.close()
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        proxy.start(4321, OWN_NAMESPACE)
    assert proxy.listener is None


class ScriptedListener:
    """accept() raises the scripted errors in order, then blocks until shutdown."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.accepts = 0
        self.stopped = threading.Event()

    def accept(self):
        self.accepts += 1
        if self.errors:
            raise self.errors.pop(0)
        self.stopped.wait()
        raise OSError(errno.EINVAL, "shut down")

    def shutdown(self, _how):
        self.stopped.set()

    def close(self):
        pass


def scripted_proxy(listener):
    proxy = LoopbackProxy(("127.0.0.1", 9), listener_factory=lambda *_: listener)
    proxy.start(4321, OWN_NAMESPACE)
    return proxy


def test_transient_accept_errors_keep_the_relay_serving(monkeypatch):
    monkeypatch.setattr(network, "ACCEPT_RETRY_SECONDS", 0)
    transient = [
        OSError(code, "transient")
        for code in (errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ECONNABORTED)
    ]
    listener = ScriptedListener(transient)
    proxy = scripted_proxy(listener)
    deadline = time.monotonic() + 5
    while listener.accepts < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    proxy.ensure_alive()
    assert proxy.error is None
    proxy.close()


@pytest.mark.parametrize(
    "error", [OSError(errno.EBADF, "bad descriptor"), ValueError("unexpected")]
)
def test_fatal_accept_error_is_recorded(error):
    proxy = scripted_proxy(ScriptedListener([error]))
    proxy._accept_thread.join(timeout=5)
    with pytest.raises(RuntimeError, match=f"relay stopped: .*{error.args[-1]}"):
        proxy.ensure_alive()
    proxy.close()


def test_failed_relay_thread_start_drops_only_that_connection(upstream, monkeypatch):
    proxy = local_proxy(upstream)
    real_thread = network.threading.Thread
    failures = []

    class FailingOnce(real_thread):
        def start(self):
            if not failures:
                failures.append(self)
                raise RuntimeError("can't start new thread")
            super().start()

    monkeypatch.setattr(network.threading, "Thread", FailingOnce)
    try:
        address = proxy.listener.getsockname()
        with socket.create_connection(address, timeout=5) as dropped:
            assert dropped.recv(10) == b""
        assert proxy._pending == 0
        assert all(pump.ident is not None for pump in proxy._pumps)
        with socket.create_connection(address, timeout=5) as client:
            client.sendall(b"y")
            assert client.recv(10) == b"echo:y"
        proxy.ensure_alive()
        assert proxy._pending == 0
    finally:
        proxy.close()


@pytest.fixture
def silent_upstream():
    """A forwarder stand-in that answers only when told to."""
    server = socket.create_server(("127.0.0.1", 0))
    connections = []

    def serve():
        while True:
            try:
                connections.append(server.accept()[0])
            except OSError:
                return

    threading.Thread(target=serve, daemon=True).start()
    yield server.getsockname(), connections
    server.close()
    for connection in connections:
        connection.close()


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


def test_pending_cap_stops_accepting_until_the_forwarder_answers(silent_upstream):
    endpoint, upstream_connections = silent_upstream
    proxy = local_proxy(endpoint, max_pending=2)
    clients = []
    try:
        address = proxy.listener.getsockname()
        clients = [socket.create_connection(address, timeout=5) for _ in range(3)]
        wait_for(lambda: len(upstream_connections) == 2)
        time.sleep(0.2)
        # The third connection waits in the sandbox's backlog.
        assert len(upstream_connections) == 2 and proxy._pending == 2
        upstream_connections[0].sendall(
            b"HTTP/1.1 407 Proxy Authentication Required\r\n"
        )
        wait_for(lambda: len(upstream_connections) == 3)
        # Relays may connect upstream in either order: one of the two first
        # clients got the answer.
        readable, _, _ = select.select(clients[:2], [], [], 5)
        assert len(readable) == 1
        assert readable[0].recv(100).startswith(b"HTTP/1.1 407")
        assert proxy._pending == 2
    finally:
        for client in clients:
            client.close()
        proxy.close()


def test_close_right_after_a_failed_relay_thread_start(upstream, monkeypatch):
    proxy = local_proxy(upstream)
    real_thread = network.threading.Thread

    class NeverStarts(real_thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(network.threading, "Thread", NeverStarts)
    with socket.create_connection(proxy.listener.getsockname(), timeout=5) as dropped:
        assert dropped.recv(10) == b""
    proxy.close()
    assert proxy.listener is None


def test_end_of_request_head_ends_pending_across_chunks(silent_upstream):
    endpoint, upstream_connections = silent_upstream
    proxy = local_proxy(endpoint, max_pending=1)
    clients = []
    try:
        address = proxy.listener.getsockname()
        clients.append(socket.create_connection(address, timeout=5))
        wait_for(lambda: len(upstream_connections) == 1)
        clients[0].sendall(b"CONNECT pypi.org:443 HTTP/1.1\r\nProxy-Authorization: x\r")
        clients.append(socket.create_connection(address, timeout=5))
        time.sleep(0.2)
        assert len(upstream_connections) == 1 and proxy._pending == 1
        clients[0].sendall(b"\n\r\n")
        wait_for(lambda: len(upstream_connections) == 2)
    finally:
        for client in clients:
            client.close()
        proxy.close()


def test_silent_forwarder_releases_pending_at_the_deadline(
    silent_upstream, monkeypatch
):
    monkeypatch.setattr(network, "RELAY_ANSWER_TIMEOUT_SECONDS", 1.0)
    endpoint, upstream_connections = silent_upstream
    proxy = local_proxy(endpoint, max_pending=1)
    clients = []
    try:
        address = proxy.listener.getsockname()
        clients = [socket.create_connection(address, timeout=5) for _ in range(2)]
        wait_for(lambda: len(upstream_connections) == 1)
        assert proxy._pending == 1
        wait_for(lambda: len(upstream_connections) == 2)
        proxy.ensure_alive()
    finally:
        for client in clients:
            client.close()
        proxy.close()


@pytest.mark.parametrize(
    ("slots", "limit"), [(1, 32), (2, 16), (16, 2), (32, 1), (64, 1)]
)
def test_relay_pending_limit_shares_the_forwarder_slots(slots, limit):
    assert network.relay_pending_limit(slots) == limit
    assert network.relay_pending_limit(slots) * min(slots, 32) <= 32


def test_relay_pending_limit_requires_slots():
    with pytest.raises(ValueError, match="slots"):
        network.relay_pending_limit(0)


def test_concurrent_release_paths_free_a_pending_slot_once(upstream):
    proxy = local_proxy(upstream)
    try:
        for _ in range(200):
            with proxy._capacity:
                proxy._pending += 1
            release = proxy._pending_releaser()
            barrier = threading.Barrier(2)

            def race(release=release, barrier=barrier):
                barrier.wait()
                release()

            threads = [threading.Thread(target=race) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            release()
            assert proxy._pending == 0
    finally:
        proxy.close()
