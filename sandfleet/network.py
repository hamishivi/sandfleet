"""Rootless networking for one private Apptainer network namespace.

Every sandbox starts with `--network=none`: loopback only, no routes. An
internet lease additionally gets a listening socket on 127.0.0.1 inside its
namespace, created from the host side, whose connections the worker relays to
the egress forwarder on the host's loopback.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import os
import select
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping

from sandfleet.egress import AUTH_TIMEOUT_SECONDS, MAX_PENDING_CONNECTIONS

NETWORK_MODES = ("loopback",)
# The sandbox-side proxy listener. A fresh namespace has nothing bound yet; the
# listener is created right after the instance starts, before any command.
SANDBOX_PROXY_PORT = 3128
MAX_RELAY_CONNECTIONS = 64
RELAY_CONNECT_TIMEOUT_SECONDS = 10.0
# A relayed connection is pending, and counts against its sandbox's share of
# the forwarder's unauthenticated slots, until the first of: the forwarder's
# first byte, the end of the sandbox's request head reaching the forwarder, or
# this long after connecting (the forwarder's own authentication timeout plus
# slack).
RELAY_ANSWER_TIMEOUT_SECONDS = AUTH_TIMEOUT_SECONDS + 2.0
TRANSIENT_ACCEPT_ERRORS = frozenset(
    {errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM, errno.ECONNABORTED}
)
ACCEPT_RETRY_SECONDS = 0.1
SIOCGSKNS = 0x894C  # ioctl: open the network namespace a socket belongs to
# Runs under nsenter in the instance's user and network namespaces: bind the
# listener there and hand its descriptor back over the inherited socket.
_LISTENER_HELPER = """
import socket, sys
channel = socket.socket(fileno=int(sys.argv[1]))
listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
listener.bind(("127.0.0.1", int(sys.argv[2])))
listener.listen(128)
socket.send_fds(channel, [b"L"], [listener.fileno()])
"""

_MANAGED_ENV_KEYS = frozenset(
    {
        "NET",
        "UNSHARE_NET",
        "NETWORK",
        "NETWORK_ARGS",
        "NETNS_PATH",
        "SHARENS",
        "DNS",
        "FAKEROOT",
        "USERNS",
        "UNSHARE_USERNS",
        "IGNORE_USERNS",
        "IGNORE_SUBUID",
        # The seccomp profile (--security) is Sandfleet's too.
        "SECURITY",
    }
)
_MANAGED_FLAGS = frozenset(
    {
        "--net",
        "--network",
        "--network-args",
        "--netns-path",
        "--sharens",
        "--dns",
        "--fakeroot",
        "--userns",
        "--ignore-userns",
        "--ignore-subuid",
    }
)
# Stop examining a short-option group once its remainder is an option value.
_SHORT_VALUE_FLAGS = frozenset({"B", "H", "S", "W", "s"})


def validate_network_config(mode: str) -> None:
    """Validate operator-owned network settings before any allocation starts."""
    if mode == "slirp4netns":
        raise ValueError(
            "network_mode 'slirp4netns' was removed in 0.11.0: sandboxes no longer "
            "use slirp4netns; use 'loopback'"
        )
    if mode not in NETWORK_MODES:
        raise ValueError(f"network_mode must be one of {NETWORK_MODES!r}, got {mode!r}")


def network_mode_argument(value: str) -> str:
    """argparse type for --network-mode that explains removed modes."""
    try:
        validate_network_config(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def forwarder_endpoint(proxy_url: str) -> tuple[str, int]:
    """Return the host-side (address, port) of the forwarder in a lease proxy URL."""
    parts = urllib.parse.urlsplit(proxy_url)
    if parts.scheme != "http" or not parts.hostname or parts.port is None:
        raise ValueError(f"Invalid egress forwarder URL for lease: {parts.hostname!r}")
    return parts.hostname, parts.port


def sandbox_proxy_url(proxy_url: str) -> str:
    """The lease's proxy URL as seen inside the sandbox: same credentials, local port."""
    parts = urllib.parse.urlsplit(proxy_url)
    userinfo = parts.netloc.rpartition("@")[0]
    netloc = f"{userinfo}@127.0.0.1:{SANDBOX_PROXY_PORT}"
    return urllib.parse.urlunsplit(parts._replace(netloc=netloc))


def _managed_short_option(argument: str) -> bool:
    if not argument.startswith("-") or argument.startswith("--"):
        return False
    for character in argument[1:].split("=", 1)[0]:
        if character in "nfu":
            return True
        if character in _SHORT_VALUE_FLAGS:
            return False
    return False


def validate_network_flags(flags: list[str]) -> None:
    """Reject overrides of the namespace, forwarding, and DNS policy."""
    for argument in flags:
        if (
            argument == "--"
            or argument.split("=", 1)[0] in _MANAGED_FLAGS
            or _managed_short_option(argument)
        ):
            raise ValueError(
                f"Sandfleet manages network isolation and DNS; "
                f"extra_start_flags cannot contain {argument!r}"
            )


def sanitized_apptainer_env(env: Mapping[str, str]) -> dict[str, str]:
    """Prevent host Apptainer settings from overriding Sandfleet networking or seccomp."""
    blocked = {
        prefix + key
        for prefix in ("APPTAINER_", "SINGULARITY_")
        for key in _MANAGED_ENV_KEYS
    }
    return {key: value for key, value in env.items() if key not in blocked}


def relay_pending_limit(slots: int) -> int:
    """Per-sandbox pending cap: the worker's sandboxes share the forwarder's slots."""
    if slots < 1:
        raise ValueError("slots must be positive")
    return max(1, MAX_PENDING_CONNECTIONS // slots)


def namespace_inode(namespace: str, kind: str = "net") -> int:
    """Return the inode of a namespace link value such as 'net:[4026531840]'."""
    prefix, separator, rest = namespace.partition(":[")
    if (
        prefix != kind
        or not separator
        or not rest.endswith("]")
        or not rest[:-1].isdigit()
    ):
        raise ValueError(f"Malformed {kind} namespace identity {namespace!r}")
    return int(rest[:-1])


def sandbox_listener(
    instance_pid: int, namespace: str, port: int = SANDBOX_PROXY_PORT
) -> socket.socket:
    """Bind 127.0.0.1:<port> inside an instance's network namespace.

    `namespace` is the instance's network namespace as already verified by the
    caller ('net:[inode]'). The worker opens the instance's user and network
    namespaces once, refuses them if the PID now names another namespace, and
    nsenter joins exactly those descriptors: the worker owns the user namespace,
    so no sandbox process takes part and the sandbox keeps no route anywhere.
    The returned socket's namespace is verified against `namespace` too.
    """
    if type(instance_pid) is not int or instance_pid <= 0:
        raise ValueError("Apptainer instance PID must be a positive integer")
    expected = namespace_inode(namespace)
    if shutil.which("nsenter") is None:
        raise RuntimeError(
            "'nsenter' is unavailable; Sandfleet needs util-linux nsenter on "
            "workers to give internet sandboxes their proxy listener"
        )
    with contextlib.ExitStack() as stack:
        descriptors = {}
        for kind in ("user", "net"):
            descriptor = os.open(
                f"/proc/{instance_pid}/ns/{kind}", os.O_RDONLY | os.O_CLOEXEC
            )
            stack.callback(os.close, descriptor)
            descriptors[kind] = descriptor
        if os.fstat(descriptors["net"]).st_ino != expected:
            raise RuntimeError(
                f"Instance PID {instance_pid} no longer has network namespace {namespace}"
            )
        listener = _bind_in_namespace(descriptors["user"], descriptors["net"], port)
    try:
        _verify_listener(listener, expected, port)
    except BaseException:
        listener.close()
        raise
    return listener


def _bind_in_namespace(user_fd: int, net_fd: int, port: int) -> socket.socket:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with parent:
        with child:
            result = subprocess.run(
                [
                    "nsenter",
                    f"--user=/proc/self/fd/{user_fd}",
                    f"--net=/proc/self/fd/{net_fd}",
                    # Keep the owner's IDs (root in the fakeroot userns): no
                    # setgroups, which a root-only mapping without subuids denies.
                    "--preserve-credentials",
                    "--",
                    sys.executable,
                    "-c",
                    _LISTENER_HELPER,
                    str(child.fileno()),
                    str(port),
                ],
                pass_fds=(child.fileno(), user_fd, net_fd),
                capture_output=True,
                timeout=30,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not create the sandbox proxy listener (exit={result.returncode}): "
                f"{result.stderr.decode('utf-8', 'replace').strip()}"
            )
        parent.settimeout(5)
        # Receive close-on-exec, so no later exec inherits the listener.
        # (socket.recv_fds does not pass its flags to recvmsg on Python 3.12.)
        _, ancillary, _, _ = parent.recvmsg(
            1, socket.CMSG_SPACE(4), socket.MSG_CMSG_CLOEXEC
        )
    received = [
        int.from_bytes(data[:4], sys.byteorder, signed=True)
        for level, kind, data in ancillary
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS
    ]
    if len(received) != 1:
        raise RuntimeError("Sandbox proxy listener helper returned no descriptor")
    return socket.socket(fileno=received[0])


def _verify_listener(listener: socket.socket, expected_inode: int, port: int) -> None:
    if listener.getsockname() != ("127.0.0.1", port):
        raise RuntimeError(
            f"Sandbox proxy listener is bound to {listener.getsockname()}"
        )
    # The worker owns the sandbox's user namespace, so it may open the socket's
    # network namespace (this needs CAP_NET_ADMIN over it).
    socket_namespace = fcntl.ioctl(listener.fileno(), SIOCGSKNS)
    try:
        actual = os.fstat(socket_namespace).st_ino
    finally:
        os.close(socket_namespace)
    if actual != expected_inode:
        raise RuntimeError(
            "Sandbox proxy listener is not in the sandbox network namespace"
        )


class LoopbackProxy:
    """Relay an internet sandbox's 127.0.0.1 proxy port to the egress forwarder.

    The listener lives in the sandbox namespace; upstream connections are made
    from the worker's own namespace, so the sandbox needs no interface or route.
    The forwarder still checks each lease's credentials. One proxy serves one
    instance: once closed it cannot be started again.

    A relay is pending from accept until the forwarder's first byte, the end
    of the sandbox's request head reaching the forwarder, or
    RELAY_ANSWER_TIMEOUT_SECONDS after connecting, whichever is first. While
    `max_pending` relays are pending the proxy stops accepting, so the
    sandbox's backlog absorbs the excess and one sandbox cannot hold more than
    its share (relay_pending_limit) of the forwarder's unauthenticated slots.
    """

    def __init__(
        self,
        upstream: tuple[str, int],
        *,
        listener_factory=sandbox_listener,
        max_connections: int = MAX_RELAY_CONNECTIONS,
        max_pending: int = 1,
    ) -> None:
        self.upstream = upstream
        self.listener_factory = listener_factory
        self.max_connections = max_connections
        self.max_pending = max_pending
        self.listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._connections: set[socket.socket] = set()
        self._pumps: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._capacity = threading.Condition(self._lock)
        self._pending = 0
        self._closing = False
        self.error: BaseException | None = None

    @property
    def pid(self) -> None:
        """Relays run as threads of the step worker; there is no helper process."""
        return None

    def start(self, instance_pid: int, namespace: str) -> None:
        if self._closing:
            raise RuntimeError("A closed sandbox proxy relay cannot be restarted")
        if self.listener is not None:
            raise RuntimeError("Sandbox proxy relay has already been started")
        self.listener = self.listener_factory(instance_pid, namespace)
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="sandfleet-proxy-relay", daemon=True
        )
        self._accept_thread.start()

    def ensure_alive(self) -> None:
        if self.listener is None or self._accept_thread is None:
            raise RuntimeError("Sandbox proxy relay has not been started")
        if not self._accept_thread.is_alive():
            raise RuntimeError(f"Sandbox proxy relay stopped: {self.error!r}")

    def _accept_loop(self) -> None:
        try:
            while self._accept_one():
                pass
        except Exception as error:
            with self._lock:
                if not self._closing:
                    self.error = error

    def _accept_one(self) -> bool:
        """Accept and hand off one connection; False once the proxy is closing."""
        assert self.listener is not None
        with self._capacity:
            while self._pending >= self.max_pending and not self._closing:
                self._capacity.wait()
            if self._closing:
                return False
        try:
            client, _ = self.listener.accept()
        except OSError as error:
            if error.errno not in TRANSIENT_ACCEPT_ERRORS:
                raise
            # Descriptor or buffer exhaustion, or a client that gave up: wait
            # and keep serving rather than cutting the lease off.
            time.sleep(ACCEPT_RETRY_SECONDS)
            return True
        with self._lock:
            if self._closing:
                client.close()
                return False
            if len(self._connections) >= 2 * self.max_connections:
                client.close()
                return True
            self._connections.add(client)
            self._pending += 1
        answered = self._pending_releaser()
        thread = threading.Thread(
            target=self._relay, args=(client, answered), daemon=True
        )
        try:
            thread.start()
        except RuntimeError:
            # No thread for this connection: drop it, keep serving the rest.
            answered()
            self._discard(client)
            return True
        # Only started threads are tracked: close() joins every tracked thread.
        with self._lock:
            self._pumps = [*(pump for pump in self._pumps if pump.is_alive()), thread]
        return True

    def _pending_releaser(self) -> Callable[[], None]:
        """Return a callable that frees one connection's pending slot exactly once.

        The request and answer threads both call it; the flag is tested and set
        under the same lock as the count, so a slot cannot be freed twice.
        """
        released = False

        def release() -> None:
            nonlocal released
            with self._capacity:
                if released:
                    return
                released = True
                self._pending -= 1
                self._capacity.notify_all()

        return release

    def _relay(self, client: socket.socket, answered: Callable[[], None]) -> None:
        try:
            upstream = socket.create_connection(
                self.upstream, timeout=RELAY_CONNECT_TIMEOUT_SECONDS
            )
        except OSError:
            answered()
            self._discard(client)
            return
        upstream.settimeout(None)
        deadline = time.monotonic() + RELAY_ANSWER_TIMEOUT_SECONDS
        with self._lock:
            closing = self._closing
            if not closing:
                self._connections.add(upstream)
        if closing:
            answered()
            self._discard(client, upstream)
            return
        reverse = threading.Thread(
            target=self._forward_answer,
            args=(upstream, client, answered, deadline),
            daemon=True,
        )
        try:
            reverse.start()
        except RuntimeError:
            answered()
            self._discard(client, upstream)
            return
        self._forward_request(client, upstream, answered)
        reverse.join()
        answered()
        self._discard(client, upstream)

    @classmethod
    def _forward_request(cls, client, upstream, answered) -> None:
        """Copy sandbox to forwarder; the end of the request head ends pending."""
        tail = b""

        def on_block(block: bytes) -> None:
            nonlocal tail
            if tail is None:
                return
            if b"\r\n\r\n" in tail + block:
                tail = None
                answered()
            else:
                tail = (tail + block)[-3:]

        cls._copy(client, upstream, on_block)

    @classmethod
    def _forward_answer(cls, upstream, client, answered, deadline: float) -> None:
        """Copy forwarder to sandbox; its first byte, or the deadline, ends pending."""
        with contextlib.suppress(OSError, ValueError):
            # Bounded: a forwarder that never answers cannot pin a pending slot.
            # poll(), unlike select(), has no descriptor-number limit.
            poller = select.poll()
            poller.register(upstream, select.POLLIN)
            poller.poll(max(0, int((deadline - time.monotonic()) * 1000)))
        answered()
        cls._copy(upstream, client, None)

    @staticmethod
    def _copy(source: socket.socket, destination: socket.socket, on_block) -> None:
        with contextlib.suppress(OSError):
            while True:
                block = source.recv(65536)
                if not block:
                    break
                destination.sendall(block)
                if on_block is not None:
                    on_block(block)
        # Propagate EOF (or failure) so the peer's pump finishes too.
        with contextlib.suppress(OSError):
            destination.shutdown(socket.SHUT_WR)
        with contextlib.suppress(OSError):
            source.shutdown(socket.SHUT_RD)

    def _discard(self, *sockets: socket.socket) -> None:
        with self._lock:
            for item in sockets:
                self._connections.discard(item)
        for item in sockets:
            item.close()

    def close(self) -> None:
        if self.listener is None:
            return
        with self._capacity:
            self._closing = True
            self._capacity.notify_all()
            connections = list(self._connections)
        # shutdown() wakes the blocked accept(); close() alone does not.
        with contextlib.suppress(OSError):
            self.listener.shutdown(socket.SHUT_RDWR)
        for item in connections:
            with contextlib.suppress(OSError):
                item.shutdown(socket.SHUT_RDWR)
        with self._lock:
            threads = [self._accept_thread, *self._pumps]
        for thread in threads:
            if thread is not None:
                thread.join(timeout=5)
        alive = [
            thread for thread in threads if thread is not None and thread.is_alive()
        ]
        if alive:
            raise RuntimeError(
                f"{len(alive)} sandbox proxy relay thread(s) did not stop; "
                "retaining the relay for cleanup retry"
            )
        self.listener.close()
        self.listener = None
        self._accept_thread = None
        self._pumps = []
