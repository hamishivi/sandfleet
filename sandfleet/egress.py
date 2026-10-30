"""Opt-in, per-lease authenticated HTTP egress forwarder for sandboxes.

Sandboxes are offline by default. When an operator enables sandbox internet,
each worker runs this forwarder in the host network namespace, where outbound
connections behave like any other host process. Only leases that requested
internet receive proxy credentials; every other connection gets 407.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hmac
import ipaddress
import re
import resource
import secrets
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass

INTERNET_POLICIES = ("off", "allowlist", "any")
DEFAULT_ALLOWED_HOSTS = (
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "github.com",
    "codeload.github.com",
    "objects.githubusercontent.com",
    "raw.githubusercontent.com",
    "release-assets.githubusercontent.com",
    "crates.io",
    "index.crates.io",
    "static.crates.io",
    "proxy.golang.org",
    "sum.golang.org",
    "repo.maven.apache.org",
    "conda.anaconda.org",
    "repo.anaconda.com",
)
ALLOWED_PORTS = frozenset({80, 443})
# Operators must explicitly allow link-local DNS answers. Literal link-local
# requests are always refused, independently of this DNS policy.
LINK_LOCAL_NETWORK = ipaddress.ip_network("169.254.0.0/16")
METADATA_ADDRESSES = frozenset(
    ipaddress.ip_address(value)
    for value in (
        "169.254.169.254",
        "169.254.169.253",
        "169.254.169.123",
        "169.254.170.2",
        "169.254.170.23",
        "100.100.100.200",
        "fd00:ec2::254",
        "fd00:ec2::253",
        "fd00:ec2::23",
    )
)
NAT64_NETWORK = ipaddress.ip_network("64:ff9b::/96")
LOCAL_NAT64_NETWORK = ipaddress.ip_network("64:ff9b:1::/48")
INTERNAL_SUFFIXES = (
    ".local",
    ".localhost",
    ".localdomain",
    ".internal",
    ".intranet",
    ".lan",
    ".home.arpa",
)
HOST_PATTERN = re.compile(
    r"(?:\*\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
)
MAX_HEAD_BYTES = 64 * 1024
AUTH_TIMEOUT_SECONDS = 10.0
MAX_PENDING_CONNECTIONS = 32
CONNECT_TIMEOUT_SECONDS = 15.0
HALF_CLOSE_GRACE_SECONDS = 30.0
COPY_CHUNK_BYTES = 64 * 1024
# Descriptors kept free for the agent API, step sockets, and logs.
AGENT_RESERVED_FDS = 512
HOP_BY_HOP_HEADERS = frozenset(
    {"proxy-authorization", "proxy-connection", "connection", "keep-alive", "host"}
)
STATUS_PHRASES = {
    400: "Bad Request",
    403: "Forbidden",
    407: "Proxy Authentication Required",
    429: "Too Many Requests",
    431: "Request Header Fields Too Large",
    502: "Bad Gateway",
    503: "Service Unavailable",
}


class Denied(Exception):
    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def parse_allowed_hosts(value: str) -> tuple[str, ...]:
    hosts = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not hosts:
        raise ValueError("internet allowlist must name at least one host")
    for host in hosts:
        if HOST_PATTERN.fullmatch(host) is None:
            raise ValueError(
                f"internet allowlist entry {host!r} must be a DNS name or *.domain"
            )
    return hosts


def is_public(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address, including any IPv4 embedded in IPv6, is public."""
    if not address.is_global or address in METADATA_ADDRESSES:
        return False
    if address.version == 4:
        return True
    if address in LOCAL_NAT64_NETWORK:
        return False
    embedded = [address.ipv4_mapped, address.sixtofour]
    if address.teredo is not None:
        embedded.append(address.teredo[1])
    if address in NAT64_NETWORK:
        embedded.append(ipaddress.IPv4Address(int(address) & 0xFFFFFFFF))
    return all(is_public(item) for item in embedded if item is not None)


@dataclass(frozen=True)
class EgressPolicy:
    """Destination policy applied to the requested name and its resolution."""

    allow_any: bool
    allowed_hosts: frozenset[str] = frozenset(DEFAULT_ALLOWED_HOSTS)
    link_local_dns: bool = False

    def _allowlisted(self, host: str) -> bool:
        if host in self.allowed_hosts:
            return True
        labels = host.split(".")
        return any(
            "*." + ".".join(labels[index:]) in self.allowed_hosts
            for index in range(1, len(labels) - 1)
        )

    def check(self, host: str, port: int) -> None:
        """Raise Denied unless the requested destination may be contacted."""
        if port not in ALLOWED_PORTS:
            raise Denied(403, f"port {port} is not allowed")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None:
            if not is_public(address):
                raise Denied(403, "non-public address")
            if not self.allow_any:
                raise Denied(403, "literal IP addresses are not allowlisted")
            return
        # A final alphabetic label rejects numeric host spellings that
        # resolvers would turn into IP addresses, e.g. 2130706433 or 127.1.
        if HOST_PATTERN.fullmatch(host) is None or host.startswith("*."):
            raise Denied(403, "invalid or internal hostname")
        if host.endswith(INTERNAL_SUFFIXES):
            raise Denied(403, "internal hostname")
        if not self.allow_any and not self._allowlisted(host):
            raise Denied(403, "host is not allowlisted")

    def check_resolved(self, addresses: list[str]) -> None:
        """Refuse names that resolve to internal or metadata addresses."""
        for value in addresses:
            address = ipaddress.ip_address(value.split("%", 1)[0])
            if is_public(address):
                continue
            if (
                self.link_local_dns
                and address in LINK_LOCAL_NETWORK
                and address not in METADATA_ADDRESSES
            ):
                continue
            raise Denied(403, "host resolves to an internal address")


@dataclass(frozen=True)
class EgressSettings:
    """Operator configuration carried from the controller to every worker."""

    mode: str = "off"
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    link_local_dns: bool = False
    max_connections: int = 256
    max_tunnels_per_lease: int = 32
    idle_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.mode not in INTERNET_POLICIES:
            raise ValueError(f"internet mode must be one of {INTERNET_POLICIES!r}")
        if not 1 <= self.max_tunnels_per_lease <= self.max_connections:
            raise ValueError(
                "internet tunnel limits must satisfy 1 <= per-lease <= total"
            )
        if self.idle_timeout_seconds < 1:
            raise ValueError("internet idle timeout must be positive")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def policy(self) -> EgressPolicy:
        return EgressPolicy(
            allow_any=self.mode == "any",
            allowed_hosts=frozenset(self.allowed_hosts),
            link_local_dns=self.link_local_dns,
        )

    def agent_arguments(self) -> list[str]:
        if not self.enabled:
            return []
        return [
            "--internet",
            self.mode,
            "--internet-allowed-hosts",
            ",".join(self.allowed_hosts),
            "--internet-max-connections",
            str(self.max_connections),
            "--internet-max-tunnels-per-lease",
            str(self.max_tunnels_per_lease),
            "--internet-idle-timeout-seconds",
            str(self.idle_timeout_seconds),
            *(["--internet-link-local-dns"] if self.link_local_dns else []),
        ]


def add_egress_arguments(parser: argparse.ArgumentParser, prefix: str) -> None:
    """Add egress options as --<prefix>, --<prefix>-allowed-hosts, ..."""
    defaults = EgressSettings()
    parser.add_argument(
        f"--{prefix}",
        choices=INTERNET_POLICIES,
        default="off",
        help="Let leases opt into internet: allowlisted hosts or any public host",
    )
    parser.add_argument(
        f"--{prefix}-allowed-hosts",
        type=parse_allowed_hosts,
        default=DEFAULT_ALLOWED_HOSTS,
        help="Comma-separated hosts (or *.domain) for the allowlist policy",
    )
    parser.add_argument(
        f"--{prefix}-link-local-dns",
        action="store_true",
        help="Accept operator-approved 169.254.0.0/16 DNS resolutions",
    )
    for name in ("max_connections", "max_tunnels_per_lease", "idle_timeout_seconds"):
        parser.add_argument(
            f"--{prefix}-{name.replace('_', '-')}",
            type=int,
            default=getattr(defaults, name),
        )


def egress_settings_from_args(args: argparse.Namespace, prefix: str) -> EgressSettings:
    dest = prefix.replace("-", "_")
    return EgressSettings(
        mode=getattr(args, dest),
        allowed_hosts=getattr(args, f"{dest}_allowed_hosts"),
        link_local_dns=getattr(args, f"{dest}_link_local_dns"),
        max_connections=getattr(args, f"{dest}_max_connections"),
        max_tunnels_per_lease=getattr(args, f"{dest}_max_tunnels_per_lease"),
        idle_timeout_seconds=getattr(args, f"{dest}_idle_timeout_seconds"),
    )


def split_host_port(authority: str, default_port: int | None) -> tuple[str, int]:
    if authority.startswith("["):
        host, separator, rest = authority[1:].partition("]")
        if not separator or (rest and not rest.startswith(":")):
            raise Denied(400, "malformed authority")
        port_text = rest[1:]
    else:
        host, separator, port_text = authority.rpartition(":")
        if not separator:
            host, port_text = authority, ""
    if not port_text:
        if default_port is None:
            raise Denied(400, "port required")
        return host.lower().rstrip("."), default_port
    if not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise Denied(400, "malformed port")
    return host.lower().rstrip("."), int(port_text)


def _basic_credentials(headers: list[tuple[str, str]]) -> tuple[str, bytes] | None:
    values = [value for name, value in headers if name.lower() == "proxy-authorization"]
    if len(values) != 1:
        return None
    scheme, _, encoded = values[0].strip().partition(" ")
    if scheme.lower() != "basic":
        return None
    try:
        decoded = base64.b64decode(encoded.strip(), validate=True)
    except binascii.Error:
        return None
    username, separator, password = decoded.partition(b":")
    if not separator:
        return None
    return username.decode("utf-8", errors="replace"), password


def _ensure_descriptor_budget(needed: int) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft != resource.RLIM_INFINITY and soft < needed:
        target = needed if hard == resource.RLIM_INFINITY else min(needed, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        soft = target
    if soft != resource.RLIM_INFINITY and soft < needed:
        raise RuntimeError(
            f"RLIMIT_NOFILE {soft} cannot hold {needed} descriptors; lower "
            "the internet connection limit"
        )


class EgressForwarder:
    """Run an HTTP CONNECT/plain-HTTP proxy on one worker for its lifetime."""

    def __init__(
        self,
        settings: EgressSettings,
        *,
        host: str,
        port: int = 0,
        log: Callable[[str], None] = print,
    ) -> None:
        self.settings = settings
        self.policy = settings.policy()
        self.host = host
        self.port = port
        self.log = log
        self._grants: dict[str, bytes] = {}
        # Loop-thread state: every handler, unauthenticated count, and the
        # established tunnels of each lease.
        self._handlers: set[asyncio.Task] = set()
        self._pending = 0
        self._tasks: dict[str, set[asyncio.Task]] = {}
        self._closing = False
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.Server | None = None
        self._thread: threading.Thread | None = None
        self.metrics = {
            "egress_connections_allowed": 0,
            "egress_connections_denied": 0,
            "egress_connections_rejected": 0,
            "egress_auth_failures": 0,
            "egress_idle_timeouts": 0,
            "egress_bytes_sent": 0,
            "egress_bytes_received": 0,
        }

    # ---------- lifecycle (called from agent threads) ----------

    def start(self) -> None:
        # Each tunnel holds two descriptors; refuse a cap the process cannot
        # honour without starving the agent API.
        _ensure_descriptor_budget(
            2 * self.settings.max_connections + AGENT_RESERVED_FDS
        )
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            loop.set_exception_handler(
                lambda _loop, context: self.log(
                    f"egress error: {context.get('message')} "
                    f"{context.get('exception')!r}"
                )
            )
            self._loop = loop
            try:
                self._server = loop.run_until_complete(
                    asyncio.start_server(
                        self._handle, self.host, self.port, limit=MAX_HEAD_BYTES
                    )
                )
            except BaseException as error:
                failure.append(error)
                ready.set()
                loop.close()
                return
            self.port = self._server.sockets[0].getsockname()[1]
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._thread = threading.Thread(
            target=run, name="sandfleet-egress", daemon=True
        )
        self._thread.start()
        ready.wait()
        if failure:
            raise RuntimeError(f"Could not start egress forwarder: {failure[0]}")

    def close(self) -> None:
        loop, server = self._loop, self._server
        if loop is None or server is None or loop.is_closed():
            return

        async def shutdown() -> None:
            self._closing = True
            server.close()
            handlers = list(self._handlers)
            for task in handlers:
                task.cancel()
            await asyncio.gather(*handlers, return_exceptions=True)
            loop.stop()

        asyncio.run_coroutine_threadsafe(shutdown(), loop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def grant(self, lease_id: str) -> str:
        """Return a proxy URL whose credentials are valid until revoke()."""
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._grants[lease_id] = token.encode()
        return f"http://{lease_id}:{token}@{self.host}:{self.port}"

    def revoke(self, lease_id: str) -> None:
        """Invalidate a lease's credentials and close its open connections."""
        with self._lock:
            self._grants.pop(lease_id, None)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._cancel_lease, lease_id)

    def metrics_snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.metrics)

    # ---------- event loop ----------

    def _count(self, **deltas: int) -> None:
        with self._lock:
            for name, value in deltas.items():
                self.metrics[f"egress_{name}"] += value

    def _cancel_lease(self, lease_id: str) -> None:
        for task in self._tasks.pop(lease_id, set()):
            task.cancel()

    def _authenticate(self, headers: list[tuple[str, str]]) -> str:
        credentials = _basic_credentials(headers)
        if credentials is not None:
            lease_id, token = credentials
            with self._lock:
                expected = self._grants.get(lease_id)
            if expected is not None and hmac.compare_digest(token, expected):
                return lease_id
        raise Denied(407, "missing or invalid lease credentials")

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        if (
            len(self._handlers) >= self.settings.max_connections
            or self._pending >= MAX_PENDING_CONNECTIONS
        ):
            self._count(connections_rejected=1)
            await self._respond(writer, 503, "forwarder is at its connection limit")
            writer.close()
            return
        self._handlers.add(task)
        self._pending += 1
        try:
            await self._serve(reader, writer, task)
        finally:
            self._handlers.discard(task)
            writer.close()

    async def _serve(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        task: asyncio.Task,
    ) -> None:
        lease_id, target = "-", "-"
        pending = True
        try:
            peer = writer.get_extra_info("peername")[0]
            # Step workers relay internet sandboxes from this node's own
            # namespace, where the forwarder binds loopback. Refuse anyone else.
            if peer != self.host:
                raise Denied(403, f"peer {peer} is not on this node")
            method, uri, headers = await asyncio.wait_for(
                self._read_head(reader), AUTH_TIMEOUT_SECONDS
            )
            if method == "CONNECT":
                host, port = split_host_port(uri, None)
                request = b""
            else:
                host, port, request = self._origin_request(method, uri, headers)
            target = f"{host}:{port}"
            lease_id = self._authenticate(headers)
            self._pending -= 1
            pending = False
            if (
                len(self._tasks.get(lease_id, ()))
                >= self.settings.max_tunnels_per_lease
            ):
                raise Denied(429, "lease is at its tunnel limit")
            self.policy.check(host, port)
            self._tasks.setdefault(lease_id, set()).add(task)
            try:
                await self._forward(reader, writer, lease_id, target, request)
            finally:
                self._tasks.get(lease_id, set()).discard(task)
        except Denied as denial:
            self._count(
                **({"auth_failures": 1} if denial.status == 407 else {}),
                connections_denied=1,
            )
            self.log(
                f"egress lease={lease_id} target={target} denied "
                f"status={denial.status} reason={denial.reason}"
            )
            await self._respond(writer, denial.status, denial.reason)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
            pass
        except asyncio.CancelledError:
            reason = "shutdown" if self._closing else "revoked"
            self.log(f"egress lease={lease_id} target={target} closed reason={reason}")
            raise
        except Exception as error:
            # A forwarder bug must cost one connection, never the agent.
            self.log(f"egress lease={lease_id} target={target} error={error!r}")
        finally:
            if pending:
                self._pending -= 1

    async def _forward(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        lease_id: str,
        target: str,
        request: bytes,
    ) -> None:
        host, _, port = target.rpartition(":")
        upstream_reader, upstream_writer = await self._open_upstream(host, int(port))
        try:
            with self._lock:
                if lease_id not in self._grants:
                    raise Denied(407, "lease credentials were revoked")
            self._count(connections_allowed=1)
            if request:
                upstream_writer.write(request)
            else:
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            sent, received, outcome = await self._relay(
                reader, writer, upstream_reader, upstream_writer
            )
        finally:
            upstream_writer.close()
        self.log(
            f"egress lease={lease_id} target={target} allowed "
            f"sent={sent} received={received}{outcome}"
        )

    @staticmethod
    async def _read_head(
        reader: asyncio.StreamReader,
    ) -> tuple[str, str, list[tuple[str, str]]]:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except asyncio.LimitOverrunError as error:
            raise Denied(431, "request head too large") from error
        lines = head.decode("latin-1").split("\r\n")[:-2]
        parts = lines[0].split(" ")
        if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
            raise Denied(400, "malformed request line")
        headers = []
        for line in lines[1:]:
            name, separator, value = line.partition(":")
            if not separator or not name or name != name.strip():
                raise Denied(400, "malformed header")
            headers.append((name, value.strip()))
        return parts[0], parts[1], headers

    @staticmethod
    def _origin_request(
        method: str, uri: str, headers: list[tuple[str, str]]
    ) -> tuple[str, int, bytes]:
        if not uri.lower().startswith("http://"):
            raise Denied(400, "only http:// URLs and CONNECT are proxied")
        authority, _, path = uri[len("http://") :].partition("/")
        if "@" in authority:
            raise Denied(400, "userinfo in request URL")
        host, port = split_host_port(authority, 80)
        # Host always names the checked destination, so a client cannot
        # front a different virtual host through an allowlisted address.
        host_header = f"[{host}]" if ":" in host else host
        if port != 80:
            host_header += f":{port}"
        lines = [f"{method} /{path} HTTP/1.1", f"Host: {host_header}"]
        lines += [
            f"{name}: {value}"
            for name, value in headers
            if name.lower() not in HOP_BY_HOP_HEADERS
        ]
        lines.append("Connection: close")
        return host, port, ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")

    async def _open_upstream(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        try:
            infos = await asyncio.wait_for(
                loop.getaddrinfo(host, port, type=socket.SOCK_STREAM),
                CONNECT_TIMEOUT_SECONDS,
            )
        except (OSError, asyncio.TimeoutError) as error:
            raise Denied(502, f"could not resolve host: {error}") from error
        addresses = list(dict.fromkeys(info[4][0] for info in infos))
        self.policy.check_resolved(addresses)
        # Connect only to checked addresses so a second lookup cannot rebind.
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await asyncio.wait_for(
                    asyncio.open_connection(address, port), CONNECT_TIMEOUT_SECONDS
                )
            except (OSError, asyncio.TimeoutError) as error:
                last_error = error
        raise Denied(502, f"could not connect upstream: {last_error}")

    async def _copy(
        self,
        source: asyncio.StreamReader,
        sink: asyncio.StreamWriter,
        metric: str,
        totals: dict[str, int],
        last_activity: list[float],
    ) -> None:
        """Copy one direction; time out only when both directions are idle."""
        loop = asyncio.get_running_loop()
        idle_timeout = self.settings.idle_timeout_seconds
        while True:
            try:
                chunk = await asyncio.wait_for(
                    source.read(COPY_CHUNK_BYTES), idle_timeout
                )
            except asyncio.TimeoutError:
                if loop.time() - last_activity[0] < idle_timeout:
                    continue
                raise
            if not chunk:
                break
            last_activity[0] = loop.time()
            sink.write(chunk)
            await sink.drain()
            totals[metric] += len(chunk)
            self._count(**{metric: len(chunk)})
        if sink.can_write_eof():
            sink.write_eof()

    async def _relay(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
    ) -> tuple[int, int, str]:
        last_activity = [asyncio.get_running_loop().time()]
        totals = {"bytes_sent": 0, "bytes_received": 0}

        def copy(source, sink, metric):
            return self._copy(source, sink, metric, totals, last_activity)

        directions = {
            asyncio.ensure_future(copy(client_reader, upstream_writer, "bytes_sent")),
            asyncio.ensure_future(
                copy(upstream_reader, client_writer, "bytes_received")
            ),
        }
        outcome = ""
        try:
            done, _ = await asyncio.wait(
                directions, return_when=asyncio.FIRST_COMPLETED
            )
            if any(isinstance(task.exception(), asyncio.TimeoutError) for task in done):
                outcome = " closed=idle"
            elif not all(task in done for task in directions):
                # One side finished: give the other a bounded half-close grace.
                await asyncio.wait(directions, timeout=HALF_CLOSE_GRACE_SECONDS)
        finally:
            for task in directions:
                task.cancel()
            await asyncio.gather(*directions, return_exceptions=True)
        if outcome:
            self._count(idle_timeouts=1)
        return totals["bytes_sent"], totals["bytes_received"], outcome

    @staticmethod
    async def _respond(writer: asyncio.StreamWriter, status: int, reason: str) -> None:
        body = f"Sandfleet egress: {reason}\n".encode()
        head = [
            f"HTTP/1.1 {status} {STATUS_PHRASES[status]}",
            "Content-Type: text/plain; charset=utf-8",
            f"Content-Length: {len(body)}",
            "Connection: close",
        ]
        if status == 407:
            head.append('Proxy-Authenticate: Basic realm="sandfleet"')
        try:
            writer.write(("\r\n".join(head) + "\r\n\r\n").encode() + body)
            await asyncio.wait_for(writer.drain(), AUTH_TIMEOUT_SECONDS)
        except (ConnectionError, asyncio.TimeoutError):
            pass
