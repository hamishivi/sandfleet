from __future__ import annotations

import asyncio
import base64
import socket
import socketserver
import threading
import time

import pytest

from sandfleet import egress
from sandfleet.egress import (
    Denied,
    EgressForwarder,
    EgressPolicy,
    EgressSettings,
    parse_allowed_hosts,
)

ALLOWLIST = EgressPolicy(allow_any=False)
ANY = EgressPolicy(allow_any=True)


@pytest.mark.parametrize(
    "host",
    ["pypi.org", "files.pythonhosted.org", "github.com", "codeload.github.com"],
)
def test_allowlist_permits_package_registries(host):
    ALLOWLIST.check(host, 443)


def test_allowlist_denies_other_hosts_and_allow_any_permits_them():
    with pytest.raises(Denied, match="not allowlisted"):
        ALLOWLIST.check("example.com", 443)
    with pytest.raises(Denied, match="not allowlisted"):
        ALLOWLIST.check("evil-pypi.org", 443)
    ANY.check("example.com", 443)


def test_allowlist_wildcards_match_subdomains_only():
    policy = EgressPolicy(
        allow_any=False, allowed_hosts=frozenset(parse_allowed_hosts("*.example.com"))
    )
    policy.check("a.b.example.com", 443)
    with pytest.raises(Denied):
        policy.check("example.com", 443)


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1",
        "10.1.2.3",
        "192.168.0.1",
        "172.16.0.1",
        "169.254.169.254",
        "100.64.0.1",
        "::1",
        "fe80::1",
        "fd00::1",
        "0.0.0.0",
        "64:ff9b::a00:1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b:1::808:808",
        "::ffff:127.0.0.1",
        "2002:7f00:1::",
        "fd00:ec2::254",
    ],
)
@pytest.mark.parametrize("policy", [ALLOWLIST, ANY])
def test_internal_literal_addresses_are_always_refused(policy, host):
    with pytest.raises(Denied, match="non-public"):
        policy.check(host, 443)


def test_public_literal_ip_needs_allow_any():
    with pytest.raises(Denied, match="literal"):
        ALLOWLIST.check("1.1.1.1", 443)
    ANY.check("1.1.1.1", 443)
    ANY.check("64:ff9b::808:808", 443)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "gpu-node-12",
        "metadata.google.internal",
        "printer.local",
        "app.localhost",
        "2130706433",
        "127.1",
        "0x7f.0.0.1",
        "*.github.com",
        "bad_host.com",
    ],
)
@pytest.mark.parametrize("policy", [ALLOWLIST, ANY])
def test_internal_or_numeric_hostnames_are_refused(policy, host):
    with pytest.raises(Denied):
        policy.check(host, 443)


def test_only_web_ports_are_allowed():
    with pytest.raises(Denied, match="port"):
        ANY.check("github.com", 22)
    ANY.check("github.com", 80)


def test_link_local_resolution_requires_operator_opt_in():
    ANY.check_resolved(["151.101.0.223"])
    with pytest.raises(Denied, match="internal address"):
        ANY.check_resolved(["169.254.12.34"])
    fake_dns = EgressPolicy(allow_any=True, link_local_dns=True)
    fake_dns.check_resolved(["169.254.12.34", "151.101.0.223"])
    for address in ("10.0.0.7", "127.0.0.1", "fd00::5", "64:ff9b::a00:7"):
        with pytest.raises(Denied, match="internal address"):
            fake_dns.check_resolved(["169.254.12.34", address])


@pytest.mark.parametrize(
    "address", ["169.254.169.254", "169.254.170.2", "169.254.169.253", "fd00:ec2::254"]
)
def test_metadata_addresses_are_refused_even_with_link_local_dns(address):
    fake_dns = EgressPolicy(allow_any=True, link_local_dns=True)
    with pytest.raises(Denied, match="internal address"):
        fake_dns.check_resolved([address])


def test_allowlist_configuration_is_validated():
    assert parse_allowed_hosts(" PyPI.org ,*.Example.com") == (
        "pypi.org",
        "*.example.com",
    )
    for value in ("", "10.0.0.1", "foo", "a.*.com"):
        with pytest.raises(ValueError):
            parse_allowed_hosts(value)


def test_settings_validate_limits_and_render_agent_arguments():
    for invalid in (
        {"mode": "on"},
        {"max_connections": 4, "max_tunnels_per_lease": 5},
        {"max_tunnels_per_lease": 0},
        {"idle_timeout_seconds": 0},
    ):
        with pytest.raises(ValueError):
            EgressSettings(**invalid)
    assert EgressSettings().agent_arguments() == []
    arguments = EgressSettings(
        mode="any", link_local_dns=True, allowed_hosts=("pypi.org",)
    ).agent_arguments()
    assert arguments[:4] == [
        "--internet",
        "any",
        "--internet-allowed-hosts",
        "pypi.org",
    ]
    assert arguments[-1] == "--internet-link-local-dns"


def test_upstream_tries_each_checked_address(monkeypatch):
    forwarder = EgressForwarder(EgressSettings(mode="any"), host="127.0.0.1")
    attempts = []

    async def open_connection(address, port):
        attempts.append(address)
        if address == "151.101.0.1":
            raise ConnectionRefusedError("refused")
        return "reader", "writer"

    async def scenario(addresses):
        loop = asyncio.get_running_loop()

        async def getaddrinfo(*_args, **_kwargs):
            return [(None, None, None, "", (address, 443)) for address in addresses]

        loop.getaddrinfo = getaddrinfo
        return await forwarder._open_upstream("pypi.org", 443)

    monkeypatch.setattr(egress.asyncio, "open_connection", open_connection)
    assert asyncio.run(scenario(["151.101.0.1", "151.101.64.1"])) == (
        "reader",
        "writer",
    )
    assert attempts == ["151.101.0.1", "151.101.64.1"]
    with pytest.raises(Denied, match="internal address"):
        asyncio.run(scenario(["151.101.0.1", "10.0.0.1"]))


class Upstream(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self):
        self.received: list[bytes] = []

        server = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                data = self.request.recv(65536)
                server.received.append(data)
                if data.startswith(b"GET "):
                    self.request.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
                    )
                    return
                if data.startswith(b"HOLD"):
                    # Ignore the client's half-close and keep the socket open.
                    while self.request.recv(65536):
                        pass
                    time.sleep(30)
                    return
                while data:
                    self.request.sendall(data)
                    data = self.request.recv(65536)

        super().__init__(("127.0.0.1", 0), Handler)


@pytest.fixture
def make_proxy(monkeypatch):
    created = []

    def make(**settings):
        upstream = Upstream()
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        logs: list[str] = []
        forwarder = EgressForwarder(
            EgressSettings(mode="allowlist", **settings),
            host="127.0.0.1",
            log=logs.append,
        )
        opened: list[tuple[str, int]] = []

        async def open_upstream(host, port):
            opened.append((host, port))
            return await asyncio.open_connection(*upstream.server_address)

        monkeypatch.setattr(forwarder, "_open_upstream", open_upstream)
        forwarder.start()
        forwarder.logs, forwarder.opened, forwarder.upstream = logs, opened, upstream
        created.append(forwarder)
        return forwarder

    yield make
    for forwarder in created:
        forwarder.close()
        forwarder.upstream.shutdown()
        forwarder.upstream.server_close()


@pytest.fixture
def proxy(make_proxy):
    return make_proxy()


def credentials(proxy_url: str) -> str:
    userinfo = proxy_url.removeprefix("http://").rsplit("@", 1)[0]
    return base64.b64encode(userinfo.encode()).decode()


def send(forwarder, request: str, *, source: str = "127.0.0.1") -> socket.socket:
    connection = socket.create_connection(
        ("127.0.0.1", forwarder.port), timeout=5, source_address=(source, 0)
    )
    connection.sendall(request.encode())
    return connection


def response(connection: socket.socket) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = connection.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


def connect_request(target: str, auth: str | None) -> str:
    header = "" if auth is None else f"Proxy-Authorization: Basic {auth}\r\n"
    return f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n{header}\r\n"


def test_unauthenticated_and_unknown_credentials_get_407(proxy):
    proxy.grant("lease-a")
    for auth in (
        None,
        base64.b64encode(b"lease-a:guess").decode(),
        base64.b64encode(b"lease-z:guess").decode(),
    ):
        with send(proxy, connect_request("pypi.org:443", auth)) as connection:
            head = response(connection)
        assert head.startswith(b"HTTP/1.1 407")
        assert b'Proxy-Authenticate: Basic realm="sandfleet"' in head
    assert proxy.opened == []
    assert proxy.metrics_snapshot()["egress_auth_failures"] == 3


def test_another_leases_token_is_rejected(proxy):
    url_a = proxy.grant("lease-a")
    url_b = proxy.grant("lease-b")
    token_b = url_b.removeprefix("http://").split("@")[0].split(":")[1]
    forged = base64.b64encode(f"lease-a:{token_b}".encode()).decode()
    with send(proxy, connect_request("pypi.org:443", forged)) as connection:
        assert response(connection).startswith(b"HTTP/1.1 407")
    with send(proxy, connect_request("pypi.org:443", credentials(url_a))) as connection:
        assert response(connection).startswith(b"HTTP/1.1 200")


def test_connect_tunnel_relays_bytes_and_logs_the_lease(proxy):
    url = proxy.grant("lease-a")
    with send(proxy, connect_request("pypi.org:443", credentials(url))) as connection:
        assert response(connection) == b"HTTP/1.1 200 Connection established\r\n\r\n"
        connection.sendall(b"hello")
        assert connection.recv(5) == b"hello"
    assert proxy.opened == [("pypi.org", 443)]
    for _ in range(100):
        if any("allowed" in line for line in proxy.logs):
            break
        time.sleep(0.02)
    assert proxy.logs == [
        "egress lease=lease-a target=pypi.org:443 allowed sent=5 received=5"
    ]
    metrics = proxy.metrics_snapshot()
    assert metrics["egress_connections_allowed"] == 1
    assert metrics["egress_bytes_sent"] == metrics["egress_bytes_received"] == 5


def test_policy_denial_returns_403_after_authentication(proxy):
    url = proxy.grant("lease-a")
    for target in ("example.com:443", "10.0.0.1:443", "169.254.169.254:80"):
        with send(proxy, connect_request(target, credentials(url))) as connection:
            assert response(connection).startswith(b"HTTP/1.1 403")
    assert proxy.opened == []
    assert proxy.metrics_snapshot()["egress_connections_denied"] == 3


def test_plain_http_is_rewritten_without_proxy_credentials(proxy):
    url = proxy.grant("lease-a")
    request = (
        "GET http://pypi.org/simple/?q=1 HTTP/1.1\r\nHost: pypi.org\r\n"
        f"Proxy-Authorization: Basic {credentials(url)}\r\n"
        "Proxy-Connection: keep-alive\r\n\r\n"
    )
    with send(proxy, request) as connection:
        assert response(connection).startswith(b"HTTP/1.1 200 OK")
    forwarded = proxy.upstream.received[0].decode()
    assert forwarded.startswith("GET /simple/?q=1 HTTP/1.1\r\n")
    assert "Proxy-" not in forwarded
    assert "Connection: close" in forwarded
    assert proxy.opened == [("pypi.org", 80)]


def test_revoked_lease_loses_access_and_open_tunnels(proxy):
    url = proxy.grant("lease-a")
    tunnel = send(proxy, connect_request("pypi.org:443", credentials(url)))
    assert response(tunnel).startswith(b"HTTP/1.1 200")
    proxy.revoke("lease-a")
    assert tunnel.recv(1) == b""
    tunnel.close()
    with send(proxy, connect_request("pypi.org:443", credentials(url))) as connection:
        assert response(connection).startswith(b"HTTP/1.1 407")


def test_connections_from_other_hosts_are_refused(proxy):
    url = proxy.grant("lease-a")
    with send(
        proxy, connect_request("pypi.org:443", credentials(url)), source="127.0.0.2"
    ) as connection:
        head = response(connection)
    assert head.startswith(b"HTTP/1.1 403")
    assert b"not on this node" in head
    assert proxy.opened == []


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def tunnel(forwarder, url):
    connection = send(forwarder, connect_request("pypi.org:443", credentials(url)))
    assert response(connection).startswith(b"HTTP/1.1 200")
    return connection


def closed_within(connection, seconds):
    connection.settimeout(seconds)
    try:
        return connection.recv(4096) == b""
    except ConnectionResetError:
        return True
    except TimeoutError:
        return False


def test_non_ascii_password_is_rejected_not_crashed(proxy):
    proxy.grant("lease-a")
    auth = base64.b64encode("lease-a:p\u00e4ss".encode()).decode()
    with send(proxy, connect_request("pypi.org:443", auth)) as connection:
        assert response(connection).startswith(b"HTTP/1.1 407")


def test_plain_http_host_header_names_the_checked_destination(proxy):
    url = proxy.grant("lease-a")
    request = (
        "GET http://pypi.org/simple/ HTTP/1.1\r\nHost: evil.example.com\r\n"
        f"Proxy-Authorization: Basic {credentials(url)}\r\n\r\n"
    )
    with send(proxy, request) as connection:
        assert response(connection).startswith(b"HTTP/1.1 200 OK")
    forwarded = proxy.upstream.received[0].decode()
    assert "\r\nHost: pypi.org\r\n" in forwarded
    assert "evil" not in forwarded


def test_global_connection_cap_rejects_excess_connections(make_proxy):
    forwarder = make_proxy(max_connections=2, max_tunnels_per_lease=2)
    url = forwarder.grant("lease-a")
    tunnels = [tunnel(forwarder, url) for _ in range(2)]
    with send(forwarder, connect_request("pypi.org:443", credentials(url))) as extra:
        assert response(extra).startswith(b"HTTP/1.1 503")
    assert forwarder.metrics_snapshot()["egress_connections_rejected"] == 1
    tunnels[0].close()
    wait_for(lambda: len(forwarder._handlers) < 2)
    tunnel(forwarder, url).close()
    tunnels[1].close()


def test_per_lease_tunnel_cap_leaves_other_leases_working(make_proxy):
    forwarder = make_proxy(max_tunnels_per_lease=1)
    url_a, url_b = forwarder.grant("lease-a"), forwarder.grant("lease-b")
    first = tunnel(forwarder, url_a)
    with send(forwarder, connect_request("pypi.org:443", credentials(url_a))) as extra:
        assert response(extra).startswith(b"HTTP/1.1 429")
    tunnel(forwarder, url_b).close()
    first.close()


def test_unauthenticated_connections_time_out_and_are_capped(monkeypatch, make_proxy):
    monkeypatch.setattr(egress, "AUTH_TIMEOUT_SECONDS", 0.5)
    monkeypatch.setattr(egress, "MAX_PENDING_CONNECTIONS", 2)
    forwarder = make_proxy()
    idle = [
        socket.create_connection(("127.0.0.1", forwarder.port), timeout=5)
        for _ in range(2)
    ]
    wait_for(lambda: forwarder._pending == 2)
    with socket.create_connection(("127.0.0.1", forwarder.port), timeout=5) as extra:
        assert response(extra).startswith(b"HTTP/1.1 503")
    for connection in idle:
        assert closed_within(connection, 3)
        connection.close()
    wait_for(lambda: forwarder._pending == 0)
    url = forwarder.grant("lease-a")
    tunnel(forwarder, url).close()


def test_idle_tunnels_are_closed(make_proxy):
    forwarder = make_proxy(idle_timeout_seconds=1)
    url = forwarder.grant("lease-a")
    connection = tunnel(forwarder, url)
    connection.sendall(b"ping")
    assert connection.recv(4) == b"ping"
    assert closed_within(connection, 5)
    connection.close()
    wait_for(lambda: any("closed=idle" in line for line in forwarder.logs))
    assert forwarder.metrics_snapshot()["egress_idle_timeouts"] == 1


def test_half_closed_tunnels_close_after_a_grace_period(monkeypatch, make_proxy):
    monkeypatch.setattr(egress, "HALF_CLOSE_GRACE_SECONDS", 0.5)
    forwarder = make_proxy()
    url = forwarder.grant("lease-a")
    connection = tunnel(forwarder, url)
    connection.sendall(b"HOLD")
    connection.shutdown(socket.SHUT_WR)
    assert closed_within(connection, 5)
    connection.close()
    wait_for(lambda: not forwarder._handlers)


def test_close_cancels_unauthenticated_and_established_connections(make_proxy):
    forwarder = make_proxy()
    url = forwarder.grant("lease-a")
    established = tunnel(forwarder, url)
    unauthenticated = socket.create_connection(("127.0.0.1", forwarder.port), timeout=5)
    wait_for(lambda: forwarder._pending == 1)
    forwarder.close()
    assert closed_within(unauthenticated, 3)
    assert closed_within(established, 3)
    assert any("reason=shutdown" in line for line in forwarder.logs)
    assert not any("reason=revoked" in line for line in forwarder.logs)
    unauthenticated.close()
    established.close()
