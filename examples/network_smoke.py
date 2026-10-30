#!/usr/bin/env python3
"""Opt-in live check of same-node network isolation, egress policy, and restart.

Offline leases must have only loopback. With --internet, leases must reach HTTPS
only through the egress forwarder; DNS and direct connections must still fail.

Use a dedicated controller and a unique pool name. This script creates exactly
one worker allocation with two one-CPU, 512-MiB sandbox slots, then deletes it.
It reads role credentials from restricted files, never command-line values.
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path
from typing import Any

from sandfleet.client import SandboxLease, SandfleetClient
from sandfleet.errors import SandfleetError

SERVER = """import http.server, sys
marker = sys.argv[2].encode()
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(marker)
    def log_message(self, *args):
        pass
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


def run_python(lease: SandboxLease, source: str) -> Any:
    result = lease.run_command(f"python3 -c {shlex.quote(source)}", timeout=40)
    if result.exit_code:
        raise AssertionError(f"Sandbox probe failed: {result!r}")
    return json.loads(result.stdout)


def network_details(lease: SandboxLease, url: str) -> dict[str, Any]:
    """Probe DNS, proxied HTTPS, direct egress, and routes from inside a sandbox."""
    return run_python(
        lease,
        "import json, os, socket, urllib.parse, urllib.request\n"
        f"url={url!r}\n"
        "def attempt(call):\n"
        "    try:\n"
        "        return call()\n"
        "    except OSError as error:\n"
        "        return None\n"
        "hostname=urllib.parse.urlsplit(url).hostname\n"
        "addresses=attempt(lambda: sorted({x[4][0] for x in "
        "socket.getaddrinfo(hostname,443)}))\n"
        "status=attempt(lambda: urllib.request.urlopen(url,timeout=15).status)\n"
        "direct=attempt(lambda: socket.create_connection(('1.1.1.1',443),timeout=5)"
        ".close() or True)\n"
        "routes=open('/proc/net/route').read().splitlines()[1:]\n"
        "def net_caps():\n"
        "    caps=dict(line.split(':\\t') for line in "
        "open('/proc/self/status').read().splitlines() if line.startswith('Cap'))\n"
        "    return any(int(caps[name],16)>>bit&1 for name in ('CapEff','CapBnd') "
        "for bit in (12,13,19))\n"
        "print(json.dumps({'netns':os.readlink('/proc/self/ns/net'),"
        "'addresses':addresses,'https_status':status,'direct':bool(direct),"
        "'routes':routes,"
        "'net_caps':net_caps()}))\n",
    )


def listen(lease: SandboxLease, port: int, marker: str) -> None:
    lease.write_file("/tmp/sandfleet-network-smoke-server.py", SERVER)
    result = run_python(
        lease,
        "import json, subprocess; "
        f"p=subprocess.Popen(['python3','/tmp/sandfleet-network-smoke-server.py',"
        f"{str(port)!r},{marker!r}],stdin=subprocess.DEVNULL,"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True); "
        "print(json.dumps({'pid':p.pid}))",
    )
    if not result["pid"]:
        raise AssertionError("Background server did not report its PID")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        response = local_request(lease, port)
        if response == marker:
            return
        if response is not None:
            raise AssertionError(f"Wrong service on port {port}: {response!r}")
        time.sleep(0.2)
    raise TimeoutError(f"Background service did not become ready on port {port}")


def local_request(lease: SandboxLease, port: int) -> str | None:
    return run_python(
        lease,
        "import json, urllib.request, urllib.error\n"
        "try:\n"
        f"    response=urllib.request.urlopen('http://127.0.0.1:{port}',timeout=2)\n"
        "    value=response.read().decode()\n"
        "except urllib.error.HTTPError:\n"
        "    raise\n"
        "except (urllib.error.URLError,TimeoutError):\n"
        "    value=None\n"
        "print(json.dumps(value))\n",
    )


def port_open(lease: SandboxLease, port: int) -> bool:
    return run_python(
        lease,
        "import json, socket\n"
        "try:\n"
        f"    connection=socket.create_connection(('127.0.0.1',{port}),timeout=2)\n"
        "except OSError:\n"
        "    value=False\n"
        "else:\n"
        "    connection.close()\n"
        "    value=True\n"
        "print(json.dumps(value))\n",
    )


def verify_cgroups(statuses: list[dict[str, Any]]) -> None:
    paths = []
    for status in statuses:
        cgroup = status["cgroup"]
        if not cgroup["cpuset.cpus.effective"]:
            raise AssertionError(f"Missing effective CPU set: {cgroup!r}")
        if not 0 < int(cgroup["memory.max.effective"]) <= 512 * 1024 * 1024:
            raise AssertionError(f"Memory limit exceeds 512 MiB: {cgroup!r}")
        paths.append(cgroup["path"])
    if len(set(paths)) != len(paths):
        raise AssertionError(f"Sandbox steps share a cgroup: {paths!r}")


def verify_namespaces(
    statuses: list[dict[str, Any]], networks: list[dict[str, Any]]
) -> None:
    if len({entry["netns"] for entry in networks}) != len(networks):
        raise AssertionError("Sandboxes share a network namespace")
    for status, details in zip(statuses, networks, strict=True):
        network = status["network"]
        if network["namespace"] != details["netns"]:
            raise AssertionError("Command did not run in the reported namespace")
        if network["namespace"] == network["host_namespace"]:
            raise AssertionError("Sandbox retained the host network namespace")
        if network["mode"] != ("loopback" if network["internet"] else "none"):
            raise AssertionError("Sandbox did not report a supported isolated network")
        if network["helper_pid"] is not None:
            raise AssertionError("Sandboxes must not run a network helper process")
        # appinit (PID 1) keeps its caps whatever --drop-caps says; only
        # sandbox processes are checked.
        if details["net_caps"]:
            raise AssertionError(
                f"CAP_NET_ADMIN/CAP_NET_RAW/CAP_SYS_PTRACE not dropped: {details!r}"
            )
        if details["addresses"] or details["direct"] or details["routes"]:
            raise AssertionError(f"Sandbox bypasses the network policy: {details!r}")
        expected = 200 if network["internet"] else None
        if details["https_status"] != expected:
            raise AssertionError(f"HTTPS should give {expected}: {details!r}")


def verify_listeners(leases: list[SandboxLease], port: int) -> None:
    listen(leases[0], port, "sandbox-a")
    listen(leases[0], port + 1, "only-a")
    if port_open(leases[1], port):
        raise AssertionError("Sandbox B can reach sandbox A over localhost")
    listen(leases[1], port, "sandbox-b")
    observed = [local_request(lease, port) for lease in leases]
    if observed != ["sandbox-a", "sandbox-b"]:
        raise AssertionError(f"Same-port services crossed sandboxes: {observed!r}")
    if port_open(leases[1], port + 1):
        raise AssertionError("Sandbox B can reach A's private localhost port")


def verify_restart(
    leases: list[SandboxLease],
    before: dict[str, Any],
    network_before: dict[str, Any],
    *,
    image: str,
    url: str,
    port: int,
) -> dict[str, Any]:
    restarted = leases[0].restart(image)
    after = leases[0].status()
    if after["worker_pid"] != before["worker_pid"]:
        raise AssertionError("Restart replaced the persistent worker process")
    if after["cgroup"]["path"] != before["cgroup"]["path"]:
        raise AssertionError("Restart replaced the persistent step cgroup")
    if restarted["instance_name"] == before["instance_name"]:
        raise AssertionError("Restart reused the old Apptainer instance")
    network_after = network_details(leases[0], url)
    verify_namespaces([restarted], [network_after])
    # Linux may recycle a namespace inode after its final reference is closed.
    # Fresh instance identity and absence of the old listener prove replacement;
    # inode inequality alone cannot establish this across sequential lifetimes.
    inode_reused = network_after["netns"] == network_before["netns"]
    if port_open(leases[0], port):
        raise AssertionError("Old sandbox service survived restart")
    listen(leases[0], port, "sandbox-a-restarted")
    if local_request(leases[1], port) != "sandbox-b":
        raise AssertionError("Restart disturbed sandbox B's service")
    return {
        "status": after,
        "network": network_after,
        "namespace_inode_reused": inode_reused,
        "old_listener_removed": True,
        "same_port_reused": True,
        "other_sandbox_survived": True,
    }


def cleanup(
    service: SandfleetClient, leases: list[SandboxLease], pool: str
) -> dict[str, Any]:
    errors: list[Exception] = []
    for lease in reversed(leases):
        try:
            lease.close()
        except Exception as error:
            errors.append(error)
    try:
        deleted = service.delete_pool(pool)
    except Exception as error:
        errors.append(error)
    if errors:
        raise ExceptionGroup("Network smoke cleanup failed", errors)
    return deleted


def require_unused_pool(client: SandfleetClient, name: str) -> None:
    try:
        client.pool(name)
    except SandfleetError as error:
        if error.status == 404:
            return
        raise
    raise ValueError(f"Pool {name!r} already exists; use a unique test pool name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--service-token-file", type=Path, required=True)
    parser.add_argument("--client-token-file", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--port", type=int, default=38171)
    parser.add_argument("--https-url", default="https://example.com/")
    parser.add_argument(
        "--internet", action="store_true", help="Acquire leases with sandbox internet"
    )
    args = parser.parse_args()
    if not args.https_url.startswith("https://"):
        parser.error("--https-url must use HTTPS")
    if not 0 < args.port < 65535:
        parser.error("--port and the next port must be valid TCP ports")
    for path in (args.service_token_file, args.client_token_file):
        if path.stat().st_mode & 0o077:
            parser.error(f"Credential file must exclude group/other access: {path}")
    return args


def main() -> int:
    args = parse_args()
    service = SandfleetClient(args.url, args.service_token_file.read_text().strip())
    client = SandfleetClient(args.url, args.client_token_file.read_text().strip())
    require_unused_pool(client, args.pool)
    leases: list[SandboxLease] = []
    evidence: dict[str, Any] = {"pool": args.pool}
    primary: BaseException | None = None
    try:
        service.create_pool(
            args.pool, desired_capacity=2, slots_per_job=2, cpus=1, memory_mb=512
        )
        for _ in range(2):
            leases.append(
                client.acquire(args.pool, timeout=args.timeout, internet=args.internet)
            )
        if leases[0].agent_url != leases[1].agent_url:
            raise AssertionError("Test sandboxes did not land on the same worker")
        evidence["agent_url"] = leases[0].agent_url
        evidence["worker_submissions"] = client.pool(args.pool)["submissions"]
        print("Two slots acquired on one worker", flush=True)
        evidence["prepared_status"] = [lease.status() for lease in leases]
        verify_cgroups(evidence["prepared_status"])
        statuses = [lease.start(args.image, command_timeout=60) for lease in leases]
        verify_cgroups(statuses)
        evidence["initial_status"] = statuses
        networks = [network_details(lease, args.https_url) for lease in leases]
        verify_namespaces(statuses, networks)
        evidence["initial_network"] = networks
        verify_listeners(leases, args.port)
        evidence["same_port_and_cross_slot_checks"] = "passed"
        print("Separate localhost services and network policy verified", flush=True)
        evidence["restart"] = verify_restart(
            leases,
            leases[0].status(),
            networks[0],
            image=args.image,
            url=args.https_url,
            port=args.port,
        )
        evidence["passed"] = True
    except BaseException as error:
        primary = error
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            evidence["pool_cleanup"] = cleanup(service, leases, args.pool)
        except Exception as error:
            evidence["cleanup_error"] = str(error)
            if primary is None:
                raise
            primary.add_note(f"Cleanup also failed: {error}")
        finally:
            args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
