#!/usr/bin/env python3
"""Exercise elastic scale-up, sandbox loss, replacement, and scale-down.

This is an opt-in integration test. It deliberately cancels one Sandfleet
worker through Slurm, so use a dedicated pool name.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from typing import Any

from sandfleet.client import SandboxLease, SandfleetClient
from sandfleet.errors import SandboxLostError
from sandfleet.http import API_VERSION, request_json


def wait_for(description: str, callback: Callable[[], Any], *, timeout: float) -> Any:
    deadline = time.monotonic() + timeout
    last_value: Any = None
    while time.monotonic() < deadline:
        last_value = callback()
        if last_value:
            return last_value
        time.sleep(2)
    raise TimeoutError(
        f"Timed out waiting for {description}; last value={last_value!r}"
    )


def lease_status(client: SandfleetClient, lease: SandboxLease) -> dict[str, Any]:
    return request_json(
        client.url,
        f"/{API_VERSION}/leases/{lease.lease_id}",
        token=client.token,
        timeout=client.timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("SANDFLEET_URL"))
    parser.add_argument("--token", default=os.environ.get("SANDFLEET_TOKEN"))
    parser.add_argument(
        "--client-token",
        default=os.environ.get("SANDFLEET_CLIENT_TOKEN"),
        help="Client-role token for lease acquisition; tokens are role-separated and the service token cannot acquire leases",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--pool", default="elastic-smoke")
    parser.add_argument("--memory-mb", type=int, default=512)
    parser.add_argument(
        "--cpus",
        type=int,
        default=1,
        help="CPUs per sandbox; on thread-granularity clusters use one full core (e.g. 2)",
    )
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not args.url or not args.token or not args.client_token:
        parser.error(
            "--url/--token/--client-token or their SANDFLEET_* equivalents are required"
        )

    client = SandfleetClient(args.url, args.token, timeout=30)
    lease_client = SandfleetClient(args.url, args.client_token, timeout=30)
    leases: list[SandboxLease] = []
    evidence: dict[str, Any] = {"pool": args.pool}
    try:
        with contextlib.suppress(Exception):
            client.delete_pool(args.pool)
        client.create_pool(
            args.pool,
            min_capacity=0,
            max_capacity=2,
            warm_capacity=0,
            slots_per_job=1,
            cpus=args.cpus,
            memory_mb=args.memory_mb,
            idle_timeout_seconds=5,
        )

        # Both requests land within the controller's debounce window and must
        # be represented by one two-task Slurm array submission.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(lease_client.acquire, args.pool, timeout=args.timeout)
                for _ in range(2)
            ]
            leases.extend(future.result() for future in futures)

        initial = lease_client.pool(args.pool)
        if (
            len(initial["submissions"]) != 1
            or initial["submissions"][0]["worker_count"] != 2
        ):
            raise AssertionError(
                f"Expected one two-worker submission, got {initial['submissions']!r}"
            )
        evidence["initial_submission"] = initial["submissions"][0]["submission_id"]
        evidence["initial_worker_count"] = initial["submissions"][0]["worker_count"]

        start_statuses = [
            lease.start(args.image, command_timeout=120) for lease in leases
        ]
        expected_limit = args.memory_mb * 1024 * 1024
        for status in start_statuses:
            observed_limit = status.get("cgroup", {}).get("memory.max.effective")
            if observed_limit is None or int(observed_limit) > expected_limit:
                raise AssertionError(
                    f"Expected effective memory limit <= {expected_limit}, got {status.get('cgroup')!r}"
                )
        for lease in leases:
            result = lease.run_command("printf sandbox-ok", timeout=30)
            if result.exit_code != 0 or result.stdout != "sandbox-ok":
                raise AssertionError(f"Sandbox command failed: {result!r}")
        evidence["memory_max_effective"] = [
            status["cgroup"]["memory.max.effective"] for status in start_statuses
        ]

        agents_by_url = {agent["url"].rstrip("/"): agent for agent in initial["agents"]}
        victim = agents_by_url[leases[0].agent_url]
        victim_worker = victim["worker_key"]
        subprocess.run(["scancel", victim_worker], check=True)
        evidence["cancelled_worker"] = victim_worker

        wait_for(
            "the cancelled lease to be marked lost",
            lambda: (
                status
                if (status := lease_status(lease_client, leases[0]))["status"] == "lost"
                else None
            ),
            timeout=args.timeout,
        )
        try:
            leases[0].run_command("true", timeout=5)
        except SandboxLostError as error:
            evidence["lost_error"] = type(error).__name__
        else:
            raise AssertionError("The cancelled sandbox did not raise SandboxLostError")

        replacement = lease_client.acquire(args.pool, timeout=args.timeout)
        leases.append(replacement)
        replacement_status = replacement.start(args.image, command_timeout=120)
        replacement_result = replacement.run_command(
            "printf replacement-ok", timeout=30
        )
        survivor_result = leases[1].run_command("printf survivor-ok", timeout=30)
        if (
            replacement_result.stdout != "replacement-ok"
            or survivor_result.stdout != "survivor-ok"
        ):
            raise AssertionError(
                "Replacement or surviving sandbox did not execute correctly"
            )
        if int(replacement_status["cgroup"]["memory.max.effective"]) > expected_limit:
            raise AssertionError(
                f"Replacement cgroup limit is too high: {replacement_status!r}"
            )

        healed = lease_client.pool(args.pool)
        if (
            len(healed["submissions"]) != 2
            or healed["submissions"][-1]["worker_count"] != 1
        ):
            raise AssertionError(
                f"Expected one replacement submission, got {healed['submissions']!r}"
            )
        evidence["replacement_submission"] = healed["submissions"][-1]["submission_id"]
        evidence["surviving_sandbox"] = True
        evidence["replacement_sandbox"] = True

        for lease in leases:
            with contextlib.suppress(Exception):
                lease.close()
        leases.clear()
        wait_for(
            "idle workers to scale down",
            lambda: (
                status
                if (status := lease_client.pool(args.pool))["current_workers"] == 0
                else None
            ),
            timeout=args.timeout,
        )
        evidence["scaled_down_to_zero"] = True
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    finally:
        for lease in leases:
            with contextlib.suppress(Exception):
                lease.close()
        with contextlib.suppress(Exception):
            client.delete_pool(args.pool)


if __name__ == "__main__":
    raise SystemExit(main())
