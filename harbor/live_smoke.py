"""Bounded, CPU-only acceptance against an explicitly isolated controller.

Requires a manifest for tests/fixtures/basic/environment and Harbor 0.23.0.
Never invoke against a shared training controller: cancellation intentionally
interrupts only this script's last lease. The operator owns pool/job cleanup.
"""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from harbor.models.task.config import TaskConfig
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.trial.trial import Trial

from examples.network_smoke import listen, network_details, port_open, verify_namespaces
from sandfleet_harbor import SandfleetEnvironment


async def check(output: Path, https_url: str):
    fixture = Path(__file__).parent / "tests/fixtures/basic"
    config = TaskConfig.model_validate_toml((fixture / "task.toml").read_text())
    report = {"tests": [], "pools": []}

    def environment(name):
        return SandfleetEnvironment(
            environment_dir=fixture / "environment",
            environment_name="basic",
            session_id=name,
            trial_paths=TrialPaths(output / name),
            task_env_config=config.environment.model_copy(deep=True),
            acquire_timeout=600,
        )

    def passed(name):
        report["tests"].append(name)
        print(f"PASS {name}", flush=True)

    first = environment("first")
    try:
        await first.start()
        report["pools"].append(first._lease.pool)
        report["first_status"] = first._lease.status()
        report["first_network"] = await asyncio.to_thread(
            network_details, first._lease, https_url
        )
        verify_namespaces([report["first_status"]], [report["first_network"]])
        passed("disk-private-network-policy")
        value = await first.exec(
            "python3 -c \"import socket; addresses={x[4][0] for x in socket.getaddrinfo('localhost',80)}; assert addresses and addresses <= {'127.0.0.1','::1'}, addresses\""
        )
        assert value.return_code == 0, value
        passed("disk-localhost-resolution")
        value = await first.exec(
            "for path in /etc/resolv.conf /etc/hosts; do "
            'if sh -c \'printf bad >> "$1"\' _ "$path" 2>/dev/null; then exit 1; fi; done'
        )
        assert value.return_code == 0, value
        passed("disk-owned-network-files-read-only")
        value = await first.exec(
            "printf '%s' \"$SMOKE\"; printf warning >&2; exit 7",
            env={"SMOKE": "quoted ; text"},
        )
        assert (value.stdout, value.stderr, value.return_code) == (
            "quoted ; text",
            "warning",
            7,
        ), value
        passed("stdout-stderr-exit-env")
        value = await first.exec("sleep 20", timeout_sec=1)
        assert value.return_code == 124, value
        passed("command-timeout")
        source = output / "source"
        source.mkdir()
        data = os.urandom(5 * 262144 + 17)
        (source / "binary").write_bytes(data)
        (source / "empty").write_bytes(b"")
        await first.upload_dir(source, "/workspace/tree")
        await first.download_dir("/workspace/tree", output / "download")
        assert (output / "download/binary").read_bytes() == data
        assert (output / "download/empty").read_bytes() == b""
        report["transfer_sha256"] = hashlib.sha256(data).hexdigest()
        passed("chunked-directory-binary-empty-roundtrip")
        value = await first.exec(
            "touch /tmp/trial-marker /root/trial-marker /workspace/trial-marker"
        )
        assert value.return_code == 0
        await asyncio.to_thread(listen, first._lease, 38171, "first-trial-only")
    finally:
        await first.stop()

    second = environment("second")
    try:
        await second.start()
        report["second_status"] = second._lease.status()
        report["second_network"] = await asyncio.to_thread(
            network_details, second._lease, https_url
        )
        verify_namespaces([report["second_status"]], [report["second_network"]])
        assert not await asyncio.to_thread(port_open, second._lease, 38171)
        passed("fresh-network-old-listener-removed")
        assert (
            report["first_status"]["worker_pid"]
            == report["second_status"]["worker_pid"]
        )
        value = await second.exec(
            "test ! -e /tmp/trial-marker && test ! -e /root/trial-marker && test ! -e /workspace/trial-marker"
        )
        assert value.return_code == 0, value
        passed("fresh-disk-same-prepared-step")
    finally:
        await second.stop()

    trial = await Trial.create(
        TrialConfig.model_validate(
            {
                "task": {"path": str(fixture)},
                "trial_name": "oracle",
                "trials_dir": str(output),
                "agent": {"name": "oracle"},
                "environment": {"import_path": "sandfleet_harbor:SandfleetEnvironment"},
            }
        )
    )
    result = await trial.run()
    report["oracle"] = result.model_dump(mode="json")
    assert result.exception_info is None, result.exception_info
    assert result.verifier_result.rewards["reward"] == 1, result.verifier_result
    assert (output / "oracle/verifier/reward.txt").read_text().strip() == "1"
    assert (output / "oracle/agent/oracle.txt").exists()
    passed("harbor-oracle-verifier-artifacts-cleanup")

    cancelled = environment("cancelled")
    try:
        await cancelled.start()
        execution = asyncio.create_task(cancelled.exec("sleep 300", timeout_sec=360))
        await asyncio.sleep(2)
        started = time.monotonic()
        execution.cancel()
        try:
            await asyncio.wait_for(execution, 90)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Expected cancellation")
        report["cancel_cleanup_seconds"] = time.monotonic() - started
        assert (
            cancelled._lease is None
        ), "Cancelled lease cleanup retained; inspect before retry"
        passed("active-command-cancellation-and-release")
    finally:
        await cancelled.stop()
    report["complete"] = True
    (output / "result.json").write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated-controller", action="store_true", required=True)
    parser.add_argument("--https-url", default="https://example.com/")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    if not args.https_url.startswith("https://"):
        parser.error("--https-url must be an HTTPS endpoint reachable from workers")
    asyncio.run(check(args.output, args.https_url))


if __name__ == "__main__":
    main()
