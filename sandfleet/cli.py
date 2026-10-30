"""Command-line client for Sandfleet."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from sandfleet.client import SandfleetClient
from sandfleet.http import secret_from_env


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def default_token(role: str) -> str:
    """Choose the least-privileged configured token for one API role."""
    variable = {
        "admin": "SANDFLEET_TOKEN",
        "client": "SANDFLEET_CLIENT_TOKEN",
        "metrics": "SANDFLEET_METRICS_TOKEN",
    }[role]
    return secret_from_env(variable)


def health(client: SandfleetClient, _args: argparse.Namespace) -> int:
    print_json(client.health())
    return 0


def metrics(client: SandfleetClient, _args: argparse.Namespace) -> int:
    print_json(client.metrics())
    return 0


def gpus(client: SandfleetClient, _args: argparse.Namespace) -> int:
    print_json(client.gpu_capabilities())
    return 0


def pool_create(client: SandfleetClient, args: argparse.Namespace) -> int:
    print_json(
        client.create_pool(
            args.name,
            desired_capacity=args.capacity,
            min_capacity=args.min_capacity,
            max_capacity=args.max_capacity,
            warm_capacity=args.warm_capacity,
            slots_per_job=args.slots_per_job,
            cpus=args.cpus,
            memory_mb=args.memory_mb,
            idle_timeout_seconds=args.idle_timeout,
        )
    )
    return 0


def pool_status(client: SandfleetClient, args: argparse.Namespace) -> int:
    print_json(client.pool(args.name))
    return 0


def pool_wait(client: SandfleetClient, args: argparse.Namespace) -> int:
    print_json(
        client.wait_for_pool(args.name, capacity=args.capacity, timeout=args.timeout)
    )
    return 0


def pool_delete(client: SandfleetClient, args: argparse.Namespace) -> int:
    print_json(client.delete_pool(args.name))
    return 0


def pool_resume(client: SandfleetClient, args: argparse.Namespace) -> int:
    print_json(client.resume_pool(args.name))
    return 0


def _execute(
    client: SandfleetClient, args: argparse.Namespace, **selection: Any
) -> int:
    with client.acquire(
        timeout=args.acquire_timeout, internet=args.internet, **selection
    ) as lease:
        status = lease.start(args.image, command_timeout=args.timeout)
        result = lease.run_command(args.shell_command, timeout=args.timeout)
        print_json(
            {
                "status": status,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
            }
        )
        return result.exit_code


def execute(client: SandfleetClient, args: argparse.Namespace) -> int:
    return _execute(client, args, pool=args.pool)


def execute_resources(client: SandfleetClient, args: argparse.Namespace) -> int:
    return _execute(
        client,
        args,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        gpu=args.gpu,
        gpu_type=args.gpu_type,
    )


def add_exec_parsers(subparsers: argparse._SubParsersAction) -> None:
    execute_parser = subparsers.add_parser("exec")
    execute_parser.add_argument("pool")
    execute_parser.add_argument("image")
    execute_parser.add_argument("shell_command")
    execute_parser.add_argument("--timeout", type=int, default=1800)
    execute_parser.add_argument("--acquire-timeout", type=int, default=900)
    execute_parser.add_argument("--internet", action="store_true")
    execute_parser.set_defaults(handler=execute, token_role="client")

    resource_exec = subparsers.add_parser("exec-resources")
    resource_exec.add_argument("image")
    resource_exec.add_argument("shell_command")
    resource_exec.add_argument("--cpus", type=int, required=True)
    resource_exec.add_argument("--memory-mb", type=int, required=True)
    resource_exec.add_argument("--gpu", type=int)
    resource_exec.add_argument(
        "--type",
        dest="gpu_type",
        action="append",
        help="Exact memory-qualified GPU type; repeat for ordered alternatives",
    )
    resource_exec.add_argument("--timeout", type=int, default=1800)
    resource_exec.add_argument("--acquire-timeout", type=int, default=900)
    resource_exec.add_argument("--internet", action="store_true")
    resource_exec.set_defaults(handler=execute_resources, token_role="client")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("SANDFLEET_URL"))
    parser.add_argument("--token")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser("health")
    health_parser.set_defaults(handler=health, token_role="client")

    metrics_parser = subparsers.add_parser("metrics")
    metrics_parser.set_defaults(handler=metrics, token_role="metrics")

    gpus_parser = subparsers.add_parser("gpus")
    gpus_parser.set_defaults(handler=gpus, token_role="client")

    pool = subparsers.add_parser("pool")
    pool_subparsers = pool.add_subparsers(dest="pool_command", required=True)
    create = pool_subparsers.add_parser("create")
    create.add_argument("name")
    capacity = create.add_mutually_exclusive_group(required=True)
    capacity.add_argument("--capacity", type=int, help="Fixed capacity")
    capacity.add_argument("--max-capacity", type=int, help="Maximum elastic capacity")
    create.add_argument("--min-capacity", type=int, default=0)
    create.add_argument("--warm-capacity", type=int)
    create.add_argument("--slots-per-job", type=int, required=True)
    create.add_argument("--cpus", type=int, default=1)
    create.add_argument("--memory-mb", type=int, default=4096)
    create.add_argument(
        "--idle-timeout",
        type=int,
        default=600,
        help="Seconds before an idle worker scales down",
    )
    create.set_defaults(handler=pool_create, token_role="admin")
    status = pool_subparsers.add_parser("status")
    status.add_argument("name")
    status.set_defaults(handler=pool_status, token_role="client")
    wait = pool_subparsers.add_parser("wait")
    wait.add_argument("name")
    wait.add_argument("--capacity", type=int)
    wait.add_argument("--timeout", type=float, default=600)
    wait.set_defaults(handler=pool_wait, token_role="client")
    delete = pool_subparsers.add_parser("delete")
    delete.add_argument("name")
    delete.set_defaults(handler=pool_delete, token_role="admin")
    resume = pool_subparsers.add_parser("resume")
    resume.add_argument("name")
    resume.set_defaults(handler=pool_resume, token_role="admin")

    add_exec_parsers(subparsers)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "exec-resources" and (
        (args.gpu is None) != (args.gpu_type is None)
    ):
        parser.error("--gpu and at least one --type must be set together")

    if args.token is None:
        try:
            args.token = default_token(args.token_role)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    if not args.url or not args.token:
        parser.error("Sandfleet URL and an appropriate role token are required")
    return args.handler(SandfleetClient(args.url, args.token), args)


if __name__ == "__main__":
    raise SystemExit(main())
