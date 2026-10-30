"""Rate-friendly Slurm command adapter for Sandfleet worker allocations."""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sandfleet.audit import DEFAULT_AUDIT_MAX_MB
from sandfleet.egress import EgressSettings
from sandfleet.mounts import validate_phase_roots
from sandfleet.network import validate_network_config


QOS_GPU_TRES = re.compile(r"(?:^|[=,])gres/gpu=(?P<limit>[0-9]+)\((?P<used>[0-9]+)\)")
GPU_GRES = re.compile(
    r"(?:^|,)\s*gpu:(?P<type>[A-Za-z0-9][A-Za-z0-9_.-]*):"
    r"(?P<count>[0-9]+)(?:\([^)]*\))?(?=,|$)"
)


def _tres_counts(raw: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value = item.partition("=")
        if separator and value.isdigit():
            counts[name] = int(value)
    return counts


def _string_list(value: Any, separator: str) -> list[str]:
    """Slurm JSON fields may be a list or a separator-joined string."""

    if isinstance(value, str):
        return [item for item in value.split(separator) if item]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _typed_gpu_gres_counts(raw: str) -> dict[str, int]:
    """Parse typed whole-GPU counts from a Slurm Gres/GresUsed string."""

    counts: dict[str, int] = {}
    for match in GPU_GRES.finditer(raw):
        gpu_type = match.group("type")
        counts[gpu_type] = counts.get(gpu_type, 0) + int(match.group("count"))
    return counts


SLURM_TIME = re.compile(
    r"(?P<days>[0-9]+)-(?P<dh>[0-9]+)(?::(?P<dm>[0-9]+)(?::(?P<ds>[0-9]+))?)?"
    r"|(?P<a>[0-9]+)(?::(?P<b>[0-9]+)(?::(?P<c>[0-9]+))?)?"
)


def slurm_time_seconds(value: str) -> int | None:
    """Seconds in a Slurm ``--time`` value; None if unlimited or not parseable.

    Accepts the sbatch forms: minutes, minutes:seconds, hours:minutes:seconds,
    days-hours, days-hours:minutes and days-hours:minutes:seconds. A zero limit
    means no limit to Slurm, so it is None too.
    """

    match = SLURM_TIME.fullmatch(value.strip())
    if match is None:
        return None
    part = {key: int(item) for key, item in match.groupdict().items() if item}
    if "days" in part:
        hours = part["days"] * 24 + part["dh"]
        seconds = hours * 3600 + part.get("dm", 0) * 60 + part.get("ds", 0)
    elif "c" in part:
        seconds = part["a"] * 3600 + part["b"] * 60 + part["c"]
    else:
        seconds = part["a"] * 60 + part.get("b", 0)
    return seconds or None


@dataclass(frozen=True)
class SlurmSettings:
    partition: str
    account: str
    time_limit: str
    source_root: str
    agent_python: str
    log_dir: str
    apptainer_binary: str
    worker_python: str
    cache_dir: str | None = None
    tmp_dir: str | None = None
    start_concurrency: int = 2
    start_jitter_seconds: float = 5.0
    memory_headroom_mb: int = 512
    qos: str | None = None
    agent_advertise_mode: str = "fqdn"
    cpu_allocation_granularity: int = 1
    max_storage_mb: int = 0
    network_mode: str = "loopback"
    gpu_container_mode: str = "auto"
    controller_loss_timeout_seconds: int = 3600
    internet: EgressSettings = EgressSettings()
    # Per-worker JSONL audit files go here; None disables the audit log.
    audit_log_dir: str | None = None
    audit_log_max_mb: int = DEFAULT_AUDIT_MAX_MB
    # Host directories whose subdirectories leases may add as read-only mounts.
    readonly_mount_roots: tuple[str, ...] = ()
    # Host directories whose subdirectories leases may mount read-only at start.
    agent_readonly_mount_roots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_network_config(self.network_mode)
        validate_phase_roots(self.agent_readonly_mount_roots, self.readonly_mount_roots)
        if self.audit_log_max_mb < 1:
            raise ValueError("audit_log_max_mb must be positive")
        if self.cpu_allocation_granularity < 1:
            raise ValueError("cpu_allocation_granularity must be positive")
        if self.max_storage_mb < 0:
            raise ValueError("max_storage_mb must be nonnegative")


@dataclass(frozen=True)
class SlurmSubmission:
    """One homogeneous worker submission, optionally backed by a job array."""

    submission_id: str
    worker_count: int
    submitted_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class SlurmSnapshot:
    """One batched scheduler observation.

    ``states`` is grouped by array base job ID. A non-array job is represented
    by a group containing its ordinary job ID.
    """

    ok: bool
    states: dict[str, dict[str, str]]
    error: str | None = None


class SlurmLauncher:
    def __init__(
        self,
        settings: SlurmSettings,
        *,
        service_token_file: str,
        controller_url: str,
        registry_path: str | None = None,
    ):
        self.settings = settings
        self.controller_url = controller_url
        self.registry_path = registry_path
        self.service_token_file = service_token_file
        Path(settings.log_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_gpu_request(gpus: int, gpu_type: str | None) -> None:
        if gpus < 0 or (gpus == 0) != (gpu_type is None):
            raise ValueError("GPU count and type must be configured together")

    @staticmethod
    def _agent_gpu_arguments(gpus: int, gpu_type: str | None) -> list[str]:
        return [] if not gpus else ["--gpus", str(gpus), "--gpu-type", str(gpu_type)]

    @staticmethod
    def _worker_gpu_arguments(gpus: int, gpu_type: str | None, slots: int) -> list[str]:
        return [] if not gpus else [f"--gres=gpu:{gpu_type}:{slots * gpus}"]

    def _registry_export(self) -> str:
        if self.registry_path:
            return f"export SANDFLEET_REGISTRY={shlex.quote(self.registry_path)}; "
        return "unset SANDFLEET_REGISTRY; "

    def _agent_sandbox_policy_arguments(self) -> list[str]:
        """Audit, disk and read-only-mount policy flags for each worker agent."""
        arguments: list[str] = []
        if self.settings.audit_log_dir:
            arguments.extend(
                [
                    "--audit-log-dir",
                    self.settings.audit_log_dir,
                    "--audit-log-max-mb",
                    str(self.settings.audit_log_max_mb),
                ]
            )
        if self.settings.max_storage_mb:
            arguments.extend(["--max-storage-mb", str(self.settings.max_storage_mb)])
        for root in self.settings.readonly_mount_roots:
            arguments.extend(["--readonly-mount-root", root])
        for root in self.settings.agent_readonly_mount_roots:
            arguments.extend(["--agent-readonly-mount-root", root])
        return arguments

    def submit_worker_batch(
        self,
        *,
        pool: str,
        worker_count: int,
        slots: int,
        cpus: int,
        memory_mb: int,
        gpus: int = 0,
        gpu_type: str | None = None,
    ) -> SlurmSubmission:
        if worker_count < 1 or slots < 1:
            raise ValueError("worker_count and slots must be positive")
        self._validate_gpu_request(gpus, gpu_type)
        granularity = self.settings.cpu_allocation_granularity
        cpus_per_slot = ((cpus + granularity - 1) // granularity) * granularity
        total_cpus = slots * cpus_per_slot
        total_memory_mb = slots * memory_mb + self.settings.memory_headroom_mb
        agent_command = [
            self.settings.agent_python,
            "-m",
            "sandfleet.agent",
            "--controller-url",
            self.controller_url,
            "--pool",
            pool,
            "--slots",
            str(slots),
            "--cpus",
            str(cpus),
            "--memory-mb",
            str(memory_mb),
            "--advertise-mode",
            self.settings.agent_advertise_mode,
            "--apptainer-binary",
            self.settings.apptainer_binary,
            "--network-mode",
            self.settings.network_mode,
            "--gpu-container-mode",
            self.settings.gpu_container_mode,
            "--controller-loss-timeout-seconds",
            str(self.settings.controller_loss_timeout_seconds),
            "--worker-python",
            self.settings.worker_python,
            "--start-concurrency",
            str(self.settings.start_concurrency),
            "--start-jitter-seconds",
            str(self.settings.start_jitter_seconds),
        ]
        if self.settings.cache_dir:
            agent_command.extend(["--cache-dir", self.settings.cache_dir])
        if self.settings.tmp_dir:
            agent_command.extend(["--tmp-dir", self.settings.tmp_dir])
        agent_command.extend(self._agent_gpu_arguments(gpus, gpu_type))
        agent_command.extend(self.settings.internet.agent_arguments())
        agent_command.extend(self._agent_sandbox_policy_arguments())

        exported_pythonpath = self.settings.source_root
        existing_pythonpath = os.environ.get("PYTHONPATH")
        if existing_pythonpath:
            exported_pythonpath = f"{exported_pythonpath}:{existing_pythonpath}"
        # Export in the script: `--export=K=V` can trigger Slurm user-env
        # retrieval, which some sites disable. Controller proxies would break
        # agent callbacks from worker nodes.
        prologue = (
            "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy; "
            f"export SANDFLEET_TOKEN_FILE={shlex.quote(self.service_token_file)}; "
            f"export PYTHONPATH={shlex.quote(exported_pythonpath)}; "
            + self._registry_export()
        )
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=sandfleet-{pool}",
            f"--partition={self.settings.partition}",
            f"--account={self.settings.account}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={total_cpus}",
            f"--mem={total_memory_mb}M",
            f"--time={self.settings.time_limit}",
            f"--output={self.settings.log_dir}/agent-{'%A_%a' if worker_count > 1 else '%j'}.out",
            f"--chdir={self.settings.source_root}",
        ]
        if self.settings.qos:
            command.append(f"--qos={self.settings.qos}")
        command.extend(self._worker_gpu_arguments(gpus, gpu_type, slots))
        if worker_count > 1:
            command.append(f"--array=0-{worker_count - 1}")
        command.extend(["--wrap", prologue + "exec " + shlex.join(agent_command)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
        # --parsable may append a cluster name after a semicolon.
        submission_id = result.stdout.strip().split(";", 1)[0]
        return SlurmSubmission(submission_id=submission_id, worker_count=worker_count)

    @staticmethod
    def cancel(job_ids: list[str]) -> None:
        if not job_ids:
            return
        result = subprocess.run(
            ["scancel", *job_ids], capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(f"scancel failed: {result.stderr.strip()}")

    @staticmethod
    def snapshot(submission_ids: list[str]) -> SlurmSnapshot:
        if not submission_ids:
            return SlurmSnapshot(ok=True, states={})
        try:
            result = subprocess.run(
                [
                    "squeue",
                    "--array",
                    "--noheader",
                    "--jobs",
                    ",".join(submission_ids),
                    "--format=%F|%K|%i|%T",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            return SlurmSnapshot(
                ok=False, states={}, error="squeue timed out after 30 seconds"
            )
        if result.returncode != 0:
            return SlurmSnapshot(
                ok=False, states={}, error=result.stderr.strip() or "squeue failed"
            )
        grouped: dict[str, dict[str, str]] = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split("|", 3)]
            if len(parts) != 4:
                return SlurmSnapshot(
                    ok=False, states={}, error=f"Malformed squeue output: {line!r}"
                )
            array_job_id, _array_task_id, job_id, state = parts
            grouped.setdefault(array_job_id, {})[job_id] = state
        return SlurmSnapshot(ok=True, states=grouped)

    def discover_gpu_inventory(self, gpu_types: set[str]) -> dict[str, Any]:
        """Return one advisory, batched inventory for the configured partition."""

        result = subprocess.run(
            ["scontrol", "show", "nodes", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "GPU inventory discovery failed: "
                + (result.stderr.strip() or "scontrol failed")
            )
        try:
            payload = json.loads(result.stdout)
            nodes = payload["nodes"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"Malformed scontrol GPU inventory: {error}") from error

        unavailable_states = {
            "DOWN",
            "DRAIN",
            "DRAINED",
            "DRAINING",
            "FAIL",
            "FAILING",
            "FUTURE",
            "MAINTENANCE",
            "NOT_RESPONDING",
            "POWERED_DOWN",
            "REBOOT_ISSUED",
            "REBOOT_REQUESTED",
            "RESERVED",
            "UNKNOWN",
        }
        inventory = {
            gpu_type: {
                "configured": 0,
                "allocated": 0,
                "free_estimate": 0,
                "largest_node_free_estimate": 0,
                "schedulable_nodes": 0,
                "unavailable_nodes": 0,
            }
            for gpu_type in gpu_types
        }
        for node in nodes:
            if not isinstance(node, dict) or self.settings.partition not in (
                _string_list(node.get("partitions"), ",")
            ):
                continue
            states = set(_string_list(node.get("state"), "+"))
            schedulable = not bool(states & unavailable_states)
            configured_tres = _tres_counts(str(node.get("tres", "")))
            allocated_tres = _tres_counts(str(node.get("tres_used", "")))
            configured_gres = _typed_gpu_gres_counts(str(node.get("gres", "")))
            allocated_gres = _typed_gpu_gres_counts(str(node.get("gres_used", "")))
            for gpu_type, summary in inventory.items():
                key = f"gres/gpu:{gpu_type}"
                count = configured_gres.get(gpu_type, configured_tres.get(key, 0))
                if not count:
                    continue
                if gpu_type in allocated_gres:
                    used = allocated_gres[gpu_type]
                elif key in allocated_tres:
                    used = allocated_tres[key]
                else:
                    # An aggregate TRES count is attributable exactly on a
                    # homogeneous node. For mixed nodes, charge the aggregate
                    # against each type conservatively rather than inventing
                    # type-specific free capacity.
                    used = min(count, allocated_tres.get("gres/gpu", 0))
                summary["configured"] += count
                summary["allocated"] += used
                if schedulable:
                    node_free = max(0, count - used)
                    summary["free_estimate"] += node_free
                    summary["largest_node_free_estimate"] = max(
                        summary["largest_node_free_estimate"], node_free
                    )
                    summary["schedulable_nodes"] += 1
                else:
                    summary["unavailable_nodes"] += 1
        return inventory

    def discover_gpu_quota(self) -> dict[str, int] | None:
        """Return the controller QOS's aggregate GPU headroom when configured."""

        if self.settings.qos is None:
            return None
        result = subprocess.run(
            [
                "scontrol",
                "show",
                "assoc_mgr",
                f"qos={self.settings.qos}",
                "flags=qos",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "GPU QOS discovery failed: "
                + (result.stderr.strip() or "scontrol assoc_mgr failed")
            )
        qos_record = result.stdout.partition("Account Limits")[0]
        group_tres = re.search(r"(?:^|\s)GrpTRES=(\S*)", qos_record)
        match = QOS_GPU_TRES.search(group_tres.group(1)) if group_tres else None
        if match is None:
            return None
        limit = int(match.group("limit"))
        used = int(match.group("used"))
        return {
            "limit": limit,
            "allocated": used,
            "free_estimate": max(0, limit - used),
        }

    def probe_gpu_access(self, gpu_type: str) -> dict[str, Any]:
        """Ask Slurm to validate one minimal GPU job without creating it."""

        command = [
            "sbatch",
            "--test-only",
            "--job-name=sandfleet-gpu-access-probe",
            f"--partition={self.settings.partition}",
            f"--account={self.settings.account}",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--mem=64M",
            "--time=00:01:00",
            f"--gres=gpu:{gpu_type}:1",
            "--wrap",
            "true",
        ]
        if self.settings.qos:
            command.insert(6, f"--qos={self.settings.qos}")
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        detail = result.stdout.strip() or result.stderr.strip()
        return {"ok": result.returncode == 0, "detail": detail}
