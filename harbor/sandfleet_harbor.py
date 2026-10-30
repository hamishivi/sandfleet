"""Harbor 0.23.0 environment using canonical Sandfleet leases and disk overlays.

Import with --environment-import-path sandfleet_harbor:SandfleetEnvironment.
Add this directory and the Sandfleet source root to PYTHONPATH. Credentials are
read only from the client-role environment; images require a pinned manifest.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import threading
import uuid

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import TaskConfig
from harbor.models.task.verifier_mode import (
    VerifierEnvironmentMode,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import ResourceMode
from sandfleet.client import SandboxLease, SandfleetClient, validate_disk_status
from sandfleet.errors import SandfleetError
from sandfleet.http import secret_from_env
from sandfleet.mounts import (
    check_protected_paths,
    check_target_overlap,
    is_within,
    validate_mount_request,
)
from sandfleet.resources import (
    GPU_MODEL_NAME,
    MAX_GPU_TYPE_FALLBACKS,
    GPURequest,
    SandboxResources,
)

from image_manifest import resolve_image

CHUNK_BYTES = 262144
REQUIRED_FEATURES = {
    "disk-overlay-v1",
    "file-chunks-v1",
    "lease-abort-v1",
    "command-cancel-v1",
    "command-env-v1",
}
# A task.toml declares read-only host directories per phase:
#   [metadata]
#   agent_mounts = [{ source = "/host/assets/data", target = "/assets/data" }]
#   verifier_mounts = [{ source = "/host/models/x", target = "/verifier/models/x" }]
# agent_mounts are mounted when the sandbox starts (Sandfleet 0.14); verifier_mounts
# appear only when the trial calls add_verifier_mounts() after its agent phase
# (Sandfleet 0.13). Each phase's sources must be under that phase's controller roots.
#
# SEPARATE verifier mode ([verifier] environment_mode = "separate"): the verifier
# runs in its own fresh environment (Harbor names its session
# "<trial>__verifier__<key>"), which adds verifier_mounts right after it starts
# (same add-mounts mechanism and roots, so verifier-only data never needs an agent
# root); there a verifier_mounts target may be /tests (Harbor uploads no tests to a
# separate verifier), so one shared verifier image can serve every task: set
# [verifier.environment] docker_image to a name in the manifest's "named" images.
# produce_mounts are added to the AGENT sandbox after its agent phase (add-mounts,
# verifier roots), for a pipeline-owned producer that runs the agent's deliverable
# on inputs before its outputs are collected as artifacts.
AGENT_MOUNTS_KEY = "agent_mounts"
VERIFIER_MOUNTS_KEY = "verifier_mounts"
PRODUCE_MOUNTS_KEY = "produce_mounts"
VERIFIER_ENV_MARKER = "__verifier__"
AGENT_MOUNTS_FEATURE = "lease-start-readonly-mounts-v1"
VERIFIER_MOUNTS_FEATURE = "lease-readonly-mounts-v1"
# GPUs use Harbor's own task.toml fields:
#   [environment]
#   gpus = 1
#   gpu_types = ["gb300_288gb"]   # optional; exact Sandfleet types or families ("GB300")
#   [metadata]
#   exclusive_gpu = true          # optional; nothing else may use the task's GPU
# Every Sandfleet GPU is a whole device of the lease's own Slurm step, never
# shared with another sandbox, so exclusive_gpu always holds here; it is
# validated so a task can declare it and an environment that cannot honour it
# can refuse.
EXCLUSIVE_GPU_KEY = "exclusive_gpu"
# Harbor's own directories, which a read-only mount must not cover.
HARBOR_WRITABLE_PATHS = ["/logs/agent", "/logs/verifier", "/logs/artifacts", "/tests"]


def declared_mounts(
    task: TaskConfig, key: str, protected: list[str] = HARBOR_WRITABLE_PATHS
) -> list[dict[str, str]]:
    """The task's mounts under ``[metadata] key`` ([] when it declares none)."""
    declared = task.metadata.get(key)
    if declared is None:
        return []
    mounts = validate_mount_request(declared)
    for item in mounts:
        if any(is_within(item["target"], path) for path in protected):
            raise ValueError(
                f"{key} target {item['target']!r} is inside a directory "
                "Harbor writes or resets"
            )
    check_protected_paths([item["target"] for item in mounts], protected)
    return mounts


def verifier_mounts(task: TaskConfig) -> list[dict[str, str]]:
    """The task's declared verifier-only mounts ([] when it declares none). A
    separate verifier environment gets no tests upload, so /tests may be one."""
    if resolve_task_verifier_mode(task) == VerifierEnvironmentMode.SEPARATE:
        return declared_mounts(
            task,
            VERIFIER_MOUNTS_KEY,
            [p for p in HARBOR_WRITABLE_PATHS if p != "/tests"],
        )
    return declared_mounts(task, VERIFIER_MOUNTS_KEY)


def produce_mounts(task: TaskConfig) -> list[dict[str, str]]:
    """The task's producer inputs ([] when none): added to the agent sandbox after
    its agent phase; SEPARATE verifier mode only."""
    return declared_mounts(task, PRODUCE_MOUNTS_KEY)


def agent_mounts(task: TaskConfig) -> list[dict[str, str]]:
    """The task's declared start-time (agent-visible) mounts ([] when none)."""
    return declared_mounts(task, AGENT_MOUNTS_KEY)


def check_source_separation(
    agent_visible: list[dict[str, str]], verifier: list[dict[str, str]]
) -> None:
    """Refuse a task whose agent or producer mount source is, contains or lies inside
    a verifier mount source (compared after resolving host symlinks): that would
    show verifier-only data (e.g. answers) to the agent or its deliverable. Producer
    mounts come from the verifier roots, so the roots do not keep them apart."""

    def real(item):
        return os.path.realpath(item["source"])

    for shown in agent_visible:
        for hidden in verifier:
            a, b = real(shown), real(hidden)
            if is_within(a, b) or is_within(b, a):
                raise ValueError(
                    f"Agent-visible mount source {shown['source']!r} overlaps the "
                    f"verifier mount source {hidden['source']!r}"
                )


def check_tests_mount(mounts: list[dict[str, str]]) -> None:
    """Refuse a separate verifier's /tests mount whose test.sh is not executable:
    Harbor runs it directly, and its chmod cannot change a read-only mount."""
    for item in mounts:
        script = Path(item["source"]) / "test.sh"
        if item["target"] == "/tests" and script.is_file():
            if not script.stat().st_mode & 0o111:
                raise ValueError(
                    f"{script} must be executable: Harbor runs /tests/test.sh "
                    "directly and cannot chmod the read-only /tests mount"
                )


def transfer_failure_is_infra(error: BaseException) -> bool:
    """Whether a failed download from a sandbox is the infrastructure's fault (the
    lease, its transport or the host disk), not the sandbox content's (a missing
    path, a symlink or special file this adapter refuses, a failing inventory
    command). Sandfleet answers 400 for a missing or unreadable path."""
    if isinstance(error, SandfleetError):
        return error.status != 400
    return not isinstance(error, (ValueError, RuntimeError))


def exclusive_gpu(task: TaskConfig) -> bool:
    """Whether the task declares ``[metadata] exclusive_gpu = true``."""
    value = task.metadata.get(EXCLUSIVE_GPU_KEY, False)
    if not isinstance(value, bool):
        raise ValueError(f"{EXCLUSIVE_GPU_KEY} must be true or false, got {value!r}")
    return value


def resolve_gpu_types(requested: list[str] | None, enabled: list[str]) -> list[str]:
    """Map a task's Harbor ``gpu_types`` onto the controller's exact GPU types.

    Sandfleet types are lowercase memory-qualified names (``gb300_288gb``);
    Harbor tasks usually name a family (``GB300``, ``H100``). An entry that is
    an enabled exact type selects it; any other entry selects every enabled type
    in that family (``<family>_<memory>gb``). Order is kept, so earlier entries
    are preferred. ``None`` means any enabled type. Raises if nothing matches.
    """
    if not enabled:
        raise RuntimeError("The Sandfleet controller enables no GPU types")
    if requested is None:
        return sorted(enabled)[:MAX_GPU_TYPE_FALLBACKS]
    selected: list[str] = []
    for name in requested:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"gpu_types entries must be non-empty strings: {name!r}")
        key = name.strip().lower()
        if key in enabled:
            matches = [key]
        elif GPU_MODEL_NAME.fullmatch(key) is not None:
            matches = []  # an exact type the controller does not enable
        else:
            matches = sorted(item for item in enabled if item.startswith(f"{key}_"))
        selected += [item for item in matches if item not in selected]
    if not selected:
        raise RuntimeError(
            f"None of the task's gpu_types {requested!r} is enabled by the Sandfleet "
            f"controller (enabled: {sorted(enabled)!r})"
        )
    return selected[:MAX_GPU_TYPE_FALLBACKS]


def configuration() -> tuple[str, str, Path]:
    if importlib.metadata.version("harbor") != "0.23.0":
        raise ValueError("Revalidate the backend before changing Harbor 0.23.0")
    url = os.environ["SANDFLEET_URL"]
    token = secret_from_env("SANDFLEET_CLIENT_TOKEN")
    manifest = os.environ["SANDFLEET_HARBOR_IMAGE_MANIFEST"]
    if not url or not token or not manifest:
        raise ValueError(
            "SANDFLEET_URL, client token, and SANDFLEET_HARBOR_IMAGE_MANIFEST are required"
        )
    return url, token, Path(manifest)


def reported_end_time(status: dict) -> float | None:
    """The worker's optional ``worker_end_time``; None if absent or malformed."""
    value = status.get("worker_end_time")
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        value = float(value)
    except OverflowError:  # an int too large for a float
        return None
    return value if math.isfinite(value) and value > 0 else None


async def finish(task: asyncio.Task):
    """Drain a known cleanup/worker task even if cancellation is repeated."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class SandfleetEnvironment(BaseEnvironment):
    # Read-only mount phases this environment implements (see AGENT_MOUNTS_KEY).
    supports_agent_mounts = True
    supports_verifier_mounts = True
    supports_produce_mounts = True
    # GPUs ([environment] gpus; capabilities.gpus) are whole devices dedicated to
    # the lease's Slurm step, so [metadata] exclusive_gpu always holds.
    supports_exclusive_gpus = True

    @staticmethod
    def type() -> str:
        return "sandfleet"

    @property
    def is_verifier_env(self) -> bool:
        """Whether this is a SEPARATE verifier environment (not an agent sandbox).
        Harbor builds a separate verifier from the task's tests/ directory and the
        agent sandbox from environment/. The session id is not used: Harbor
        truncates long ones (dropping its "__verifier__" marker), and a trial name
        could contain the marker."""
        return self.environment_dir.name == "tests"

    @classmethod
    def preflight(cls) -> None:
        configuration()

    def __init__(
        self,
        *args,
        acquire_timeout: float = 900,
        internet: bool | None = None,
        command_timeout_sec: float | None = None,
        **kwargs,
    ):
        url, token, self._manifest = configuration()
        self._client = SandfleetClient(url, token, timeout=30)
        if not math.isfinite(acquire_timeout) or acquire_timeout <= 0:
            raise ValueError("acquire_timeout must be positive and finite")
        if internet is not None and not isinstance(internet, bool):
            raise ValueError("internet must be true, false or unset")
        if command_timeout_sec is not None and not (
            # bool is an int subclass; True must not become a 1 s timeout.
            isinstance(command_timeout_sec, int | float)
            and not isinstance(command_timeout_sec, bool)
            and math.isfinite(command_timeout_sec)
            and command_timeout_sec > 0
        ):
            raise ValueError("command_timeout_sec must be a positive, finite number")
        self._acquire_timeout = acquire_timeout
        # None follows the task: public tasks get internet when the controller offers it.
        self._internet = internet
        self._command_timeout_override = command_timeout_sec
        self._lease = None
        self._agent_mounts: list[dict[str, str]] = []
        self._verifier_mounts: list[dict[str, str]] = []
        self._verifier_mount_result = None
        self._produce_mounts: list[dict[str, str]] = []
        self._produce_mount_result = None
        # Downloads (artifacts, logs) that failed for an infrastructure reason
        # (transfer_failure_is_infra), so a caller can mask a trial whose separate
        # verifier would otherwise score incomplete artifacts.
        self.transfer_infra_failures: list[str] = []
        self._gpu_types: list[str] = []
        self._gpu_uuids: list[str] = []
        self._gpu_type = None
        self._gpu_evidence = None
        # When the lease's worker job ends (Unix seconds), as its worker reports
        # at start (Sandfleet >= 0.16); None if unknown. Informational: callers
        # may warn when a trial needs longer. Leases are not placed by it.
        self.worker_end_time: float | None = None
        self._lock = asyncio.Lock()
        self._cancel = threading.Event()
        self._ready = False
        super().__init__(*args, **kwargs)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Offline leases have no network at all (Sandfleet >= 0.9: --network=none,
        # loopback only), so no-network tasks are enforced, not approximated.
        # Allowlist policies are not implemented and stay refused.
        return EnvironmentCapabilities(gpus=True, disable_internet=True)

    @property
    def gpu_count(self) -> int:
        """The task's GPU count (``[environment] gpus``; 0 without GPUs)."""
        return self.task_env_config.gpus or 0

    def _maybe_override_task_env_config(self):
        # Harbor binds positional and keyword arguments before invoking this hook.
        if any(
            value is not None
            for value in (
                self._override_cpus,
                self._override_memory_mb,
                self._override_storage_mb,
                self._override_gpus,
                self._override_tpu,
            )
        ):
            raise ValueError(
                "Resource overrides are forbidden for this evaluation backend"
            )
        exact = {ResourceMode.AUTO, ResourceMode.GUARANTEE}
        if {self._cpu_resource_mode, self._memory_resource_mode} - exact:
            raise ValueError("Sandfleet always guarantees the exact task CPU/RAM")

    def _validate_definition(self):
        cfg = self.task_env_config
        if any(
            type(value) is not int or value <= 0
            for value in (cfg.cpus, cfg.memory_mb, cfg.storage_mb)
        ):
            raise ValueError(
                "Positive integer CPU, RAM and disk resources are required"
            )
        if cfg.storage_mb < 64:
            raise ValueError("Disk overlays require at least 64 MiB")
        if cfg.gpus is not None and (type(cfg.gpus) is not int or cfg.gpus < 0):
            raise ValueError("[environment] gpus must be a non-negative integer")
        if cfg.gpu_types is not None and not self.gpu_count:
            raise ValueError("[environment] gpu_types needs gpus >= 1")
        if any(
            (self.environment_dir / name).exists()
            for name in ("docker-compose.yaml", "docker-compose.yml")
        ):
            raise ValueError("Multi-container Compose tasks are not supported")
        named = cfg.docker_image if self.is_verifier_env else None
        self._image = resolve_image(self._manifest, self.environment_dir, name=named)
        task = TaskConfig.model_validate_toml(
            (self.environment_dir.parent / "task.toml").read_text()
        )
        self._agent_mounts = agent_mounts(task)
        self._verifier_mounts = verifier_mounts(task)
        self._produce_mounts = produce_mounts(task)
        check_source_separation(
            self._agent_mounts + self._produce_mounts, self._verifier_mounts
        )
        # Each environment keeps only its own role's mounts.
        separate = resolve_task_verifier_mode(task) == VerifierEnvironmentMode.SEPARATE
        if VERIFIER_ENV_MARKER in (self.session_id or "") and not self.is_verifier_env:
            raise ValueError(
                f"Session {self.session_id!r} looks like a separate verifier's but "
                "the environment is not built from the task's tests/"
            )
        if self.is_verifier_env and not separate:
            raise ValueError("A verifier environment needs a separate verifier task")
        if not separate:
            if self._produce_mounts:
                raise ValueError(f"{PRODUCE_MOUNTS_KEY} need a separate verifier")
        elif self.is_verifier_env:
            self._agent_mounts, self._produce_mounts = [], []
            check_tests_mount(self._verifier_mounts)
        else:
            self._verifier_mounts = []
        if exclusive_gpu(task) and not self.gpu_count:
            raise ValueError(
                f"[metadata] {EXCLUSIVE_GPU_KEY} needs [environment] gpus >= 1"
            )
        check_target_overlap(
            [
                item["target"]
                for item in self._agent_mounts
                + self._verifier_mounts
                + self._produce_mounts
            ]
        )
        phases = [task, *(task.steps or [])]
        # Default deadline for commands Harbor sends without a timeout (e.g. agent
        # install steps). Trial-level overrides are not visible here, so
        # command_timeout_sec lets callers raise it to match them.
        self._command_timeout = math.ceil(
            self._command_timeout_override
            or max(
                timeout
                for phase in phases
                for timeout in (phase.agent.timeout_sec, phase.verifier.timeout_sec)
                if timeout is not None
            )
        )
        self._cwd = cfg.workdir or self._image["workdir"]
        if not self._cwd.startswith("/") or "\n" in self._cwd or "\x00" in self._cwd:
            raise ValueError("An absolute single-line working directory is required")

    def _check_cancelled(self):
        if self._cancel.is_set():
            raise InterruptedError("Harbor operation cancelled")

    async def _call(self, operation, *args, command_id=None):
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(operation, *args))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if command_id is not None and self._lease is not None:
                    try:
                        response = await finish(
                            asyncio.create_task(
                                asyncio.to_thread(
                                    self._lease.cancel_command, command_id
                                )
                            )
                        )
                        if response.get("error") or (
                            response["state"] == "running"
                            and not response["interrupted"]
                        ):
                            raise RuntimeError(
                                f"Command was not interrupted: {response}"
                            )
                        if response["state"] not in {
                            "pending",
                            "running",
                            "already_finished",
                        }:
                            raise RuntimeError(
                                f"Invalid cancellation state: {response}"
                            )
                        await finish(task)
                    except Exception as cancel_error:
                        error.add_note(f"Command cancellation failed: {cancel_error}")
                    else:
                        # Harbor still verifies after an agent timeout. Keep the
                        # original filesystem and lease only after a clean stop.
                        raise
                await self._drain_cancelled(task, error)
                raise

    async def _drain_cancelled(self, task, error):
        self._cancel.set()
        if self._lease is not None:
            try:
                await finish(asyncio.create_task(asyncio.to_thread(self._lease.abort)))
            except Exception as abort_error:
                error.add_note(f"Lease interrupt failed: {abort_error}")
        try:
            await finish(task)
        except Exception as operation_error:
            error.add_note(f"Cancelled operation drained: {operation_error}")
        try:
            await finish(asyncio.create_task(asyncio.to_thread(self._stop_sync)))
        except Exception as cleanup_error:
            error.add_note(f"Lease retained after cleanup failure: {cleanup_error}")

    async def start(self, force_build: bool = False) -> None:
        if force_build:
            raise ValueError(
                "Build task images into the pinned manifest before starting trials"
            )
        await self._call(self._start_sync)
        if not self.is_verifier_env:  # its tests come from a read-only mount
            await self._upload_environment_dir_after_start()

    def _start_sync(self):
        if self._lease is not None:
            raise RuntimeError("Environment already holds a lease")
        # Never clear a cancellation delivered before this thread was scheduled.
        self._check_cancelled()
        cfg = self.task_env_config
        health = self._client.health()
        if not health["ok"] or not REQUIRED_FEATURES.issubset(health["features"]):
            raise RuntimeError(
                "Controller does not advertise the required Harbor backend protocol"
            )
        if cfg.storage_mb > health["max_storage_mb"]:
            raise ValueError("Task storage exceeds the operator's enabled disk policy")
        self._check_mount_policy(health)
        offered = (
            "lease-internet-v1" in health["features"]
            # lease-internet-v1 and the health "internet" mode ship together.
            and health["internet"] != "off"
        )
        if self._internet and not offered:
            raise RuntimeError(
                "internet=true but the controller does not offer sandbox internet"
            )
        # A lease's network is fixed when it starts; Harbor's Trial refuses phase
        # policies that differ from the baseline, and so does a direct caller here.
        if any(
            policy != self._network_policy for policy in self._phase_network_policies
        ):
            raise ValueError(
                "Sandfleet cannot change a sandbox's network policy between phases"
            )
        if self._internet and self._network_disabled:
            raise ValueError("internet=true contradicts the task's no-network policy")
        internet = (
            offered and self._network_is_public
            if self._internet is None
            else self._internet
        )
        gpu_request = self._gpu_request()
        self._check_cancelled()
        self._lease = self._client.acquire(
            cpus=cfg.cpus,
            memory_mb=cfg.memory_mb,
            timeout=self._acquire_timeout,
            cancel_event=self._cancel,
            internet=internet,
            **gpu_request,
        )
        try:
            self._check_cancelled()
            status = self._lease.start(
                self._image["sif_path"],
                command_timeout=self._command_timeout,
                pwd=self._cwd,
                storage_mb=cfg.storage_mb,
                image_sha256=self._image["sif_sha256"],
                readonly_mounts=self._agent_mounts,
            )
            self._verify_status(status)
            self.worker_end_time = reported_end_time(status)
            self._gpu_uuids = self._verify_gpus(status)
            if (self._verifier_mounts or self._produce_mounts) and not status.get(
                "readonly_mount_roots"
            ):
                raise RuntimeError(
                    "The lease's worker cannot add verifier/producer mounts (it "
                    "predates Sandfleet 0.13 or has no read-only mount roots)"
                )
            self._check_cancelled()
            self._ready = True
            self._checked("mkdir -p /logs/agent /logs/verifier /logs/artifacts")
            if self.gpu_count:
                self._check_gpus_visible(status)
            if self.is_verifier_env and self._verifier_mounts:
                self._verifier_mount_result = self._add_verifier_mounts_sync()
            self._write_evidence(status)
        except BaseException as error:
            try:
                self._stop_sync()
            except Exception as cleanup_error:
                error.add_note(f"Lease cleanup failed: {cleanup_error}")
            raise

    def _gpu_request(self) -> dict:
        """The acquire() GPU arguments; fails before queueing for a lease if the
        controller cannot serve the task's GPUs."""
        if not self.gpu_count:
            return {}
        capabilities = self._client.gpu_capabilities()
        if not capabilities["enabled"]:
            raise RuntimeError(
                f"The task needs {self.gpu_count} GPU(s), but the Sandfleet controller has "
                "no GPU sandboxes configured (no operator GPU type catalog / GPU pool)"
            )
        types = capabilities["types"]
        enabled = [
            name
            for name, spec in types.items()
            if spec["max_per_sandbox"] >= self.gpu_count
        ]
        if not enabled:
            limits = {name: spec["max_per_sandbox"] for name, spec in types.items()}
            raise RuntimeError(
                f"The task needs {self.gpu_count} GPUs per sandbox; the Sandfleet "
                f"controller allows at most {limits!r}"
            )
        self._gpu_types = resolve_gpu_types(self.task_env_config.gpu_types, enabled)
        return {"gpu": self.gpu_count, "gpu_type": self._gpu_types}

    def _verify_gpus(self, status) -> list[str]:
        """The lease's GPU UUIDs, checked against the task's request.

        The worker probes the devices its Slurm step can open; Sandfleet slots
        own disjoint whole devices, so these GPUs belong to this sandbox alone.
        """
        if not self.gpu_count:
            return []
        gpu = status["cgroup"]["gpu"]
        identities = gpu["identities"]
        if (
            status["gpus"] != self.gpu_count
            or len(gpu["accessible"]) != self.gpu_count
            or len(identities) != self.gpu_count
        ):
            raise RuntimeError(
                f"The lease does not hold exactly the task's {self.gpu_count} GPU(s): "
                f"gpus={status['gpus']}, devices={len(gpu['accessible'])}, "
                f"identities={len(identities)}"
            )
        # status["gpu_type"] is the private Slurm GRES type; the public type is
        # the one the controller resolved the request to, which names its pool.
        cfg = self.task_env_config
        accepted = {
            SandboxResources(
                cfg.cpus, cfg.memory_mb, GPURequest(self.gpu_count, (gpu_type,))
            ).pool_name: gpu_type
            for gpu_type in self._gpu_types
        }
        self._gpu_type = accepted.get(self._require_lease_object().pool)
        if self._gpu_type is None:
            raise RuntimeError(
                f"The lease's pool {self._require_lease_object().pool!r} does not "
                f"serve one of the task's GPU types ({self._gpu_types!r})"
            )
        uuids = sorted(item["uuid"] for item in identities)
        if len(set(uuids)) != self.gpu_count:
            raise RuntimeError(f"The lease reports duplicate GPU identities {uuids}")
        return uuids

    def _check_gpus_visible(self, status):
        """Before the agent runs: the sandbox sees exactly the lease's GPUs, and no
        IMEX channel (multi-node NVLink memory sharing on GB200/GB300 NVL72): the
        worker's Slurm step can open the host's channels, so only Apptainer's
        contained /dev keeps them out of the sandbox."""
        result = self._run_sync(
            "nvidia-smi --query-gpu=uuid --format=csv,noheader; status=$?; "
            "ls -A /dev/nvidia-caps-imex-channels 2>/dev/null | sed 's/^/imex: /'; "
            'exit "$status"',
            user="root",
            timeout_sec=120,
            strict_output=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        imex = [line for line in lines if line.startswith("imex: ")]
        visible = sorted(line for line in lines if not line.startswith("imex: "))
        if result.return_code or visible != self._gpu_uuids:
            raise RuntimeError(
                "The sandbox does not see exactly its leased GPU(s): "
                f"expected {self._gpu_uuids}, nvidia-smi exit {result.return_code} "
                f"listed {visible} ({result.stderr.strip()[:500]})"
            )
        if imex:
            raise RuntimeError(f"The sandbox can see IMEX channels: {imex}")
        self._gpu_evidence = {
            "count": self.gpu_count,
            "type": self._gpu_type,
            "slurm_type": status["gpu_type"],
            "uuids": self._gpu_uuids,
            "names": sorted(
                {item["name"] for item in status["cgroup"]["gpu"]["identities"]}
            ),
            "cuda_visible_devices": status["cgroup"]["gpu"]["cuda_visible_devices"],
        }

    def _check_mount_policy(self, health):
        """Fail before acquiring a lease if the controller cannot mount a phase's
        sources: each phase is checked against its own roots only."""
        for key, mounts, feature, roots_field, version in (
            (
                AGENT_MOUNTS_KEY,
                self._agent_mounts,
                AGENT_MOUNTS_FEATURE,
                "agent_readonly_mount_roots",
                "0.14",
            ),
            (
                VERIFIER_MOUNTS_KEY,
                self._verifier_mounts,
                VERIFIER_MOUNTS_FEATURE,
                "readonly_mount_roots",
                "0.13",
            ),
            (
                PRODUCE_MOUNTS_KEY,
                self._produce_mounts,
                VERIFIER_MOUNTS_FEATURE,
                "readonly_mount_roots",
                "0.13",
            ),
        ):
            if not mounts:
                continue
            if feature not in health["features"]:
                raise RuntimeError(
                    f"The task declares {key} but the controller does not "
                    f"advertise {feature} (Sandfleet {version})"
                )
            roots = health[roots_field]
            for item in mounts:
                if not any(is_within(item["source"], root) for root in roots):
                    raise ValueError(
                        f"{key} source {item['source']!r} is not under one of the "
                        f"controller's {roots_field} {roots!r}"
                    )

    @property
    def verifier_mounts(self) -> list[dict[str, str]]:
        """The task's declared verifier-only mounts (``[metadata] verifier_mounts``)."""
        return [dict(item) for item in self._verifier_mounts]

    async def add_verifier_mounts(self):
        """Mount the task's verifier-only host directories read-only; None if it has none.

        Call after the agent phase and before the verifier, never earlier: the
        agent must not see them. Needs Sandfleet 0.13 workers. The worker kills
        every sandbox process and restarts the sandbox from its disk overlay with
        the mounts, so the agent's files persist but anything it left running
        (including servers a verifier might connect to) is stopped. Repeated
        calls return the first result. If it raises, do not run the verifier;
        the sandbox may be gone.
        """
        if not self._verifier_mounts:
            return None
        if self._verifier_mount_result is None:
            self._verifier_mount_result = await self._call(
                self._add_verifier_mounts_sync
            )
        return self._verifier_mount_result

    async def add_produce_mounts(self):
        """Mount the task's producer inputs read-only in the agent sandbox; None if
        it has none. Call after the agent phase, before the producer runs; same
        restart-from-overlay mechanism (and process kill) as add_verifier_mounts.
        Repeated calls return the first result."""
        if not self._produce_mounts:
            return None
        if self._produce_mount_result is None:
            self._produce_mount_result = await self._call(
                self._add_readonly_mounts_sync, self._produce_mounts, "producer"
            )
        return self._produce_mount_result

    def _add_verifier_mounts_sync(self):
        return self._add_readonly_mounts_sync(self._verifier_mounts, "verifier")

    def _add_readonly_mounts_sync(self, mounts, label):
        self._check_cancelled()
        result = self._require_lease().add_readonly_mounts(mounts)
        mounted = {item["target"] for item in result["readonly_mounts"]}
        missing = [item["target"] for item in mounts if item["target"] not in mounted]
        if missing:
            raise RuntimeError(f"Sandfleet did not report {label} mounts at {missing}")
        self._verify_status(result)
        # The restart reuses the lease's Slurm step, so it must keep the same GPUs.
        uuids = self._verify_gpus(result)
        if uuids != self._gpu_uuids:
            raise RuntimeError(
                f"The sandbox restart changed its GPUs: {self._gpu_uuids} -> {uuids}"
            )
        return {
            "added_mounts": result["added_mounts"],
            "stopped_processes": result["stopped_processes"],
        }

    def _verify_status(self, status):
        cfg = self.task_env_config
        if self._network_disabled:
            # The worker's own report: no relay, loopback only (Sandfleet >= 0.11).
            network = status.get("network")
            if (
                not isinstance(network, dict)
                or network.get("internet") is not False
                or network.get("mode") != "none"
            ):
                raise RuntimeError(
                    "The task is no-network, but the worker does not report the "
                    f"sandbox as offline (network={network!r})"
                )
        if status["cpus"] != cfg.cpus or status["memory_mb"] != cfg.memory_mb:
            raise RuntimeError("Assigned lease does not match the task CPU/RAM request")
        if int(status["cgroup"]["memory.max.effective"]) != cfg.memory_mb * 1048576:
            raise RuntimeError("Effective RAM limit does not match the task")
        validate_disk_status(
            status["storage"],
            storage_mb=cfg.storage_mb,
            image_sha256=self._image["sif_sha256"],
        )

    def _write_evidence(self, status):
        # A separate verifier shares the trial directory with the agent sandbox.
        name = (
            "sandfleet-verifier-environment.json"
            if self.is_verifier_env
            else "sandfleet-environment.json"
        )
        path = self.trial_paths.trial_dir / name
        evidence = {
            "image": self._image,
            "resources": {"cpus": status["cpus"], "memory_mb": status["memory_mb"]},
            "storage": status["storage"],
            "cgroup": status["cgroup"],
            "worker_pid": status["worker_pid"],
        }
        if self._gpu_evidence is not None:
            evidence["gpus"] = self._gpu_evidence
        # No lease/controller tokens or inherited environment are serialized.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as target:
            os.chmod(path, 0o600)
            json.dump(evidence, target, indent=2)

    async def stop(self, delete: bool = True):
        await self._call(self._stop_sync)

    def _stop_sync(self):
        self._ready = False
        self.worker_end_time = None
        if self._lease is not None:
            self._lease.close()
            self._lease = None

    def _require_lease_object(self) -> SandboxLease:
        if self._lease is None:
            raise RuntimeError("Environment holds no lease")
        return self._lease

    def _require_lease(self) -> SandboxLease:
        if not self._ready or self._lease is None:
            raise RuntimeError("Environment has not started")
        return self._lease

    def _run_sync(
        self,
        command,
        cwd=None,
        env=None,
        timeout_sec=None,
        user=None,
        command_id=None,
        *,
        strict_output=False,
    ):
        self._check_cancelled()
        lease = self._require_lease()
        if self._resolve_user(user) not in (None, 0, "0", "root"):
            raise ValueError("This backend does not implement non-root user switching")
        timeout = self._command_timeout if timeout_sec is None else timeout_sec
        if (
            not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Command timeout must be positive and finite")
        values = {**self._image["env"], **(self._merge_env(env) or {})}
        if any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
            or not isinstance(value, str)
            for key, value in values.items()
        ):
            raise ValueError("Invalid command environment mapping")
        wrapped = f"cd -- {shlex.quote(cwd or self._cwd)} && " + shlex.join(
            ["bash", "-c", command]
        )
        options = {} if command_id is None else {"command_id": command_id}
        result = lease.run_command(
            wrapped, timeout=math.ceil(timeout), env=values, **options
        )
        if result.output_truncated and strict_output:
            raise RuntimeError(
                "Sandfleet command output exceeded its transport bound; output is incomplete"
            )
        stderr = result.stderr
        if result.output_truncated:
            stderr += (
                "\n[Sandfleet: command output truncated at the transport limit.]\n"
            )
        # The worker deadline kills the command; report it as GNU timeout would.
        return_code = 124 if getattr(result, "timed_out", False) else result.exit_code
        return ExecResult(stdout=result.stdout, stderr=stderr, return_code=return_code)

    async def exec(
        self, command, cwd=None, env=None, timeout_sec=None, user=None
    ) -> ExecResult:
        command_id = uuid.uuid4().hex
        return await self._call(
            self._run_sync,
            command,
            cwd,
            env,
            timeout_sec,
            user,
            command_id,
            command_id=command_id,
        )

    async def kill_sandbox_processes(self) -> int:
        """SIGKILL every sandbox process except its init; return how many were killed.

        Call after the agent phase and before verification, so nothing the agent
        left running (for example a background loop rewriting
        /logs/verifier/reward.json) can tamper with the verifier. The filesystem
        is kept; reset the verifier directories afterwards. Raises if any process
        survives; do not run the verifier then. Needs Sandfleet 0.12 workers.
        """
        return await self._call(self._kill_processes_sync)

    def _kill_processes_sync(self) -> int:
        self._check_cancelled()
        return self._require_lease().kill_processes()["killed"]

    def _checked(self, command):
        result = self._run_sync(command, user="root", strict_output=True)
        if result.return_code:
            raise RuntimeError(
                f"File transfer command failed ({result.return_code}): {result.stderr}"
            )
        return result.stdout

    def _upload_file(self, source, target):
        lease = self._require_lease()
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("Uploads require regular files, not symlinks")
        offset = 0
        with source.open("rb") as handle:
            while True:
                self._check_cancelled()
                data = handle.read(CHUNK_BYTES)
                lease.write_file_chunk(
                    target, data, offset=offset, truncate=offset == 0
                )
                offset += len(data)
                if len(data) < CHUNK_BYTES:
                    break
        self._checked(
            f"chmod {source.stat().st_mode & 0o777:o} -- {shlex.quote(target)}"
        )

    async def upload_file(self, source_path, target_path):
        await self._call(self._upload_file, source_path, target_path)

    def _upload_dir(self, source, target):
        source = Path(source)
        if source.is_symlink() or not source.is_dir():
            raise ValueError("Upload source must be a real directory")
        self._checked(f"mkdir -p -- {shlex.quote(target)}")
        for path in sorted(source.rglob("*")):
            destination = str(
                PurePosixPath(target) / path.relative_to(source).as_posix()
            )
            if path.is_symlink():
                raise ValueError("Directory uploads do not follow symlinks")
            if path.is_dir():
                self._checked(f"mkdir -p -- {shlex.quote(destination)}")
            else:
                self._upload_file(path, destination)

    async def upload_dir(self, source_dir, target_dir):
        await self._call(self._upload_dir, source_dir, target_dir)

    def _download_file(self, source, target):
        lease = self._require_lease()
        target = Path(target)
        if target.is_symlink():
            raise ValueError("Download target is a symlink")
        target.parent.mkdir(parents=True, exist_ok=True)
        offset = 0
        with target.open("wb") as output:
            while True:
                self._check_cancelled()
                data = lease.read_file_chunk(source, offset=offset, size=CHUNK_BYTES)
                output.write(data)
                offset += len(data)
                if len(data) < CHUNK_BYTES:
                    break

    async def download_file(self, source_path, target_path):
        await self._recorded_download(self._download_file, source_path, target_path)

    async def _recorded_download(self, operation, source, target):
        # Harbor collects artifacts best-effort (it logs a failure and goes on), so
        # record infrastructure failures where a caller can see them.
        try:
            await self._call(operation, source, target)
        except Exception as error:
            if transfer_failure_is_infra(error):
                self.transfer_infra_failures.append(
                    f"{source}: {type(error).__name__}: {error}"
                )
            raise

    def _download_dir(self, source, target):
        # Stream files individually: creating a tar inside a full task disk can
        # prevent verifier-log recovery exactly when those logs are needed most.
        encoded = self._checked(
            "set -o pipefail; "
            + f"test -d {shlex.quote(source)} && test ! -L {shlex.quote(source)} && find {shlex.quote(source)} -xdev -printf '%y\\0%P\\0%m\\0' | base64 -w0"
        )
        parts = base64.b64decode(encoded, validate=True).decode("utf-8").split("\x00")
        if parts[-1] != "" or (len(parts) - 1) % 3:
            raise ValueError("Malformed remote directory inventory")
        target = Path(target)
        if target.is_symlink():
            raise ValueError("Download root is a symlink")
        target.mkdir(parents=True, exist_ok=True)
        base = target.resolve()
        for kind, relative, mode in zip(parts[0:-1:3], parts[1:-1:3], parts[2:-1:3]):
            name = PurePosixPath(relative)
            if name.is_absolute() or ".." in name.parts or kind not in ("d", "f"):
                raise ValueError("Unsafe file type/path in remote artifacts")
            destination = base / relative
            if (
                not destination.resolve().is_relative_to(base)
                or destination.is_symlink()
            ):
                raise ValueError("Artifact path escapes download directory")
            if kind == "d":
                destination.mkdir(parents=True, exist_ok=True)
            else:
                self._download_file(str(PurePosixPath(source) / name), destination)
                destination.chmod(int(mode, 8) & 0o777)

    async def download_dir(self, source_dir, target_dir):
        await self._recorded_download(self._download_dir, source_dir, target_dir)
