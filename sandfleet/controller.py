"""Sandfleet control plane: elastic pools, worker healing, and lease routing."""

from __future__ import annotations

import argparse
import contextlib
import hmac
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from sandfleet.audit import DEFAULT_AUDIT_MAX_MB
from sandfleet.dashboard import DASHBOARD_HTML
from sandfleet.egress import add_egress_arguments, egress_settings_from_args
from sandfleet.errors import NoCapacityError, SandboxOOMError, SandfleetError
from sandfleet.http import API_VERSION, JsonHandler, request_json, secret_from_env
from sandfleet.resources import (
    GPU_CONTAINER_MODES,
    DynamicPoolPolicy,
    GPUPlacement,
    GPURequest,
    GPUTypeCatalog,
    SandboxResources,
)
from sandfleet.slurm import (
    SlurmLauncher,
    SlurmSettings,
    SlurmSubmission,
    slurm_time_seconds,
)
from sandfleet.network import NETWORK_MODES, network_mode_argument
from sandfleet.state_file import (
    DurableStateFile,
    StateRecovery,
    StateValidationError,
)
from sandfleet.version import VERSION

POOL_NAME = re.compile(r"[A-Za-z0-9_-]+")
# Older agents run workers whose "offline" sandboxes have DNS and NAT egress.
MIN_AGENT_VERSION = "0.9.0"
VERSION_PREFIX = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _version_key(version: Any) -> tuple[int, ...]:
    """Leading numeric components padded to three; () if there are none."""
    match = VERSION_PREFIX.match(version) if isinstance(version, str) else None
    return () if match is None else tuple(int(part or 0) for part in match.groups())


def check_agent_version(version: Any) -> None:
    """Reject agents that predate the current sandbox network policy."""
    if _version_key(version) < _version_key(MIN_AGENT_VERSION):
        raise ValueError(
            f"Agent version {version!r} is older than {MIN_AGENT_VERSION}, whose "
            "offline sandboxes are not offline; replace the worker allocation"
        )


def _validate_gpu_controller_config(
    *,
    catalog: GPUTypeCatalog | None,
    max_gpus_per_sandbox: int,
    dynamic_pool_policy: DynamicPoolPolicy | None,
    device_isolation_verified: bool,
    refresh_seconds: int,
) -> None:
    if catalog is None and max_gpus_per_sandbox != 0:
        raise ValueError("A per-sandbox GPU limit requires a GPU type catalog")
    if catalog is not None and max_gpus_per_sandbox < 1:
        raise ValueError("GPU type catalog requires a positive per-sandbox GPU limit")
    if (catalog is not None) != device_isolation_verified:
        raise ValueError(
            "GPU catalog and explicit device-isolation verification must be configured together"
        )
    if catalog is not None and dynamic_pool_policy is None:
        raise ValueError("GPU sandboxes require declarative resource pools")
    if catalog is not None and dynamic_pool_policy.max_gpus_per_worker < 1:
        raise ValueError("GPU sandboxes require a positive per-worker GPU limit")
    if catalog is not None and refresh_seconds < 30:
        raise ValueError("GPU capability refresh interval must be at least 30 seconds")


@dataclass(frozen=True)
class PoolSpec:
    name: str
    min_capacity: int
    max_capacity: int
    warm_capacity: int
    slots_per_job: int
    cpus: int
    memory_mb: int
    idle_timeout_seconds: int
    gpu_count: int = 0
    gpu_type: str | None = None
    slurm_gpu_type: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PoolSpec":
        name = str(payload["name"]).strip()
        fixed_capacity = payload.get("desired_capacity")
        elastic_capacity = payload.get("max_capacity")
        if (fixed_capacity is None) == (elastic_capacity is None):
            raise ValueError("Set exactly one of desired_capacity and max_capacity")

        if fixed_capacity is not None:
            unexpected = {"min_capacity", "warm_capacity"} & payload.keys()
            if unexpected:
                raise ValueError("Fixed pools cannot set min_capacity or warm_capacity")
            minimum = maximum = warm = int(fixed_capacity)
        else:
            maximum = int(elastic_capacity)
            minimum = int(payload.get("min_capacity", 0))
            warm = int(payload.get("warm_capacity", minimum))

        return cls(
            name=name,
            min_capacity=minimum,
            max_capacity=maximum,
            warm_capacity=warm,
            slots_per_job=int(payload.get("slots_per_job", max(1, maximum))),
            cpus=int(payload.get("cpus", 1)),
            memory_mb=int(payload.get("memory_mb", 4096)),
            idle_timeout_seconds=int(payload.get("idle_timeout_seconds", 600)),
        )

    def validate(
        self,
        *,
        max_capacity: int,
        max_cpus_per_sandbox: int,
        max_memory_mb_per_sandbox: int,
    ) -> None:
        if POOL_NAME.fullmatch(self.name) is None:
            raise ValueError("Pool name must contain only letters, digits, '-' and '_'")
        if not 1 <= self.max_capacity <= max_capacity:
            raise ValueError(f"max_capacity must be between 1 and {max_capacity}")
        if not 0 <= self.min_capacity <= self.max_capacity:
            raise ValueError("min_capacity must be between 0 and max_capacity")
        if not 0 <= self.warm_capacity <= self.max_capacity:
            raise ValueError("warm_capacity must be between 0 and max_capacity")
        if not 1 <= self.slots_per_job <= self.max_capacity:
            raise ValueError("slots_per_job must be between 1 and max_capacity")
        if not 1 <= self.cpus <= max_cpus_per_sandbox:
            raise ValueError(f"cpus must be between 1 and {max_cpus_per_sandbox}")
        if not 64 <= self.memory_mb <= max_memory_mb_per_sandbox:
            raise ValueError(
                f"memory_mb must be between 64 and {max_memory_mb_per_sandbox}"
            )
        if self.idle_timeout_seconds < 0:
            raise ValueError("idle_timeout_seconds must be non-negative")
        self._validate_gpu()

    def _validate_gpu(self) -> None:
        gpu_fields = (self.gpu_type, self.slurm_gpu_type)
        if self.gpu_count == 0 and any(value is not None for value in gpu_fields):
            raise ValueError("CPU pools cannot set GPU type fields")
        if self.gpu_count > 0 and any(value is None for value in gpu_fields):
            raise ValueError("GPU pools require logical and Slurm GPU types")
        if self.gpu_count < 0:
            raise ValueError("GPU count must be non-negative")


@dataclass
class PoolRecord:
    name: str
    min_capacity: int
    max_capacity: int
    warm_capacity: int
    slots_per_job: int
    cpus: int
    memory_mb: int
    idle_timeout_seconds: int
    gpu_count: int = 0
    gpu_type: str | None = None
    slurm_gpu_type: str | None = None
    dynamic: bool = False
    created_at: float = field(default_factory=time.time)
    next_scale_at: float = 0.0
    scale_up_after: float = 0.0
    scale_failures: int = 0
    last_scale_error: str | None = None

    @classmethod
    def from_spec(cls, spec: PoolSpec, *, dynamic: bool = False) -> "PoolRecord":
        return cls(**asdict(spec), dynamic=dynamic)

    def spec(self) -> PoolSpec:
        return PoolSpec(
            name=self.name,
            min_capacity=self.min_capacity,
            max_capacity=self.max_capacity,
            warm_capacity=self.warm_capacity,
            slots_per_job=self.slots_per_job,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            idle_timeout_seconds=self.idle_timeout_seconds,
            gpu_count=self.gpu_count,
            gpu_type=self.gpu_type,
            slurm_gpu_type=self.slurm_gpu_type,
        )

    def reset_scale_failures(self) -> None:
        self.scale_failures = 0
        self.last_scale_error = None
        self.next_scale_at = 0.0

    def record_scale_failure(self, error: Exception | str, retry_at: float) -> None:
        self.scale_failures += 1
        self.last_scale_error = str(error)
        self.next_scale_at = max(self.next_scale_at, retry_at)


@dataclass
class SubmissionRecord:
    submission_id: str
    pool: str
    worker_count: int
    slots: int
    submitted_at: float
    registered_worker_keys: list[str] = field(default_factory=list)
    cached_states: dict[str, str] = field(default_factory=dict)
    last_seen_at: float | None = None
    terminal: bool = False
    failure_accounted: bool = False

    def observe_scheduler(
        self, states: dict[str, str], *, now: float, grace_seconds: int
    ) -> bool:
        """Update cached state and report an unregistered terminal submission."""
        self.cached_states = states
        if states:
            self.last_seen_at = now
            return False
        if now - self.submitted_at < grace_seconds:
            return False
        self.terminal = True
        failed_before_registration = (
            not self.registered_worker_keys and not self.failure_accounted
        )
        self.failure_accounted = self.failure_accounted or failed_before_registration
        return failed_before_registration


@dataclass
class AgentRecord:
    agent_id: str
    pool: str
    url: str
    slots: int
    free_slots: int
    job_id: str | None
    worker_key: str
    submission_id: str
    node: str
    internet: bool
    ready_slots: int = 0
    preparing_slots: int = 0
    failed_slots: int = 0
    slot_errors: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, int] = field(default_factory=dict)
    status: str = "active"
    idle_since: float | None = None
    last_heartbeat: float = field(default_factory=time.time)
    cleanup_pending: bool = False
    cleanup_attempts: int = 0
    next_cleanup_at: float = 0.0
    last_cleanup_error: str | None = None

    def update_heartbeat(self, payload: dict[str, Any], now: float) -> None:
        self.last_heartbeat = now
        self.status = "active"
        for key in ("free_slots", "ready_slots", "preparing_slots", "failed_slots"):
            if key in payload:
                setattr(self, key, int(payload[key]))
        if "slot_errors" in payload:
            self.slot_errors = {
                str(key): str(value) for key, value in payload["slot_errors"].items()
            }
        if "metrics" in payload:
            self.metrics = {
                str(key): int(value) for key, value in payload["metrics"].items()
            }
        fully_idle = self.ready_slots == self.free_slots == self.slots
        self.idle_since = (self.idle_since or now) if fully_idle else None

    def finish_cleanup(self) -> None:
        self.cleanup_pending = False
        self.next_cleanup_at = 0.0
        self.last_cleanup_error = None


@dataclass
class LeaseRecord:
    lease_id: str
    lease_token: str
    pool: str
    agent_id: str
    agent_url: str
    internet: bool
    status: str = "active"
    lost_reason: str | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    ttl_seconds: int = 0
    last_renewed_at: float | None = None


@dataclass
class LeaseRequestRecord:
    request_id: str
    pool: str
    created_at: float
    expires_at: float
    internet: bool
    status: str = "pending"
    lease_id: str | None = None
    error: str | None = None
    finished_at: float | None = None


@dataclass(frozen=True)
class PersistedControllerState:
    version: int
    pools: dict[str, PoolRecord]
    submissions: dict[str, SubmissionRecord]
    agents: dict[str, AgentRecord]
    leases: dict[str, LeaseRecord]
    lease_requests: dict[str, LeaseRequestRecord]
    durable_metrics: dict[str, int]


class ControllerState:
    """Durable desired state plus a single rate-limited Slurm reconciler."""

    # Version 8: every saved agent passed the MIN_AGENT_VERSION registration gate.
    STATE_VERSION = 8
    VERSION_FOUR_DURABLE_METRICS = (
        "controller_terminations",
        "worker_cleanup_failures",
        "worker_cleanups_completed",
        "worker_preemptions_absorbed",
    )
    DURABLE_METRICS = (
        *VERSION_FOUR_DURABLE_METRICS,
        "state_recoveries",
    )

    def __init__(
        self,
        *,
        token: str,
        client_token: str,
        metrics_token: str,
        launcher: SlurmLauncher,
        max_capacity: int,
        max_cpus_per_sandbox: int,
        max_memory_mb_per_sandbox: int,
        max_gpus_per_sandbox: int = 0,
        state_file: str | None = None,
        agent_stale_seconds: int = 30,
        agent_lost_seconds: int = 60,
        submission_grace_seconds: int = 30,
        min_scheduler_poll_seconds: int = 15,
        min_submit_interval_seconds: int = 10,
        max_workers_per_scale: int = 64,
        max_scale_failures: int = 5,
        request_retention_seconds: int = 3600,
        state_save_debounce_seconds: float = 1.0,
        scale_up_debounce_seconds: float = 2.0,
        step_start_rate: float = 8.0,
        step_start_burst: int = 8,
        lease_ttl_seconds: int = 3600,
        max_pools: int = 32,
        max_pending_requests: int = 8192,
        dynamic_pool_policy: DynamicPoolPolicy | None = None,
        gpu_type_catalog: GPUTypeCatalog | None = None,
        gpu_device_isolation_verified: bool = False,
        gpu_capability_refresh_seconds: int = 300,
    ):
        if agent_lost_seconds < agent_stale_seconds:
            raise ValueError("agent_lost_seconds must be >= agent_stale_seconds")
        self.token = token
        self.client_token = client_token
        self.metrics_token = metrics_token
        self.launcher = launcher
        self.max_capacity = max_capacity
        self.max_cpus_per_sandbox = max_cpus_per_sandbox
        self.max_memory_mb_per_sandbox = max_memory_mb_per_sandbox
        self.max_gpus_per_sandbox = max_gpus_per_sandbox
        self.state_file = None if state_file is None else Path(state_file)
        self.state_store = (
            None
            if self.state_file is None
            else DurableStateFile[PersistedControllerState](self.state_file)
        )
        self.agent_stale_seconds = agent_stale_seconds
        self.agent_lost_seconds = agent_lost_seconds
        self.submission_grace_seconds = submission_grace_seconds
        self.min_scheduler_poll_seconds = min_scheduler_poll_seconds
        self.min_submit_interval_seconds = min_submit_interval_seconds
        self.max_workers_per_scale = max_workers_per_scale
        self.max_scale_failures = max_scale_failures
        self.request_retention_seconds = request_retention_seconds
        self.state_save_debounce_seconds = state_save_debounce_seconds
        self.scale_up_debounce_seconds = scale_up_debounce_seconds
        if step_start_rate <= 0 or step_start_burst < 1:
            raise ValueError("step_start_rate and step_start_burst must be positive")
        if lease_ttl_seconds < 0 or max_pools < 1 or max_pending_requests < 1:
            raise ValueError("lease TTL and controller limits are invalid")
        self.step_start_rate = step_start_rate
        self.step_start_burst = step_start_burst
        self.lease_ttl_seconds = lease_ttl_seconds
        self.max_pools = max_pools
        self.max_pending_requests = max_pending_requests
        if dynamic_pool_policy is not None:
            dynamic_pool_policy.validate(
                controller_max_capacity=max_capacity, max_pools=max_pools
            )
        self.dynamic_pool_policy = dynamic_pool_policy
        _validate_gpu_controller_config(
            catalog=gpu_type_catalog,
            max_gpus_per_sandbox=max_gpus_per_sandbox,
            dynamic_pool_policy=dynamic_pool_policy,
            device_isolation_verified=gpu_device_isolation_verified,
            refresh_seconds=gpu_capability_refresh_seconds,
        )
        self.gpu_type_catalog = gpu_type_catalog
        self.gpu_device_isolation_verified = gpu_device_isolation_verified
        self.gpu_capability_refresh_seconds = gpu_capability_refresh_seconds

        self._initialize_runtime(step_start_burst)
        self._load()

    def _initialize_runtime(self, step_start_burst: int) -> None:
        """Initialize mutable state that is either loaded or rebuilt on startup."""

        self.pools: dict[str, PoolRecord] = {}
        self.submissions: dict[str, SubmissionRecord] = {}
        self.agents: dict[str, AgentRecord] = {}
        self.leases: dict[str, LeaseRecord] = {}
        self.lease_requests: dict[str, LeaseRequestRecord] = {}
        self.scheduler_ok = True
        self.scheduler_error: str | None = None
        self.scheduler_checked_at: float | None = None
        self.last_scheduler_poll_at = 0.0
        self.last_submission_at = 0.0
        self.last_state_save_at = 0.0
        self.started_at = time.time()
        self._step_tokens = float(step_start_burst)
        self._step_token_updated_at = time.monotonic()
        self.metrics: dict[str, int] = {
            "slurm_snapshot_calls": 0,
            "slurm_snapshot_failures": 0,
            "slurm_submit_calls": 0,
            "slurm_submit_failures": 0,
            "slurm_cancel_calls": 0,
            "slurm_cancel_items": 0,
            "step_permits_granted": 0,
            "step_permits_delayed": 0,
            "leases_created": 0,
            "internet_leases_created": 0,
            "leases_released": 0,
            "leases_expired": 0,
            "lease_release_transport_failures": 0,
            "lease_release_cleanup_pending": 0,
            "controller_terminations": 0,
            "worker_cleanup_failures": 0,
            "worker_cleanups_completed": 0,
            "worker_preemptions_absorbed": 0,
            "state_recoveries": 0,
        }

        self.lock, self.assignment_lock, self.reconcile_lock = (
            threading.RLock(),
            threading.Lock(),
            threading.Lock(),
        )
        self.reconcile_wakeup = threading.Event()
        self.reconcile_stop = threading.Event()
        self.reconcile_thread: threading.Thread | None = None
        self.reconcile_error: BaseException | None = None
        self.state_recovery: dict[str, Any] | None = None
        self.gpu_capability_snapshot: dict[str, Any] | None = None
        self.gpu_capability_refresh_requested = self.gpu_type_catalog is not None
        self.last_gpu_capability_refresh_at = 0.0
        self.gpu_capability_refresh_thread: threading.Thread | None = None

    # ---------- persistence ----------

    @staticmethod
    def _pool_items_from_state(
        payload: dict[str, Any], version: int
    ) -> list[dict[str, Any]]:
        pool_items = [dict(item) for item in payload["pools"]]
        if version == 2:
            for item in pool_items:
                item["dynamic"] = False
        if version < 6:
            for item in pool_items:
                item["gpu_count"] = 0
                item["gpu_type"] = None
                item["slurm_gpu_type"] = None
        return pool_items

    @staticmethod
    def _agent_items_from_state(
        payload: dict[str, Any], version: int
    ) -> list[dict[str, Any]]:
        agent_items = [dict(item) for item in payload["agents"]]
        if version < 4:
            for item in agent_items:
                item["cleanup_pending"] = item["status"] == "lost"
                item["cleanup_attempts"] = 0
                item["next_cleanup_at"] = 0.0
                item["last_cleanup_error"] = None
        return agent_items

    @classmethod
    def _decode_state(cls, payload: dict[str, Any]) -> PersistedControllerState:
        try:
            version = int(payload["version"])
            if version not in {2, 3, 4, 5, 6, 7, cls.STATE_VERSION}:
                raise StateValidationError(
                    f"Unsupported Sandfleet state version {version}"
                )
            pool_items = cls._pool_items_from_state(payload, version)
            pools = {item["name"]: PoolRecord(**item) for item in pool_items}
            submissions = {
                item["submission_id"]: SubmissionRecord(**item)
                for item in payload["submissions"]
            }
            # Agents and leases saved before sandbox internet existed were offline.
            migrated = {} if version >= 7 else {"internet": False}
            agent_items = cls._agent_items_from_state(payload, version)
            agents = {
                item["agent_id"]: AgentRecord(**{**item, **migrated})
                for item in agent_items
            }
            leases = {
                item["lease_id"]: LeaseRecord(**{**item, **migrated})
                for item in payload["leases"]
            }
            lease_requests = {
                item["request_id"]: LeaseRequestRecord(**{**item, **migrated})
                for item in payload["lease_requests"]
            }
            metric_names = (
                cls.DURABLE_METRICS
                if version >= 5
                else cls.VERSION_FOUR_DURABLE_METRICS
                if version == 4
                else ()
            )
            durable_metrics = {
                name: int(payload["durable_metrics"][name]) for name in metric_names
            }
        except StateValidationError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise StateValidationError(f"Invalid Sandfleet state: {error}") from error
        return PersistedControllerState(
            version=version,
            pools=pools,
            submissions=submissions,
            agents=agents,
            leases=leases,
            lease_requests=lease_requests,
            durable_metrics=durable_metrics,
        )

    def _load(self) -> None:
        if self.state_store is None:
            return
        state, recovery = self.state_store.load(self._decode_state)
        if state is None:
            return
        self.pools = state.pools
        self.submissions = state.submissions
        self.agents = state.agents
        # Give live workers one loss-grace window to prove liveness after a
        # controller restart. They are not eligible for new placement until a
        # fresh heartbeat arrives, but existing leases are not falsely lost.
        recovered_at = time.time()
        for agent in self.agents.values():
            if agent.status in {"active", "stale"}:
                agent.status = "stale"
                agent.free_slots = 0
                agent.ready_slots = 0
                agent.last_heartbeat = recovered_at
        self.leases = state.leases
        if state.version < 8:
            # Agents from older state never passed the version gate and may run
            # pre-0.9 workers whose "offline" sandboxes had DNS and NAT egress.
            for agent in self.agents.values():
                self._mark_agent_lost_locked(
                    agent,
                    f"Agent predates the Sandfleet {MIN_AGENT_VERSION} network policy",
                    cleanup_worker=True,
                )
        self.lease_requests = state.lease_requests
        self.metrics.update(state.durable_metrics)
        if recovery is not None:
            self._record_state_recovery(recovery)

    def _record_state_recovery(self, recovery: StateRecovery) -> None:
        recovered_at = time.time()
        self.metrics["state_recoveries"] += 1
        self.state_recovery = {
            "recovered_at": recovered_at,
            "source": str(recovery.source),
            "quarantined": (
                None if recovery.quarantined is None else str(recovery.quarantined)
            ),
            "failures": list(recovery.failures),
        }
        print(
            f"WARNING: recovered controller state from {recovery.source}; "
            f"quarantined={recovery.quarantined}",
            file=sys.stderr,
            flush=True,
        )
        self._save_locked(force=True)

    def token_is_admin(self, token: str) -> bool:
        return bool(token) and hmac.compare_digest(token, self.token)

    def token_is_client(self, token: str) -> bool:
        return bool(token) and hmac.compare_digest(token, self.client_token)

    def token_is_metrics(self, token: str) -> bool:
        return bool(token) and hmac.compare_digest(token, self.metrics_token)

    def _save_locked(self, *, force: bool = False) -> None:
        if self.state_store is None:
            return
        now = time.monotonic()
        if (
            not force
            and now - self.last_state_save_at < self.state_save_debounce_seconds
        ):
            return
        payload = {
            "version": self.STATE_VERSION,
            "pools": [asdict(value) for value in self.pools.values()],
            "submissions": [asdict(value) for value in self.submissions.values()],
            "agents": [asdict(value) for value in self.agents.values()],
            "leases": [asdict(value) for value in self.leases.values()],
            "lease_requests": [asdict(value) for value in self.lease_requests.values()],
            "durable_metrics": {
                name: self.metrics[name] for name in self.DURABLE_METRICS
            },
        }
        self.state_store.save(payload)
        self.last_state_save_at = now

    def _save(self, *, force: bool = False) -> None:
        with self.lock:
            self._save_locked(force=force)

    def _wake(self) -> None:
        self.reconcile_wakeup.set()

    # ---------- pool and agent control ----------

    def _insert_pool_locked(self, spec: PoolSpec, *, dynamic: bool) -> PoolRecord:
        existing = self.pools.get(spec.name)
        if existing is not None:
            if existing.spec() != spec or existing.dynamic != dynamic:
                raise ValueError(
                    f"Pool {spec.name!r} already exists with a different specification"
                )
            return existing
        if len(self.pools) >= self.max_pools:
            raise ValueError(f"Controller pool limit ({self.max_pools}) reached")
        aggregate_capacity = (
            sum(pool.max_capacity for pool in self.pools.values()) + spec.max_capacity
        )
        if aggregate_capacity > self.max_capacity:
            raise ValueError(
                f"Aggregate pool capacity {aggregate_capacity} exceeds controller limit {self.max_capacity}"
            )
        pool = PoolRecord.from_spec(spec, dynamic=dynamic)
        self.pools[spec.name] = pool
        return pool

    def create_pool(self, payload: dict[str, Any]) -> dict[str, Any]:
        spec = PoolSpec.from_payload(payload)
        spec.validate(
            max_capacity=self.max_capacity,
            max_cpus_per_sandbox=self.max_cpus_per_sandbox,
            max_memory_mb_per_sandbox=self.max_memory_mb_per_sandbox,
        )
        with self.lock:
            self._insert_pool_locked(spec, dynamic=False)
            self._save_locked(force=True)
        self._wake()
        return self.pool_status(spec.name)

    def _pool_for_resources(self, payload: dict[str, Any]) -> str:
        policy = self.dynamic_pool_policy
        if policy is None:
            raise ValueError(
                "Declarative resource requests are disabled; start the controller "
                "with --enable-dynamic-resources and explicit worker CPU/memory limits"
            )
        if not isinstance(payload, dict):
            raise ValueError("resources must be a JSON object")
        resources = SandboxResources.from_payload(payload)
        resources.validate(
            max_cpus_per_sandbox=self.max_cpus_per_sandbox,
            max_memory_mb_per_sandbox=self.max_memory_mb_per_sandbox,
            max_gpus_per_sandbox=self.max_gpus_per_sandbox,
            gpu_type_catalog=self.gpu_type_catalog,
        )
        gpu_placement = None
        if resources.gpu_request is not None:
            gpu_placement = self._resolve_gpu_placement(resources.gpu_request)
        pool_resources = resources
        if gpu_placement is not None:
            pool_resources = SandboxResources(
                cpus=resources.cpus,
                memory_mb=resources.memory_mb,
                gpu_request=GPURequest(
                    gpu=resources.gpu_request.gpu,
                    type=(gpu_placement.logical_type,),
                ),
            )
        spec = PoolSpec(
            name=pool_resources.pool_name,
            min_capacity=0,
            max_capacity=policy.max_capacity,
            warm_capacity=policy.warm_capacity,
            slots_per_job=policy.slots_for(pool_resources),
            cpus=resources.cpus,
            memory_mb=resources.memory_mb,
            idle_timeout_seconds=policy.idle_timeout_seconds,
            gpu_count=(
                0 if resources.gpu_request is None else resources.gpu_request.gpu
            ),
            gpu_type=None if gpu_placement is None else gpu_placement.logical_type,
            slurm_gpu_type=(
                None if gpu_placement is None else gpu_placement.slurm_type
            ),
        )
        spec.validate(
            max_capacity=self.max_capacity,
            max_cpus_per_sandbox=self.max_cpus_per_sandbox,
            max_memory_mb_per_sandbox=self.max_memory_mb_per_sandbox,
        )
        with self.lock:
            existing = self.pools.get(spec.name)
            if existing is None:
                dynamic_count = sum(pool.dynamic for pool in self.pools.values())
                if dynamic_count >= policy.max_pools:
                    raise ValueError(f"Dynamic pool limit ({policy.max_pools}) reached")
                self._insert_pool_locked(spec, dynamic=True)
                self._save_locked(force=True)
            elif not existing.dynamic or (
                existing.cpus,
                existing.memory_mb,
                existing.gpu_count,
                existing.gpu_type,
                existing.slurm_gpu_type,
            ) != (
                spec.cpus,
                spec.memory_mb,
                spec.gpu_count,
                spec.gpu_type,
                spec.slurm_gpu_type,
            ):
                raise ValueError(f"Resource pool name collision for {spec.name!r}")
            # Persisted packing/capacity is authoritative. A new controller's
            # defaults apply only to newly created shapes, not live pools.
        return spec.name

    def _gpu_pool_error_locked(self, pool: PoolRecord) -> str | None:
        if not pool.gpu_count:
            return None
        catalog = self.gpu_type_catalog
        spec = catalog.types.get(pool.gpu_type or "") if catalog is not None else None
        if spec is None:
            return f"GPU type {pool.gpu_type!r} is not enabled by this controller"
        if spec.slurm_type != pool.slurm_gpu_type:
            return f"GPU type {pool.gpu_type!r} now maps to a different Slurm type"
        if pool.gpu_count > self.max_gpus_per_sandbox:
            return (
                f"Pool requests {pool.gpu_count} GPUs; this controller allows "
                f"{self.max_gpus_per_sandbox} per sandbox"
            )
        return None

    def _resolve_gpu_placement(self, request: GPURequest) -> GPUPlacement:
        catalog = self.gpu_type_catalog
        assert catalog is not None
        with self.lock:
            snapshot = self.gpu_capability_snapshot
        if snapshot is None or snapshot["error"] is not None:
            # Availability is advisory; Slurm still decides at submission.
            return catalog.resolve(request)

        placements = []
        for requested_type in request.type:
            if requested_type not in catalog.types:
                continue
            placement = catalog.resolve(GPURequest(request.gpu, (requested_type,)))
            observed = snapshot["types"].get(placement.logical_type, {})
            inventory = observed.get("inventory") or {}
            if (
                observed.get("scheduler_request_valid") is True
                and int(inventory.get("configured", 0)) > 0
            ):
                placements.append((placement, inventory))
        if not placements:
            raise ValueError(
                "None of the requested GPU types is available to this controller's Slurm identity"
            )
        return next(
            (
                placement
                for placement, inventory in placements
                if int(inventory.get("largest_node_free_estimate", 0)) >= request.gpu
            ),
            placements[0][0],
        )

    @staticmethod
    def _worker_identity(payload: dict[str, Any]) -> tuple[str, str]:
        job_id = str(payload.get("job_id") or "local")
        array_job_id = payload.get("array_job_id")
        array_task_id = payload.get("array_task_id")
        if array_job_id is not None and array_task_id is not None:
            return f"{array_job_id}_{array_task_id}", str(array_job_id)
        return job_id, job_id

    def _mark_agent_lost_locked(
        self, agent: AgentRecord, reason: str, *, cleanup_worker: bool
    ) -> None:
        if agent.status == "lost":
            return
        agent.status = "lost"
        agent.free_slots = 0
        agent.ready_slots = 0
        agent.cleanup_pending = cleanup_worker
        agent.next_cleanup_at = 0.0
        agent.last_cleanup_error = None
        for lease in self.leases.values():
            if lease.agent_id == agent.agent_id and lease.status == "active":
                lease.status = "lost"
                lease.lost_reason = reason

    def register_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        check_agent_version(payload.get("version"))
        worker_key, submission_id = self._worker_identity(payload)
        now = time.time()
        free_slots = int(payload.get("free_slots", payload["slots"]))
        ready_slots = int(payload.get("ready_slots", free_slots))
        # Agents older than sandbox internet do not report the capability.
        internet = payload.get("internet", False)
        if type(internet) is not bool:
            raise ValueError("Agent internet capability must be a JSON boolean")
        agent = AgentRecord(
            agent_id=str(payload["agent_id"]),
            pool=str(payload["pool"]),
            url=str(payload["url"]).rstrip("/"),
            slots=int(payload["slots"]),
            free_slots=free_slots,
            job_id=None if payload.get("job_id") is None else str(payload["job_id"]),
            worker_key=worker_key,
            submission_id=submission_id,
            node=str(payload.get("node", "")),
            internet=internet,
            ready_slots=ready_slots,
            preparing_slots=int(payload.get("preparing_slots", 0)),
            failed_slots=int(payload.get("failed_slots", 0)),
            slot_errors={
                str(key): str(value)
                for key, value in payload.get("slot_errors", {}).items()
            },
            metrics={
                str(key): int(value)
                for key, value in payload.get("metrics", {}).items()
            },
            idle_since=now
            if ready_slots == int(payload["slots"]) and free_slots == ready_slots
            else None,
            last_heartbeat=now,
        )
        with self.lock:
            pool = self.pools[agent.pool]
            reported_gpus = int(payload.get("gpus", 0))
            reported_gpu_type = payload.get("gpu_type")
            if (
                reported_gpus != pool.gpu_count
                or reported_gpu_type != pool.slurm_gpu_type
            ):
                raise ValueError("Agent GPU allocation does not match its pool")
            for existing in self.agents.values():
                if (
                    existing.worker_key == worker_key
                    and existing.agent_id != agent.agent_id
                ):
                    self._mark_agent_lost_locked(
                        existing, "Worker process restarted", cleanup_worker=False
                    )
            self.agents[agent.agent_id] = agent
            submission = self.submissions.get(submission_id)
            if (
                submission is not None
                and worker_key not in submission.registered_worker_keys
            ):
                submission.registered_worker_keys.append(worker_key)
            pool.reset_scale_failures()
            self._save_locked()
        self._wake()
        return {"registered": True, "agent_id": agent.agent_id}

    def heartbeat(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            agent = self.agents[agent_id]
            if agent.status in {"lost", "draining"}:
                return {"ok": False, "status": agent.status}
            agent.update_heartbeat(payload, now)
            has_capacity = agent.free_slots > 0
        if has_capacity:
            self._wake()
        return {"ok": True}

    def acquire_step_permit(self) -> dict[str, Any]:
        """Globally pace new srun step creation across every node agent."""
        with self.lock:
            now = time.monotonic()
            elapsed = max(0.0, now - self._step_token_updated_at)
            self._step_tokens = min(
                float(self.step_start_burst),
                self._step_tokens + elapsed * self.step_start_rate,
            )
            self._step_token_updated_at = now
            if self._step_tokens >= 1.0:
                self._step_tokens -= 1.0
                self.metrics["step_permits_granted"] += 1
                return {
                    "granted": True,
                    "retry_after_seconds": 0,
                    "rate_per_second": self.step_start_rate,
                    "burst": self.step_start_burst,
                }
            self.metrics["step_permits_delayed"] += 1
            retry = (1.0 - self._step_tokens) / self.step_start_rate
            return {
                "granted": False,
                "retry_after_seconds": max(0.05, retry),
                "rate_per_second": self.step_start_rate,
                "burst": self.step_start_burst,
            }

    def live_agents(self, pool: str) -> list[AgentRecord]:
        cutoff = time.time() - self.agent_stale_seconds
        with self.lock:
            return [
                agent
                for agent in self.agents.values()
                if agent.pool == pool
                and agent.status == "active"
                and agent.last_heartbeat >= cutoff
            ]

    # ---------- leases ----------

    def _reserve_lease(self, pool_name: str, internet: bool) -> LeaseRecord | None:
        candidates = sorted(
            (
                agent
                for agent in self.live_agents(pool_name)
                if agent.free_slots > 0 and (agent.internet or not internet)
            ),
            key=lambda agent: (-agent.free_slots, agent.agent_id),
        )
        for candidate in candidates:
            try:
                result = request_json(
                    candidate.url,
                    f"/{API_VERSION}/control/leases",
                    method="POST",
                    token=self.token,
                    payload={"internet": internet},
                    timeout=15,
                )
            except NoCapacityError:
                with self.lock:
                    self.agents[candidate.agent_id].free_slots = 0
                continue
            except SandfleetError as error:
                if error.status is not None:
                    raise
                with self.lock:
                    current = self.agents[candidate.agent_id]
                    current.status = "stale"
                    current.free_slots = 0
                continue
            lease = LeaseRecord(
                lease_id=str(result["lease_id"]),
                lease_token=str(result["lease_token"]),
                pool=pool_name,
                agent_id=candidate.agent_id,
                agent_url=candidate.url,
                internet=internet,
                expires_at=(
                    None
                    if self.lease_ttl_seconds == 0
                    else time.time() + self.lease_ttl_seconds
                ),
                ttl_seconds=self.lease_ttl_seconds,
                last_renewed_at=time.time(),
            )
            with self.lock:
                self.leases[lease.lease_id] = lease
                current = self.agents.get(candidate.agent_id)
                if current is not None:
                    current.free_slots = int(result["free_slots"])
                    current.idle_since = None
                self._save_locked()
                self.metrics["leases_created"] += 1
                self.metrics["internet_leases_created"] += internet
            return lease
        return None

    def request_lease(
        self,
        pool_name: str | None = None,
        *,
        resources: dict[str, Any] | None = None,
        timeout_seconds: int = 900,
        internet: bool = False,
    ) -> dict[str, Any]:
        if (pool_name is None) == (resources is None):
            raise ValueError("Set exactly one of pool and resources")
        if type(internet) is not bool:
            raise ValueError("internet must be a JSON boolean")
        if internet and not self.launcher.settings.internet.enabled:
            raise ValueError("Sandbox internet is disabled on this controller")
        if resources is not None:
            pool_name = self._pool_for_resources(resources)
        assert pool_name is not None
        with self.lock:
            pool = self.pools[pool_name]
            gpu_pool_error = self._gpu_pool_error_locked(pool)
            if gpu_pool_error is not None:
                raise ValueError(gpu_pool_error)
            pending_total = sum(
                request.status == "pending" for request in self.lease_requests.values()
            )
            if pending_total >= self.max_pending_requests:
                raise NoCapacityError(
                    f"Controller pending request limit ({self.max_pending_requests}) reached"
                )
            now = time.time()
            had_pending_demand = self._pending_request_count_locked(pool_name) > 0
            request = LeaseRequestRecord(
                request_id=uuid.uuid4().hex,
                pool=pool_name,
                created_at=now,
                expires_at=now + max(1, timeout_seconds),
                internet=internet,
            )
            self.lease_requests[request.request_id] = request
            if not had_pending_demand:
                pool.scale_up_after = now + self.scale_up_debounce_seconds
            self._save_locked()
        self._assign_pending()
        self._wake()
        return self.lease_request_status(request.request_id)

    def lease_request_status(self, request_id: str) -> dict[str, Any]:
        with self.lock:
            request = self.lease_requests[request_id]
            result = asdict(request)
            pool = self.pools[request.pool]
            result["resources"] = {
                "cpus": pool.cpus,
                "memory_mb": pool.memory_mb,
            }
            if pool.gpu_count:
                result["resources"].update(
                    {"gpu": pool.gpu_count, "type": [pool.gpu_type]}
                )
            if request.lease_id is not None:
                lease = self.leases.get(request.lease_id)
                if lease is not None:
                    result["lease"] = asdict(lease)
            result["poll_after_seconds"] = 2
            return result

    def cancel_lease_request(self, request_id: str) -> dict[str, Any]:
        with self.lock:
            request = self.lease_requests.get(request_id)
            if request is None:
                return {"cancelled": False}
            if request.status != "assigned":
                request.status = "cancelled"
                request.finished_at = time.time()
                self._save_locked()
                return {"cancelled": True}
            if request.lease_id is None:
                raise RuntimeError("Assigned lease request has no lease ID")
            lease_id = request.lease_id

        # Keep an assigned request assigned until agent cleanup succeeds so a
        # failed cancellation remains retryable.
        self.release(lease_id)
        with self.lock:
            request = self.lease_requests[request_id]
            request.status = "cancelled"
            request.finished_at = time.time()
            self._save_locked()
        return {"cancelled": True}

    def _assign_pending(self) -> None:
        if not self.assignment_lock.acquire(blocking=False):
            return
        try:
            while True:
                with self.lock:
                    pending = sorted(
                        (
                            request
                            for request in self.lease_requests.values()
                            if request.status == "pending"
                        ),
                        key=lambda request: (request.created_at, request.request_id),
                    )
                assigned_any = False
                for request in pending:
                    try:
                        lease = self._reserve_lease(request.pool, request.internet)
                    except SandfleetError as error:
                        if error.status is None:
                            raise
                        # An agent refused this request; fail it alone so
                        # later requests are still assigned.
                        self._fail_request(request.request_id, error)
                        continue
                    if lease is None:
                        continue
                    with self.lock:
                        current = self.lease_requests.get(request.request_id)
                        if current is None or current.status != "pending":
                            orphaned = True
                        else:
                            current.status = "assigned"
                            current.lease_id = lease.lease_id
                            current.finished_at = time.time()
                            self._save_locked()
                            orphaned = False
                    if orphaned:
                        self.release(lease.lease_id)
                    else:
                        assigned_any = True
                if not assigned_any:
                    return
        finally:
            self.assignment_lock.release()

    def _fail_request(self, request_id: str, error: Exception) -> None:
        with self.lock:
            current = self.lease_requests.get(request_id)
            if current is not None and current.status == "pending":
                current.status = "failed"
                current.error = f"Agent rejected lease reservation: {error}"
                current.finished_at = time.time()
                self._save_locked()

    def lease_status(self, lease_id: str) -> dict[str, Any]:
        with self.lock:
            return asdict(self.leases[lease_id])

    def renew_lease(self, lease_id: str) -> dict[str, Any]:
        with self.lock:
            lease = self.leases[lease_id]
            if lease.status != "active":
                raise RuntimeError(f"Lease {lease_id!r} is {lease.status}")
            now = time.time()
            lease.last_renewed_at = now
            lease.expires_at = (
                None if lease.ttl_seconds == 0 else now + lease.ttl_seconds
            )
            self._save_locked()
            return asdict(lease)

    def release(self, lease_id: str) -> dict[str, Any]:
        with self.lock:
            lease = self.leases.get(lease_id)
            if lease is None:
                return {"released": False}
        free_slots: int | None = None
        if lease.status == "active":
            result = request_json(
                lease.agent_url,
                f"/{API_VERSION}/control/leases/{lease_id}",
                method="DELETE",
                token=self.token,
                timeout=30,
            )
            free_slots = int(result["free_slots"])
        with self.lock:
            if self.leases.pop(lease_id, None) is None:
                return {"released": False}
            for request in self.lease_requests.values():
                if request.lease_id == lease_id:
                    request.status = "released"
                    request.finished_at = time.time()
            agent = self.agents.get(lease.agent_id)
            if agent is not None and free_slots is not None:
                agent.free_slots = free_slots
                if agent.free_slots == agent.ready_slots == agent.slots:
                    agent.idle_since = time.time()
            self.metrics["leases_released"] += 1
            self._save_locked()
        self._wake()
        return {"released": True}

    # ---------- cached status and reconciliation ----------

    def _active_lease_count_locked(self, pool: str) -> int:
        return sum(
            1
            for lease in self.leases.values()
            if lease.pool == pool and lease.status == "active"
        )

    def _pending_request_count_locked(self, pool: str) -> int:
        return sum(
            1
            for request in self.lease_requests.values()
            if request.pool == pool and request.status == "pending"
        )

    def _target_capacity_locked(self, pool: PoolRecord) -> int:
        demand = self._active_lease_count_locked(
            pool.name
        ) + self._pending_request_count_locked(pool.name)
        return min(
            pool.max_capacity, max(pool.min_capacity, demand + pool.warm_capacity)
        )

    def _pending_submission_workers_locked(
        self, submission: SubmissionRecord, now: float
    ) -> int:
        if submission.terminal:
            return 0
        registered = set(submission.registered_worker_keys)
        if submission.cached_states:
            return sum(
                1
                for worker_key in submission.cached_states
                if worker_key not in registered
            )
        if now - submission.submitted_at < self.submission_grace_seconds:
            return max(0, submission.worker_count - len(registered))
        return 0

    def _current_worker_count_locked(self, pool_name: str, now: float) -> int:
        agents = sum(
            1
            for agent in self.agents.values()
            if agent.pool == pool_name
            and agent.status in {"active", "stale"}
            and now - agent.last_heartbeat <= self.agent_lost_seconds
        )
        pending = sum(
            self._pending_submission_workers_locked(submission, now)
            for submission in self.submissions.values()
            if submission.pool == pool_name
        )
        return agents + pending

    def _pool_status_locked(self, pool: PoolRecord, now: float) -> dict[str, Any]:
        agents = [agent for agent in self.agents.values() if agent.pool == pool.name]
        live = [
            agent
            for agent in agents
            if agent.status == "active"
            and now - agent.last_heartbeat <= self.agent_stale_seconds
        ]
        submissions = [
            submission
            for submission in self.submissions.values()
            if submission.pool == pool.name
        ]
        target_capacity = self._target_capacity_locked(pool)
        return {
            **asdict(pool),
            "desired_capacity": target_capacity,
            "target_capacity": target_capacity,
            "elastic": pool.min_capacity != pool.max_capacity,
            "ready_capacity": sum(agent.ready_slots for agent in live),
            "free_capacity": sum(agent.free_slots for agent in live),
            "active_leases": self._active_lease_count_locked(pool.name),
            "pending_leases": self._pending_request_count_locked(pool.name),
            "current_workers": self._current_worker_count_locked(pool.name, now),
            "agents": [asdict(agent) for agent in agents],
            "submissions": [asdict(submission) for submission in submissions],
            "job_states": {
                worker_key: state
                for submission in submissions
                for worker_key, state in submission.cached_states.items()
            },
            "scheduler": self._scheduler_status_locked(),
            "scaling_suspended": pool.scale_failures >= self.max_scale_failures,
        }

    def _scheduler_status_locked(self) -> dict[str, Any]:
        return {
            "ok": self.scheduler_ok,
            "error": self.scheduler_error,
            "checked_at": self.scheduler_checked_at,
        }

    def pool_status(self, name: str) -> dict[str, Any]:
        with self.lock:
            return self._pool_status_locked(self.pools[name], time.time())

    def _refresh_scheduler(self, now: float) -> bool:
        if now - self.last_scheduler_poll_at < self.min_scheduler_poll_seconds:
            return self.scheduler_ok
        with self.lock:
            submission_ids = [
                submission.submission_id
                for submission in self.submissions.values()
                if not submission.terminal
            ]
            self.metrics["slurm_snapshot_calls"] += 1
        snapshot = self.launcher.snapshot(submission_ids)
        with self.lock:
            self.last_scheduler_poll_at = now
            self.scheduler_ok = snapshot.ok
            self.scheduler_error = snapshot.error
            self.scheduler_checked_at = now
            if not snapshot.ok:
                self.metrics["slurm_snapshot_failures"] += 1
                return False
            for submission in self.submissions.values():
                if submission.terminal:
                    continue
                states = snapshot.states.get(submission.submission_id, {})
                if submission.observe_scheduler(
                    states, now=now, grace_seconds=self.submission_grace_seconds
                ):
                    pool = self.pools[submission.pool]
                    backoff = min(
                        900,
                        self.min_submit_interval_seconds
                        * (2 ** min(pool.scale_failures + 1, 6)),
                    )
                    pool.record_scale_failure(
                        f"Worker submission {submission.submission_id} ended before an agent registered",
                        now + backoff,
                    )
            return True

    def _mark_timed_out_agents_locked(self, now: float) -> bool:
        changed = False
        for agent in self.agents.values():
            age = now - agent.last_heartbeat
            if agent.status == "active" and age > self.agent_stale_seconds:
                agent.status = "stale"
                agent.free_slots = 0
                agent.ready_slots = 0
                changed = True
            if agent.status == "stale" and age > self.agent_lost_seconds:
                self._mark_agent_lost_locked(
                    agent,
                    f"Worker heartbeat missing for {age:.0f}s",
                    cleanup_worker=True,
                )
                changed = True
        return changed

    def _reap_agents_locked(
        self, now: float, *, scheduler_ok: bool
    ) -> tuple[list[AgentRecord], bool]:
        cleanup: list[AgentRecord] = []
        changed = self._mark_timed_out_agents_locked(now)
        owned_worker_keys = {
            agent.worker_key for agent in self.agents.values() if agent.status != "lost"
        }
        for agent in self.agents.values():
            if agent.status != "lost" or not agent.cleanup_pending:
                continue
            if agent.worker_key in owned_worker_keys:
                agent.finish_cleanup()
                changed = True
                continue
            if not scheduler_ok:
                continue
            submission = self.submissions.get(agent.submission_id)
            if (
                submission is not None
                and agent.worker_key not in submission.cached_states
            ):
                agent.finish_cleanup()
                if agent.cleanup_attempts == 0:
                    self.metrics["worker_preemptions_absorbed"] += 1
                else:
                    self.metrics["worker_cleanups_completed"] += 1
                changed = True
            elif now >= agent.next_cleanup_at:
                cleanup.append(agent)
        return cleanup, changed

    def _reap_requests_locked(self, now: float) -> bool:
        changed = False
        for request in self.lease_requests.values():
            if request.status == "pending" and now >= request.expires_at:
                request.status = "expired"
                request.error = "Timed out waiting for elastic sandbox capacity"
                request.finished_at = now
                changed = True
        expired = [
            request_id
            for request_id, request in self.lease_requests.items()
            if request.finished_at is not None
            and now - request.finished_at >= self.request_retention_seconds
        ]
        for request_id in expired:
            del self.lease_requests[request_id]
        return changed or bool(expired)

    def _retry_lost_agent_cleanup(self, agents: list[AgentRecord], now: float) -> None:
        if not agents:
            return
        with self.lock:
            pending = [
                agent
                for agent in agents
                if agent.status == "lost" and agent.cleanup_pending
            ]
            for agent in pending:
                agent.cleanup_attempts += 1
            self._save_locked(force=True)
        if not pending:
            return
        try:
            self._cancel_jobs([agent.worker_key for agent in pending])
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            with self.lock:
                self.metrics["worker_cleanup_failures"] += len(pending)
                for agent in pending:
                    backoff = min(
                        300,
                        max(1, self.min_submit_interval_seconds)
                        * (2 ** min(agent.cleanup_attempts - 1, 6)),
                    )
                    agent.last_cleanup_error = str(error)
                    agent.next_cleanup_at = now + backoff
                self._save_locked(force=True)
            return
        with self.lock:
            self.metrics["worker_cleanups_completed"] += len(pending)
            for agent in pending:
                agent.finish_cleanup()
            self._save_locked(force=True)

    def _reap_agents_and_requests(self, now: float, *, scheduler_ok: bool) -> None:
        with self.lock:
            cleanup, agents_changed = self._reap_agents_locked(
                now, scheduler_ok=scheduler_ok
            )
            requests_changed = self._reap_requests_locked(now)
            expired_leases = [
                lease.lease_id
                for lease in self.leases.values()
                if lease.status == "active"
                and lease.expires_at is not None
                and now >= lease.expires_at
            ]
            if agents_changed or requests_changed:
                self._save_locked()
        self._retry_lost_agent_cleanup(cleanup, now)
        for lease_id in expired_leases:
            try:
                released = self.release(lease_id)["released"]
            except SandfleetError as error:
                cleanup_pending = (
                    error.status == 503 and error.error_type == "LeaseCleanupPending"
                ) or isinstance(error, SandboxOOMError)
                if error.status is not None and not cleanup_pending:
                    raise
                with self.lock:
                    metric = (
                        "lease_release_cleanup_pending"
                        if cleanup_pending
                        else "lease_release_transport_failures"
                    )
                    self.metrics[metric] += 1
                continue
            if released:
                with self.lock:
                    self.metrics["leases_expired"] += 1

    def _record_submission(self, pool: PoolRecord, submission: SlurmSubmission) -> None:
        with self.lock:
            self.submissions[submission.submission_id] = SubmissionRecord(
                submission_id=submission.submission_id,
                pool=pool.name,
                worker_count=submission.worker_count,
                slots=pool.slots_per_job,
                submitted_at=submission.submitted_at,
            )
            pool.next_scale_at = (
                submission.submitted_at + self.min_submit_interval_seconds
            )
            self.last_submission_at = submission.submitted_at
            self._save_locked(force=True)

    def _cancel_jobs(self, job_ids: list[str]) -> None:
        if not job_ids:
            return
        with self.lock:
            self.metrics["slurm_cancel_calls"] += 1
            self.metrics["slurm_cancel_items"] += len(job_ids)
        self.launcher.cancel(job_ids)

    def _scale_up_once(self, now: float) -> bool:
        if now - self.last_submission_at < self.min_submit_interval_seconds:
            return False
        with self.lock:
            candidates: list[tuple[PoolRecord, int]] = []
            for pool in self.pools.values():
                if pool.scale_failures >= self.max_scale_failures:
                    continue
                gpu_pool_error = self._gpu_pool_error_locked(pool)
                if gpu_pool_error is not None:
                    pool.last_scale_error = gpu_pool_error
                    continue
                target_workers = math.ceil(
                    self._target_capacity_locked(pool) / pool.slots_per_job
                )
                current_workers = self._current_worker_count_locked(pool.name, now)
                if target_workers > current_workers and now >= max(
                    pool.next_scale_at, pool.scale_up_after
                ):
                    candidates.append((pool, target_workers - current_workers))
            if not candidates:
                return False
            pool, deficit = min(
                candidates, key=lambda item: (item[0].next_scale_at, item[0].created_at)
            )
            worker_count = min(deficit, self.max_workers_per_scale)
        try:
            with self.lock:
                self.metrics["slurm_submit_calls"] += 1
            gpu_arguments: dict[str, Any] = {}
            if pool.gpu_count:
                gpu_arguments = {
                    "gpus": pool.gpu_count,
                    "gpu_type": pool.slurm_gpu_type,
                }
            submission = self.launcher.submit_worker_batch(
                pool=pool.name,
                worker_count=worker_count,
                slots=pool.slots_per_job,
                cpus=pool.cpus,
                memory_mb=pool.memory_mb,
                **gpu_arguments,
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            with self.lock:
                self.metrics["slurm_submit_failures"] += 1
                backoff = min(
                    300,
                    self.min_submit_interval_seconds
                    * (2 ** min(pool.scale_failures + 1, 5)),
                )
                pool.record_scale_failure(error, now + backoff)
                self._save_locked()
            return False
        self._record_submission(pool, submission)
        return True

    def resume_pool(self, name: str) -> dict[str, Any]:
        with self.lock:
            pool = self.pools[name]
            pool.reset_scale_failures()
            pool.scale_up_after = 0.0
            self._save_locked(force=True)
        self._wake()
        return self.pool_status(name)

    def gpu_capabilities(self) -> dict[str, Any]:
        """Return operator configuration plus the last reconciler-owned probe."""

        catalog = self.gpu_type_catalog
        if catalog is None:
            return {
                "enabled": False,
                "checked_at": None,
                "error": None,
                "types": {},
                "qos_gpu_quota": None,
                "qos_gpu_quota_error": None,
                "availability_is_advisory": True,
                "refresh_interval_seconds": self.gpu_capability_refresh_seconds,
            }
        policy = self.dynamic_pool_policy
        assert policy is not None
        with self.lock:
            observed = self.gpu_capability_snapshot
            observed_types = {} if observed is None else observed["types"]
            return {
                "enabled": True,
                "checked_at": None if observed is None else observed["checked_at"],
                "error": None if observed is None else observed["error"],
                "qos_gpu_quota": (
                    None if observed is None else observed["qos_gpu_quota"]
                ),
                "qos_gpu_quota_error": (
                    None if observed is None else observed["qos_gpu_quota_error"]
                ),
                "types": {
                    logical_type: {
                        "memory_mb": spec.memory_mb,
                        "max_per_sandbox": min(
                            self.max_gpus_per_sandbox,
                            policy.max_gpus_per_worker,
                        ),
                        "inventory": observed_types.get(logical_type, {}).get(
                            "inventory"
                        ),
                        "scheduler_request_valid": observed_types.get(
                            logical_type, {}
                        ).get("scheduler_request_valid"),
                        "scheduler_probe_detail": observed_types.get(
                            logical_type, {}
                        ).get("scheduler_probe_detail"),
                    }
                    for logical_type, spec in sorted(catalog.types.items())
                },
                "request_schema": {
                    "gpu": "positive integer GPU count",
                    "type": "ordered non-empty list of exact memory-qualified types",
                },
                "availability_is_advisory": True,
                "refresh_interval_seconds": self.gpu_capability_refresh_seconds,
                "allocation_proof": (
                    "A prepared lease must report exactly the requested GPU device and "
                    "nvidia-smi identity counts; inventory and scheduler probes do not "
                    "reserve capacity."
                ),
            }

    def request_gpu_capability_refresh(self) -> dict[str, Any]:
        if self.gpu_type_catalog is None:
            raise ValueError("GPU sandboxes are not configured")
        with self.lock:
            self.gpu_capability_refresh_requested = True
        self._wake()
        return {"refresh_requested": True}

    def _start_gpu_capability_refresh(self) -> None:
        """Start a due Slurm probe off the reconcile loop; client GETs remain cache-only."""

        if self.gpu_type_catalog is None:
            return
        monotonic_now = time.monotonic()
        with self.lock:
            thread = self.gpu_capability_refresh_thread
            refresh_due = (
                monotonic_now - self.last_gpu_capability_refresh_at
                >= self.gpu_capability_refresh_seconds
            )
            if (thread is not None and thread.is_alive()) or not (
                self.gpu_capability_refresh_requested or refresh_due
            ):
                return
            self.gpu_capability_refresh_requested = False
            self.last_gpu_capability_refresh_at = monotonic_now
            self.gpu_capability_refresh_thread = threading.Thread(
                target=self._refresh_gpu_capabilities,
                name="sandfleet-gpu-capabilities",
                daemon=True,
            )
            self.gpu_capability_refresh_thread.start()

    def _refresh_gpu_capabilities(self) -> None:
        # Discovery is advisory: malformed scheduler output must not stop the controller.
        catalog = self.gpu_type_catalog
        assert catalog is not None
        checked_at = time.time()
        try:
            inventory = self.launcher.discover_gpu_inventory(
                {spec.slurm_type for spec in catalog.types.values()}
            )
            probes: dict[str, dict[str, Any]] = {}
            for slurm_type in sorted(inventory):
                try:
                    probes[slurm_type] = self.launcher.probe_gpu_access(slurm_type)
                except Exception as error:
                    probes[slurm_type] = {"ok": False, "detail": repr(error)}
            try:
                qos_gpu_quota = self.launcher.discover_gpu_quota()
                qos_gpu_quota_error = None
            except Exception as error:
                qos_gpu_quota = None
                qos_gpu_quota_error = repr(error)
            snapshot = {
                "checked_at": checked_at,
                "error": None,
                "qos_gpu_quota": qos_gpu_quota,
                "qos_gpu_quota_error": qos_gpu_quota_error,
                "types": {
                    logical_type: {
                        "inventory": inventory[spec.slurm_type],
                        "scheduler_request_valid": probes[spec.slurm_type]["ok"],
                        "scheduler_probe_detail": probes[spec.slurm_type]["detail"],
                    }
                    for logical_type, spec in catalog.types.items()
                },
            }
        except Exception as error:
            snapshot = {
                "checked_at": checked_at,
                "error": repr(error),
                "qos_gpu_quota": None,
                "qos_gpu_quota_error": None,
                "types": {},
            }
        with self.lock:
            self.gpu_capability_snapshot = snapshot

    def _begin_scale_down_locked(self, now: float) -> list[AgentRecord]:
        draining: list[AgentRecord] = []
        for pool in self.pools.values():
            target_workers = math.ceil(
                self._target_capacity_locked(pool) / pool.slots_per_job
            )
            current_workers = self._current_worker_count_locked(pool.name, now)
            surplus = max(0, current_workers - target_workers)
            if surplus == 0:
                continue
            candidates = sorted(
                (
                    agent
                    for agent in self.agents.values()
                    if agent.pool == pool.name
                    and agent.status == "active"
                    and agent.free_slots == agent.slots
                    and agent.idle_since is not None
                    and now - agent.idle_since >= pool.idle_timeout_seconds
                ),
                key=lambda agent: (agent.idle_since or now, agent.agent_id),
            )
            for agent in candidates[:surplus]:
                agent.status = "draining"
                agent.free_slots = 0
                draining.append(agent)
        if draining:
            self._save_locked()
        return draining

    def _scale_down(self, now: float) -> None:
        with self.lock:
            draining = self._begin_scale_down_locked(now)
        if not draining:
            return
        try:
            self._cancel_jobs([agent.worker_key for agent in draining])
        except Exception:
            with self.lock:
                for agent in draining:
                    agent.status = "active"
                    agent.free_slots = agent.slots
                self._save_locked(force=True)
            raise

    def reconcile_once(self) -> None:
        if not self.reconcile_lock.acquire(blocking=False):
            return
        try:
            now = time.time()
            self._start_gpu_capability_refresh()
            scheduler_ok = self._refresh_scheduler(now)
            self._reap_agents_and_requests(now, scheduler_ok=scheduler_ok)
            self._assign_pending()
            if scheduler_ok:
                self._scale_up_once(now)
                self._scale_down(now)
            self._save()
        finally:
            self.reconcile_lock.release()

    def start_reconciler(
        self,
        interval_seconds: int,
        *,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if self.reconcile_thread is not None:
            return

        def loop() -> None:
            try:
                while not self.reconcile_stop.is_set():
                    self.reconcile_wakeup.clear()
                    self.reconcile_once()
                    self.reconcile_wakeup.wait(interval_seconds)
            except BaseException as error:
                self._record_reconcile_termination(error)
                if on_error is not None:
                    on_error(error)
                raise

        self.reconcile_thread = threading.Thread(
            target=loop, name="sandfleet-reconciler", daemon=True
        )
        self.reconcile_thread.start()

    def _record_reconcile_termination(self, error: BaseException) -> None:
        self.reconcile_error = error
        with self.lock:
            self.metrics["controller_terminations"] += 1
            try:
                self._save_locked(force=True)
            except Exception as save_error:
                error.add_note(
                    f"Could not persist controller termination: {save_error}"
                )

    def stop_reconciler(self) -> None:
        self.reconcile_stop.set()
        self.reconcile_wakeup.set()
        if self.reconcile_thread is not None:
            self.reconcile_thread.join(timeout=10)
            self.reconcile_thread = None

    # ---------- observability ----------

    def metrics_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self.lock:
            pools = [
                self._pool_status_locked(pool, now) for pool in self.pools.values()
            ]
            agents = [asdict(agent) for agent in self.agents.values()]
            counters = dict(self.metrics)
            tokens = self._step_tokens
            scheduler = self._scheduler_status_locked()
        return {
            "version": VERSION,
            "timestamp": now,
            "uptime_seconds": now - self.started_at,
            "scheduler": scheduler,
            "totals": {
                "pools": len(pools),
                "agents": len(agents),
                "ready_capacity": sum(pool["ready_capacity"] for pool in pools),
                "free_capacity": sum(pool["free_capacity"] for pool in pools),
                "active_leases": sum(pool["active_leases"] for pool in pools),
                "pending_leases": sum(pool["pending_leases"] for pool in pools),
            },
            "step_limiter": {
                "rate_per_second": self.step_start_rate,
                "burst": self.step_start_burst,
                "available_tokens": round(tokens, 3),
            },
            "persistence": {"last_recovery": self.state_recovery},
            "limits": {
                "max_aggregate_capacity": self.max_capacity,
                "max_pools": self.max_pools,
                "max_pending_requests": self.max_pending_requests,
                "lease_ttl_seconds": self.lease_ttl_seconds,
                "dynamic_resources": (
                    None
                    if self.dynamic_pool_policy is None
                    else asdict(self.dynamic_pool_policy)
                ),
            },
            "counters": counters,
            "pools": [
                {
                    key: pool[key]
                    for key in (
                        "name",
                        "dynamic",
                        "cpus",
                        "memory_mb",
                        "min_capacity",
                        "max_capacity",
                        "warm_capacity",
                        "target_capacity",
                        "ready_capacity",
                        "free_capacity",
                        "active_leases",
                        "pending_leases",
                        "current_workers",
                        "scale_failures",
                        "last_scale_error",
                        "scaling_suspended",
                    )
                }
                for pool in pools
            ],
            "agents": agents,
        }

    def prometheus_metrics(self) -> str:
        snapshot = self.metrics_snapshot()
        lines = [
            "# HELP sandfleet_up Whether the controller scheduler view is healthy.",
            "# TYPE sandfleet_up gauge",
            f"sandfleet_up {1 if snapshot['scheduler']['ok'] else 0}",
            "# TYPE sandfleet_uptime_seconds gauge",
            f"sandfleet_uptime_seconds {snapshot['uptime_seconds']:.3f}",
        ]
        for name, value in snapshot["totals"].items():
            lines.extend(
                (f"# TYPE sandfleet_{name} gauge", f"sandfleet_{name} {value}")
            )
        for name, value in snapshot["counters"].items():
            lines.extend(
                (
                    f"# TYPE sandfleet_{name}_total counter",
                    f"sandfleet_{name}_total {value}",
                )
            )
        for pool in snapshot["pools"]:
            label = str(pool["name"]).replace("\\", "\\\\").replace('"', '\\"')
            for metric_name in (
                "target_capacity",
                "ready_capacity",
                "free_capacity",
                "active_leases",
                "pending_leases",
            ):
                lines.append(
                    f'sandfleet_pool_{metric_name}{{pool="{label}"}} {pool[metric_name]}'
                )
        return "\n".join(lines) + "\n"

    # ---------- deletion and shutdown ----------

    def delete_pool(self, name: str) -> dict[str, Any]:
        with self.lock:
            pool = self.pools.get(name)
            if pool is None:
                return {"deleted": False}
            lease_ids = [
                lease_id
                for lease_id, lease in self.leases.items()
                if lease.pool == name
            ]
            submission_ids = [
                submission.submission_id
                for submission in self.submissions.values()
                if submission.pool == name
            ]
        try:
            for lease_id in lease_ids:
                self.release(lease_id)
        finally:
            # Also tears down sandboxes whose agent could not be reached.
            self._cancel_jobs(submission_ids)
        with self.lock:
            self.pools.pop(name, None)
            self.submissions = {
                key: submission
                for key, submission in self.submissions.items()
                if submission.pool != name
            }
            self.agents = {
                key: agent for key, agent in self.agents.items() if agent.pool != name
            }
            self.lease_requests = {
                key: request
                for key, request in self.lease_requests.items()
                if request.pool != name
            }
            self._save_locked(force=True)
        return {"deleted": True, "cancelled_jobs": submission_ids}

    def close(self, *, cleanup_jobs: bool) -> None:
        self.stop_reconciler()
        if not cleanup_jobs:
            self._save(force=True)
            return
        with self.lock:
            pool_names = list(self.pools)
        errors = []
        for name in pool_names:
            try:
                self.delete_pool(name)
            except Exception as error:
                error.add_note(f"while deleting pool {name!r}")
                errors.append(error)
        if errors:
            raise errors[0]


class ControllerHandler(JsonHandler):
    @property
    def state(self) -> ControllerState:
        return self.server.state  # type: ignore[attr-defined, no-any-return]

    def _require(self, role: str) -> None:
        token = self.bearer_token()
        allowed = {
            "admin": self.state.token_is_admin,
            "client": self.state.token_is_client,
            "metrics": self.state.token_is_metrics,
        }[role](token)
        if not allowed:
            raise PermissionError(f"Invalid or missing {role} bearer token")

    def do_GET(self) -> None:  # noqa: N802
        try:
            parts = self.path_parts()
            if not parts or parts == ["dashboard"]:
                self.send_text(
                    200, DASHBOARD_HTML, content_type="text/html; charset=utf-8"
                )
            elif parts == ["metrics"]:
                self._require("metrics")
                self.send_text(
                    200,
                    self.state.prometheus_metrics(),
                    content_type="text/plain; version=0.0.4",
                )
            elif parts == [API_VERSION, "metrics"]:
                self._require("metrics")
                self.send_json(200, self.state.metrics_snapshot())
            else:
                self._require("client")
                if parts == [API_VERSION, "health"]:
                    self.send_json(
                        200,
                        {
                            "ok": self.state.reconcile_error is None,
                            "version": VERSION,
                            "features": [
                                "disk-overlay-v1",
                                "file-chunks-v1",
                                "lease-abort-v1",
                                "command-env-v1",
                                "command-cancel-v1",
                                "lease-internet-v1",
                                "lease-kill-processes-v1",
                                "lease-readonly-mounts-v1",
                                "lease-start-readonly-mounts-v1",
                            ],
                            "max_storage_mb": self.state.launcher.settings.max_storage_mb,
                            "readonly_mount_roots": list(
                                self.state.launcher.settings.readonly_mount_roots
                            ),
                            "agent_readonly_mount_roots": list(
                                self.state.launcher.settings.agent_readonly_mount_roots
                            ),
                            "internet": self.state.launcher.settings.internet.mode,
                            # Each worker job's --time (None: unlimited/unknown);
                            # leases are not placed by a worker's remaining time.
                            "worker_time_limit_seconds": slurm_time_seconds(
                                self.state.launcher.settings.time_limit
                            ),
                            "reconcile_error": (
                                None
                                if self.state.reconcile_error is None
                                else str(self.state.reconcile_error)
                            ),
                            "scheduler_ok": self.state.scheduler_ok,
                            "scheduler_error": self.state.scheduler_error,
                            "state_recovery": self.state.state_recovery,
                        },
                    )
                elif parts == [API_VERSION, "capabilities", "gpus"]:
                    self.send_json(200, self.state.gpu_capabilities())
                elif len(parts) == 3 and parts[:2] == [API_VERSION, "pools"]:
                    self.send_json(200, self.state.pool_status(parts[2]))
                elif len(parts) == 3 and parts[:2] == [API_VERSION, "lease-requests"]:
                    self.send_json(200, self.state.lease_request_status(parts[2]))
                elif len(parts) == 3 and parts[:2] == [API_VERSION, "leases"]:
                    self.send_json(200, self.state.lease_status(parts[2]))
                else:
                    raise KeyError(self.path)
        except Exception as error:
            self.send_error_json(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parts = self.path_parts()
            payload = self.read_json()
            if parts == [API_VERSION, "pools"]:
                self._require("admin")
                self.send_json(201, self.state.create_pool(payload))
            elif parts == [API_VERSION, "capabilities", "gpus", "refresh"]:
                self._require("admin")
                self.send_json(202, self.state.request_gpu_capability_refresh())
            elif parts == [API_VERSION, "agents", "register"]:
                self._require("admin")
                self.send_json(201, self.state.register_agent(payload))
            elif (
                len(parts) == 4
                and parts[:2] == [API_VERSION, "agents"]
                and parts[3] == "heartbeat"
            ):
                self._require("admin")
                self.send_json(200, self.state.heartbeat(parts[2], payload))
            elif parts == [API_VERSION, "step-permits"]:
                self._require("admin")
                self.send_json(200, self.state.acquire_step_permit())
            elif parts == [API_VERSION, "lease-requests"]:
                self._require("client")
                self.send_json(
                    202,
                    self.state.request_lease(
                        (None if payload.get("pool") is None else str(payload["pool"])),
                        resources=payload.get("resources"),
                        timeout_seconds=int(payload.get("timeout_seconds", 900)),
                        internet=payload.get("internet", False),
                    ),
                )
            elif (
                len(parts) == 4
                and parts[:2] == [API_VERSION, "leases"]
                and parts[3] == "renew"
            ):
                self._require("client")
                self.send_json(200, self.state.renew_lease(parts[2]))
            elif (
                len(parts) == 4
                and parts[:2] == [API_VERSION, "pools"]
                and parts[3] == "resume"
            ):
                self._require("admin")
                self.send_json(200, self.state.resume_pool(parts[2]))
            else:
                raise KeyError(self.path)
        except Exception as error:
            self.send_error_json(error)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            parts = self.path_parts()
            if len(parts) == 3 and parts[:2] == [API_VERSION, "leases"]:
                self._require("client")
                self.send_json(200, self.state.release(parts[2]))
            elif len(parts) == 3 and parts[:2] == [API_VERSION, "lease-requests"]:
                self._require("client")
                self.send_json(200, self.state.cancel_lease_request(parts[2]))
            elif len(parts) == 3 and parts[:2] == [API_VERSION, "pools"]:
                self._require("admin")
                self.send_json(200, self.state.delete_pool(parts[2]))
            else:
                raise KeyError(self.path)
        except Exception as error:
            self.send_error_json(error)


def _add_gpu_parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-gpus-per-sandbox", type=int, default=0)
    parser.add_argument(
        "--gpu-type-catalog",
        help="Operator-owned JSON mapping from logical GPU names to Slurm GRES types",
    )
    parser.add_argument(
        "--gpu-device-isolation-verified",
        action="store_true",
        help="Assert that Slurm device cgroups enforce the configured GPU GRES devices",
    )
    parser.add_argument("--dynamic-worker-max-gpus", type=int, default=0)
    parser.add_argument("--gpu-capability-refresh-seconds", type=int, default=300)


def add_sandbox_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--apptainer-binary", default="apptainer")
    parser.add_argument(
        "--network-mode",
        type=network_mode_argument,
        default="loopback",
        help=f"Sandbox networking, one of {NETWORK_MODES}",
    )
    parser.add_argument(
        "--gpu-container-mode",
        choices=GPU_CONTAINER_MODES,
        default="auto",
        help="Apptainer GPU integration for GPU sandboxes: --nv, --nvccli, or auto-detect",
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--tmp-dir")
    parser.add_argument(
        "--max-storage-mb",
        type=int,
        default=0,
        help="Opt-in maximum nominal disk-overlay MiB per lease; 0 disables",
    )
    parser.add_argument(
        "--sandbox-readonly-mount-root",
        action="append",
        default=[],
        dest="sandbox_readonly_mount_roots",
        help="Verifier-phase root: host directory whose subdirectories leases may "
        "add to a started disk-overlay sandbox as read-only mounts (repeatable); "
        "none disables",
    )
    parser.add_argument(
        "--sandbox-agent-readonly-mount-root",
        action="append",
        default=[],
        dest="sandbox_agent_readonly_mount_roots",
        help="Agent-phase root: host directory whose subdirectories leases may "
        "mount read-only when the sandbox starts (repeatable); none disables. "
        "Must not contain a verifier-phase root",
    )
    parser.add_argument(
        "--sandbox-audit-log",
        choices=("on", "off"),
        default="on",
        help="Record every sandbox command and file write in per-worker JSONL "
        "files under <log-dir>/audit",
    )
    parser.add_argument(
        "--sandbox-audit-log-max-mb",
        type=int,
        default=DEFAULT_AUDIT_MAX_MB,
        help="Rotate a worker's audit file at this size (one older file is kept)",
    )
    add_egress_arguments(parser, "sandbox-internet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--advertise-host")
    parser.add_argument(
        "--agent-advertise-mode",
        choices=("fqdn", "hostname", "route-ip"),
        default="fqdn",
        help="Address agents publish to clients (default: fqdn)",
    )
    parser.add_argument(
        "--cpu-allocation-granularity",
        type=int,
        default=1,
        help="Round each sandbox's worker allocation to this many Slurm CPUs",
    )
    parser.add_argument("--registry")
    parser.add_argument("--state-file")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--account", required=True)
    parser.add_argument(
        "--qos",
        help=(
            "Optional Slurm QOS for worker submissions. Required on clusters that "
            "route jobs by QOS name and ignore --partition."
        ),
    )
    parser.add_argument("--time-limit", default="01:00:00")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--agent-python", default=sys.executable)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--log-dir", required=True)
    add_sandbox_arguments(parser)
    parser.add_argument("--start-concurrency", type=int, default=2)
    parser.add_argument("--start-jitter-seconds", type=float, default=5.0)
    parser.add_argument("--max-capacity", type=int, default=4096)
    parser.add_argument("--max-cpus-per-sandbox", type=int, default=64)
    parser.add_argument("--max-memory-mb-per-sandbox", type=int, default=262144)
    _add_gpu_parser_arguments(parser)
    parser.add_argument("--reconcile-interval", type=int, default=5)
    parser.add_argument("--agent-stale-seconds", type=int, default=30)
    parser.add_argument("--agent-lost-seconds", type=int, default=60)
    parser.add_argument("--submission-grace-seconds", type=int, default=30)
    parser.add_argument("--min-scheduler-poll-seconds", type=int, default=15)
    parser.add_argument("--min-submit-interval-seconds", type=int, default=10)
    parser.add_argument("--max-workers-per-scale", type=int, default=64)
    parser.add_argument("--max-scale-failures", type=int, default=5)
    parser.add_argument("--request-retention-seconds", type=int, default=3600)
    parser.add_argument("--scale-up-debounce-seconds", type=float, default=2.0)
    parser.add_argument("--step-start-rate", type=float, default=8.0)
    parser.add_argument("--step-start-burst", type=int, default=8)
    parser.add_argument("--lease-ttl-seconds", type=int, default=3600)
    parser.add_argument("--max-pools", type=int, default=32)
    parser.add_argument("--max-pending-requests", type=int, default=8192)
    parser.add_argument(
        "--enable-dynamic-resources",
        action="store_true",
        help="Allow clients to request policy-bounded CPU and memory shapes",
    )
    parser.add_argument("--dynamic-pool-slots-per-job", type=int, default=16)
    parser.add_argument("--dynamic-worker-max-cpus", type=int)
    parser.add_argument("--dynamic-worker-max-memory-mb", type=int)
    parser.add_argument("--dynamic-pool-max-capacity", type=int, default=512)
    parser.add_argument("--dynamic-pool-warm-capacity", type=int, default=0)
    parser.add_argument("--dynamic-pool-idle-timeout-seconds", type=int, default=600)
    parser.add_argument("--max-dynamic-pools", type=int, default=8)
    parser.add_argument("--cleanup-on-exit", action="store_true")
    return parser


def load_tokens(parser: argparse.ArgumentParser) -> tuple[str, str, str, str]:
    service_token_file = os.environ.get("SANDFLEET_TOKEN_FILE")
    if not service_token_file:
        parser.error("SANDFLEET_TOKEN_FILE must point to a worker-readable token file")

    try:
        token = secret_from_env("SANDFLEET_TOKEN")
        client_token = secret_from_env("SANDFLEET_CLIENT_TOKEN")
        metrics_token = secret_from_env("SANDFLEET_METRICS_TOKEN")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not token:
        parser.error("SANDFLEET_TOKEN_FILE is empty")
    if not client_token or not metrics_token:
        parser.error("SANDFLEET_CLIENT_TOKEN and SANDFLEET_METRICS_TOKEN are required")
    return token, client_token, metrics_token, service_token_file


def build_state(
    args: argparse.Namespace,
    *,
    url: str,
    token: str,
    client_token: str,
    metrics_token: str,
    service_token_file: str,
) -> ControllerState:
    gpu_type_catalog = None
    if args.gpu_type_catalog:
        catalog_path = Path(args.gpu_type_catalog)
        try:
            catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not load GPU type catalog: {error}") from error
        gpu_type_catalog = GPUTypeCatalog.from_payload(catalog_payload)
    dynamic_pool_policy = (
        DynamicPoolPolicy(
            max_cpus_per_worker=args.dynamic_worker_max_cpus,
            max_memory_mb_per_worker=args.dynamic_worker_max_memory_mb,
            slots_per_job=args.dynamic_pool_slots_per_job,
            max_capacity=args.dynamic_pool_max_capacity,
            warm_capacity=args.dynamic_pool_warm_capacity,
            idle_timeout_seconds=args.dynamic_pool_idle_timeout_seconds,
            max_pools=args.max_dynamic_pools,
            cpu_allocation_granularity=args.cpu_allocation_granularity,
            max_gpus_per_worker=args.dynamic_worker_max_gpus,
        )
        if args.enable_dynamic_resources
        else None
    )
    settings = SlurmSettings(
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        agent_advertise_mode=args.agent_advertise_mode,
        cpu_allocation_granularity=args.cpu_allocation_granularity,
        time_limit=args.time_limit,
        source_root=str(Path(args.source_root).resolve()),
        agent_python=args.agent_python,
        log_dir=str(Path(args.log_dir).resolve()),
        audit_log_dir=(
            str(Path(args.log_dir).resolve() / "audit")
            if args.sandbox_audit_log == "on"
            else None
        ),
        audit_log_max_mb=args.sandbox_audit_log_max_mb,
        apptainer_binary=args.apptainer_binary,
        network_mode=args.network_mode,
        gpu_container_mode=args.gpu_container_mode,
        # A worker that cannot reach the controller for a lease TTL holds no valid lease.
        controller_loss_timeout_seconds=args.lease_ttl_seconds,
        worker_python=args.worker_python,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        max_storage_mb=args.max_storage_mb,
        readonly_mount_roots=tuple(args.sandbox_readonly_mount_roots),
        agent_readonly_mount_roots=tuple(args.sandbox_agent_readonly_mount_roots),
        internet=egress_settings_from_args(args, "sandbox-internet"),
        start_concurrency=args.start_concurrency,
        start_jitter_seconds=args.start_jitter_seconds,
    )
    launcher = SlurmLauncher(
        settings,
        service_token_file=service_token_file,
        controller_url=url,
        registry_path=str(Path(args.registry).resolve()) if args.registry else None,
    )
    return ControllerState(
        token=token,
        client_token=client_token,
        metrics_token=metrics_token,
        launcher=launcher,
        max_capacity=args.max_capacity,
        max_cpus_per_sandbox=args.max_cpus_per_sandbox,
        max_memory_mb_per_sandbox=args.max_memory_mb_per_sandbox,
        max_gpus_per_sandbox=args.max_gpus_per_sandbox,
        state_file=args.state_file,
        agent_stale_seconds=args.agent_stale_seconds,
        agent_lost_seconds=args.agent_lost_seconds,
        submission_grace_seconds=args.submission_grace_seconds,
        min_scheduler_poll_seconds=args.min_scheduler_poll_seconds,
        min_submit_interval_seconds=args.min_submit_interval_seconds,
        max_workers_per_scale=args.max_workers_per_scale,
        max_scale_failures=args.max_scale_failures,
        request_retention_seconds=args.request_retention_seconds,
        scale_up_debounce_seconds=args.scale_up_debounce_seconds,
        step_start_rate=args.step_start_rate,
        step_start_burst=args.step_start_burst,
        lease_ttl_seconds=args.lease_ttl_seconds,
        max_pools=args.max_pools,
        max_pending_requests=args.max_pending_requests,
        dynamic_pool_policy=dynamic_pool_policy,
        gpu_type_catalog=gpu_type_catalog,
        gpu_device_isolation_verified=args.gpu_device_isolation_verified,
        gpu_capability_refresh_seconds=args.gpu_capability_refresh_seconds,
    )


def write_registry(path: str, url: str) -> None:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = registry_path.with_suffix(registry_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump({"url": url, "pid": os.getpid(), "started_at": time.time()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(registry_path)


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.enable_dynamic_resources and (
        args.dynamic_worker_max_cpus is None
        or args.dynamic_worker_max_memory_mb is None
    ):
        parser.error(
            "--enable-dynamic-resources requires --dynamic-worker-max-cpus "
            "and --dynamic-worker-max-memory-mb"
        )
    if bool(args.gpu_type_catalog) != args.gpu_device_isolation_verified:
        parser.error(
            "--gpu-type-catalog and --gpu-device-isolation-verified must be set together"
        )
    if args.gpu_type_catalog and (
        not args.enable_dynamic_resources
        or args.max_gpus_per_sandbox < 1
        or args.dynamic_worker_max_gpus < 1
    ):
        parser.error(
            "--gpu-type-catalog requires dynamic resources plus positive "
            "--max-gpus-per-sandbox and --dynamic-worker-max-gpus limits"
        )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    token, client_token, metrics_token, service_token_file = load_tokens(parser)
    advertised_host = args.advertise_host or socket.getfqdn()
    url = f"http://{advertised_host}:{args.port}"
    state = build_state(
        args,
        url=url,
        token=token,
        client_token=client_token,
        metrics_token=metrics_token,
        service_token_file=service_token_file,
    )
    server = ThreadingHTTPServer((args.host, args.port), ControllerHandler)
    server.state = state  # type: ignore[attr-defined]
    server.log_callback = lambda message: print(message, flush=True)  # type: ignore[attr-defined]

    if args.registry:
        write_registry(args.registry, url)
    print(f"Sandfleet controller ready at {url}", flush=True)
    state.start_reconciler(
        args.reconcile_interval, on_error=lambda _error: server.shutdown()
    )

    def shutdown(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    cleanup_error: BaseException | None = None
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        try:
            state.close(
                cleanup_jobs=args.cleanup_on_exit and state.reconcile_error is None
            )
        except BaseException as error:
            cleanup_error = error
        finally:
            server.server_close()
            if args.registry:
                with contextlib.suppress(FileNotFoundError):
                    Path(args.registry).unlink()
    if state.reconcile_error is not None:
        if cleanup_error is not None:
            state.reconcile_error.add_note(
                f"Controller shutdown also failed: {cleanup_error}"
            )
        raise state.reconcile_error
    if cleanup_error is not None:
        raise cleanup_error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
