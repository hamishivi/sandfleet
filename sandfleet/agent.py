"""Sandfleet node agent running inside a Slurm worker allocation."""

from __future__ import annotations

import argparse
import contextlib
import hmac
import os
import random
import secrets
import signal
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from sandfleet.audit import DEFAULT_AUDIT_MAX_MB, AuditLog, truncate_utf8
from sandfleet.errors import LeaseCleanupPending, NoCapacityError, SandfleetError
from sandfleet.recovery import controller_url as discover_controller, transient
from sandfleet.http import (
    API_VERSION,
    JsonHandler,
    decode_bytes,
    encode_bytes,
    request_json,
    secret_from_env,
)
from sandfleet.egress import (
    EgressForwarder,
    add_egress_arguments,
    egress_settings_from_args,
)
from sandfleet.mounts import validate_mount_request, validate_phase_roots
from sandfleet.step_backend import SlurmStepSandbox
from sandfleet.step_worker import env_enabled
from sandfleet.network import (
    NETWORK_MODES,
    network_mode_argument,
    relay_pending_limit,
    validate_network_config,
)
from sandfleet.resources import GPU_CONTAINER_MODES
from sandfleet.version import VERSION

RELEASE_IDLE_GRACE_SECONDS = 1.0
RELEASE_DRAIN_SECONDS = 20.0
# Lease operations recorded in the audit log; reads and status are not.
AUDITED_OPERATIONS = frozenset(
    {
        "start",
        "restart",
        "exec",
        "exec-env",
        "write-file",
        "put-archive",
        "kill-processes",
        "add-mounts",
    }
)


def _b64_size(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    return len(value) * 3 // 4 - value[-2:].count("=")


def _audited_mounts(mounts: Any) -> Any:
    try:
        return validate_mount_request(mounts)
    except ValueError:
        # Record what was refused, bounded like a command.
        return truncate_utf8(repr(mounts))[0]


def audit_fields(
    operation: str, payload: dict[str, Any], result: dict[str, Any] | None
) -> dict[str, Any]:
    """Describe one lease operation without its file contents or env values."""
    if operation in {"start", "restart"}:
        fields = {
            "image": payload.get("image"),
            "storage_mb": payload.get("storage_mb"),
        }
        if "readonly_mounts" in payload:
            fields["readonly_mounts"] = _audited_mounts(payload["readonly_mounts"])
        return fields
    if operation == "write-file":
        return {
            "path": payload.get("path"),
            "bytes": _b64_size(payload.get("content_b64")),
            "offset": payload.get("offset"),
        }
    if operation == "put-archive":
        return {
            "root": payload.get("root"),
            "bytes": _b64_size(payload.get("content_b64")),
        }
    if operation == "kill-processes":
        result = result or {}
        return {
            "killed": result.get("killed"),
            "cancelled_commands": result.get("cancelled_commands"),
        }
    if operation == "add-mounts":
        return {
            "mounts": _audited_mounts(payload.get("mounts")),
            "stopped_processes": (result or {}).get("stopped_processes"),
        }
    command, truncated = truncate_utf8(str(payload.get("command")))
    fields: dict[str, Any] = {
        "command": command,
        "command_truncated": truncated,
        "command_id": payload.get("command_id"),
        "timeout": payload.get("timeout"),
    }
    env = payload.get("env")
    if isinstance(env, dict):
        fields["env_keys"] = sorted(str(key) for key in env)
    if result is not None:
        for key in ("exit_code", "cancelled", "timed_out"):
            fields[key] = result.get(key)
    return fields


@dataclass
class LeaseRecord:
    lease_id: str
    token: str
    slot_id: int
    proxy_url: str | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    interrupt_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class PreparedSlot:
    slot_id: int
    status: str = "empty"
    sandbox: SlurmStepSandbox | None = None
    lease_id: str | None = None
    failures: int = 0
    next_prepare_at: float = 0.0
    last_error: str | None = None


def job_end_time(environ: dict[str, str] | None = None) -> float | None:
    """This worker job's scheduled end (Unix seconds) from SLURM_JOB_END_TIME.

    Slurm sets it in the job's environment when the job starts, so the worker
    never calls ``squeue``. A later ``scontrol update TimeLimit`` is not
    reflected. None outside a job, for an unlimited job, or on a Slurm that does
    not set it.
    """

    raw = (os.environ if environ is None else environ).get("SLURM_JOB_END_TIME", "")
    raw = raw.strip()
    return float(raw) if raw.isdigit() and int(raw) > 0 else None


class AgentState:
    def __init__(
        self,
        *,
        service_token: str,
        slots: int,
        cpus: int,
        memory_mb: int,
        apptainer_binary: str,
        worker_python: str,
        cache_dir: str | None,
        tmp_dir: str | None,
        start_concurrency: int,
        gpus: int = 0,
        gpu_type: str | None = None,
        start_jitter_seconds: float = 0.0,
        allocation_cpus: frozenset[int] | None = None,
        allocation_cgroup: str | None = None,
        max_storage_mb: int = 0,
        network_mode: str = "loopback",
        gpu_container_mode: str = "auto",
        egress: EgressForwarder | None = None,
        audit: AuditLog | None = None,
        readonly_mount_roots: tuple[str, ...] = (),
        agent_readonly_mount_roots: tuple[str, ...] = (),
        job_end_time: float | None = None,
    ):
        validate_network_config(network_mode)
        self.agent_readonly_mount_roots, self.readonly_mount_roots = (
            validate_phase_roots(agent_readonly_mount_roots, readonly_mount_roots)
        )
        self.service_token = service_token
        self.slots = slots
        self.cpus = cpus
        self.memory_mb = memory_mb
        if gpus < 0 or (gpus == 0) != (gpu_type is None):
            raise ValueError("GPU count and type must be configured together")
        self.gpus = gpus
        self.gpu_type = gpu_type
        self.apptainer_binary = apptainer_binary
        self.worker_python = worker_python
        self.cache_dir = cache_dir
        self.tmp_dir = tmp_dir
        self.max_storage_mb = max_storage_mb
        self.network_mode = network_mode
        self.gpu_container_mode = gpu_container_mode
        self.egress = egress
        self.audit = audit
        # Reported to lease holders (worker_end_time) so a caller can warn when a
        # long trial lands on a worker whose Slurm job ends first.
        self.job_end_time = job_end_time
        self.leases: dict[str, LeaseRecord] = {}
        self.prepared_slots = {
            slot_id: PreparedSlot(slot_id) for slot_id in range(slots)
        }
        self.lock = threading.RLock()
        self.start_concurrency = start_concurrency
        self.start_jitter_seconds = start_jitter_seconds
        self.allocation_cpus = allocation_cpus
        self.allocation_cgroup = allocation_cgroup
        self.stop_event = threading.Event()
        self.prepare_wakeup = threading.Event()
        self.prepare_threads: list[threading.Thread] = []
        self.metrics = {
            "step_prepare_attempts": 0,
            "step_prepare_successes": 0,
            "step_prepare_failures": 0,
            "step_recycles": 0,
            "step_recycle_failures": 0,
        }

    @staticmethod
    def _retry_at(failures: int) -> float:
        return time.monotonic() + min(60.0, 2 ** min(failures, 6))

    def _fail_slot_locked(self, slot: PreparedSlot, error: Exception | str) -> None:
        slot.sandbox = None
        slot.status = "failed"
        slot.failures += 1
        slot.last_error = str(error)
        slot.next_prepare_at = self._retry_at(slot.failures)

    def _validate_step_isolation_locked(self, sandbox: SlurmStepSandbox) -> None:
        if self.allocation_cpus is not None:
            if not sandbox.cgroup_path or sandbox.cgroup_path == self.allocation_cgroup:
                raise RuntimeError("Slurm step does not have a distinct cgroup")
            outside = sandbox.cpuset - self.allocation_cpus
            if outside:
                raise RuntimeError(
                    f"Slurm step CPU set escapes the worker allocation: {sorted(outside)}"
                )
        for slot in self.prepared_slots.values():
            if slot.sandbox is None:
                continue
            cpu_overlap = sandbox.cpuset & slot.sandbox.cpuset
            if self.allocation_cpus is not None and cpu_overlap:
                raise RuntimeError(
                    f"Slurm step CPU set overlaps slot {slot.slot_id}: {sorted(cpu_overlap)}"
                )
            gpu_overlap = sandbox.gpu_devices & slot.sandbox.gpu_devices
            if gpu_overlap:
                raise RuntimeError(
                    f"Slurm step GPU devices overlap slot {slot.slot_id}: {sorted(gpu_overlap)}"
                )

    def capacity(self) -> dict[str, Any]:
        with self.lock:
            ready = sum(
                slot.status in {"ready", "leased"}
                for slot in self.prepared_slots.values()
            )
            free = sum(
                slot.status == "ready" and slot.lease_id is None
                for slot in self.prepared_slots.values()
            )
            preparing = sum(
                slot.status == "preparing" for slot in self.prepared_slots.values()
            )
            failed = sum(
                slot.status == "failed" for slot in self.prepared_slots.values()
            )
            slot_errors = {
                str(slot.slot_id): slot.last_error
                for slot in self.prepared_slots.values()
                if slot.last_error is not None
            }
            return {
                "slots": self.slots,
                "ready_slots": ready,
                "free_slots": free,
                "preparing_slots": preparing,
                "failed_slots": failed,
                "gpus": self.gpus,
                "gpu_type": self.gpu_type,
                "slot_errors": slot_errors,
                "metrics": {
                    **self.metrics,
                    **({} if self.egress is None else self.egress.metrics_snapshot()),
                    **(
                        {}
                        if self.audit is None
                        else {"audit_write_failures": self.audit.failures}
                    ),
                },
            }

    def prepare_one(self, acquire_permit: Callable[[], None]) -> bool:
        now = time.monotonic()
        with self.lock:
            candidate = next(
                (
                    slot
                    for slot in self.prepared_slots.values()
                    if slot.status in {"empty", "failed"}
                    and slot.next_prepare_at <= now
                ),
                None,
            )
            if candidate is None:
                return False
            candidate.status = "preparing"
            self.metrics["step_prepare_attempts"] += 1
        sandbox: SlurmStepSandbox | None = None
        try:
            acquire_permit()
            if self.start_jitter_seconds:
                time.sleep(random.uniform(0.0, self.start_jitter_seconds))
            sandbox = self.new_sandbox()
            sandbox.prepare()
            with self.lock:
                if self.stop_event.is_set():
                    raise RuntimeError("Agent is stopping")
                self._validate_step_isolation_locked(sandbox)
                candidate.sandbox = sandbox
                candidate.status = "ready"
                candidate.failures = 0
                candidate.last_error = None
                self.metrics["step_prepare_successes"] += 1
            return True
        except (OSError, RuntimeError) as error:
            if sandbox is not None:
                sandbox.discard()
            with self.lock:
                self._fail_slot_locked(candidate, error)
                self.metrics["step_prepare_failures"] += 1
            return False

    def _discard_broken_steps(self) -> None:
        broken: list[SlurmStepSandbox] = []
        with self.lock:
            if self.stop_event.is_set():
                # close() owns the steps and removes their directories.
                return
            lost = [
                slot.slot_id
                for slot in self.prepared_slots.values()
                if slot.status == "ready"
                and slot.sandbox is not None
                and slot.sandbox.storage_lost
            ]
            if lost:
                # Slurm removes the job's /tmp during preemption or teardown
                # without signalling this batch step. Every later start would
                # fail, so stop serving instead of recycling doomed slots.
                raise RuntimeError(
                    f"Prepared step storage vanished for slots {lost}; "
                    "the worker allocation is being torn down"
                )
            for slot in self.prepared_slots.values():
                if (
                    slot.status == "ready"
                    and slot.sandbox is not None
                    and not slot.sandbox.healthy
                ):
                    broken.append(slot.sandbox)
                    self._fail_slot_locked(slot, "Slurm step exited unexpectedly")
                    self.metrics["step_prepare_failures"] += 1
        for sandbox in broken:
            sandbox.discard()

    def _prepare_loop(
        self,
        acquire_permit: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        try:
            while not self.stop_event.is_set():
                self._discard_broken_steps()
                if not self.prepare_one(acquire_permit):
                    self.prepare_wakeup.wait(1.0)
                    self.prepare_wakeup.clear()
        except Exception as error:
            self.stop_event.set()
            self.prepare_wakeup.set()
            on_error(error)
            raise

    def start_prewarmer(
        self,
        acquire_permit: Callable[[], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        for index in range(self.start_concurrency):
            thread = threading.Thread(
                target=self._prepare_loop,
                args=(acquire_permit, on_error),
                name=f"sandfleet-prewarm-{index}",
                daemon=True,
            )
            self.prepare_threads.append(thread)
            thread.start()

    def reserve(self, *, internet: bool = False) -> dict[str, Any]:
        if type(internet) is not bool:
            raise ValueError("internet must be a JSON boolean")
        if internet and self.egress is None:
            raise ValueError("Sandbox internet is not enabled on this worker")
        with self.lock:
            if self.stop_event.is_set():
                raise NoCapacityError("Agent is stopping")
            slot = next(
                (
                    slot
                    for slot in self.prepared_slots.values()
                    if slot.status == "ready" and slot.lease_id is None
                ),
                None,
            )
            if slot is None:
                raise NoCapacityError("Agent has no free sandbox slots")
            lease_id = uuid.uuid4().hex
            token = secrets.token_urlsafe(32)
            slot.status = "leased"
            slot.lease_id = lease_id
            self.leases[lease_id] = LeaseRecord(
                lease_id=lease_id,
                token=token,
                slot_id=slot.slot_id,
                proxy_url=self.egress.grant(lease_id) if internet else None,
            )
            return {
                "lease_id": lease_id,
                "lease_token": token,
                "free_slots": self.free_slots,
            }

    def get_lease(self, lease_id: str, token: str) -> LeaseRecord:
        with self.lock:
            lease = self.leases.get(lease_id)
        if lease is None:
            raise KeyError(f"Unknown lease {lease_id!r}")
        if not hmac.compare_digest(token, lease.token):
            raise PermissionError("Invalid lease token")
        return lease

    def sandbox_for(self, lease: LeaseRecord) -> SlurmStepSandbox:
        with self.lock:
            slot = self.prepared_slots[lease.slot_id]
            if (
                slot.lease_id != lease.lease_id
                or slot.sandbox is None
                or slot.status != "leased"
            ):
                raise RuntimeError("Lease has lost its prepared Slurm step")
            return slot.sandbox

    def mark_broken(self, lease: LeaseRecord, error: Exception) -> None:
        with self.lock:
            slot = self.prepared_slots[lease.slot_id]
            if slot.lease_id == lease.lease_id:
                slot.last_error = str(error)
                if slot.sandbox is None or not slot.sandbox.healthy:
                    slot.status = "broken"

    @contextlib.contextmanager
    def _release_lock(self, lease: LeaseRecord):
        if not lease.lock.acquire(timeout=RELEASE_IDLE_GRACE_SECONDS):
            # An expired/discarded client cannot cancel its own in-flight RPC.
            # Interrupt this lease out of band, then wait for the RPC to drain.
            # Idle releases never take this path and retain their prepared step.
            if not lease.interrupt_lock.acquire(timeout=RELEASE_IDLE_GRACE_SECONDS):
                raise LeaseCleanupPending(
                    "Lease interruption is busy; release must be retried"
                )
            try:
                with self.lock:
                    current = self.leases.get(lease.lease_id)
                    sandbox = (
                        self.prepared_slots[lease.slot_id].sandbox
                        if current is lease
                        else None
                    )
                if sandbox is not None:
                    sandbox.abort()
            finally:
                lease.interrupt_lock.release()
            if not lease.lock.acquire(timeout=RELEASE_DRAIN_SECONDS):
                raise LeaseCleanupPending(
                    "Lease execution did not drain; retaining lease for retry"
                )
        try:
            if not lease.interrupt_lock.acquire(timeout=RELEASE_IDLE_GRACE_SECONDS):
                raise LeaseCleanupPending(
                    "Lease interruption is busy; release must be retried"
                )
            try:
                yield
            finally:
                lease.interrupt_lock.release()
        finally:
            lease.lock.release()

    def release(self, lease_id: str) -> dict[str, Any]:
        with self.lock:
            lease = self.leases.get(lease_id)
        if lease is None:
            # Release is intentionally idempotent.
            return {"released": False, "free_slots": self.free_slots}
        if lease.proxy_url is not None:
            self.egress.revoke(lease_id)
        with self._release_lock(lease):
            with self.lock:
                if self.leases.get(lease_id) is not lease:
                    return {"released": False, "free_slots": self.free_slots}
                slot = self.prepared_slots[lease.slot_id]
                sandbox = slot.sandbox
            reusable = False
            if sandbox is not None and sandbox.healthy:
                try:
                    sandbox.recycle()
                    reusable = True
                except (OSError, RuntimeError) as error:
                    with self.lock:
                        slot.last_error = str(error)
                        self.metrics["step_recycle_failures"] += 1
                    if self.max_storage_mb:
                        # A failed disk stop must not unlink a possibly mounted
                        # image or lose the lease needed to retry cleanup.
                        raise
            if not reusable and sandbox is not None:
                sandbox.discard()
            with self.lock:
                self.leases.pop(lease_id, None)
                slot.lease_id = None
                if reusable:
                    slot.status = "ready"
                    self.metrics["step_recycles"] += 1
                else:
                    self._fail_slot_locked(
                        slot, slot.last_error or "Slurm step is unavailable"
                    )
                    self.prepare_wakeup.set()
        return {"released": True, "free_slots": self.free_slots}

    def audit_operation(
        self,
        lease: LeaseRecord,
        operation: str,
        payload: dict[str, Any],
        started: float,
        *,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if self.audit is None or operation not in AUDITED_OPERATIONS:
            return
        record = {
            "op": operation,
            "lease_id": lease.lease_id,
            "slot": lease.slot_id,
            "internet": lease.proxy_url is not None,
            "duration_s": round(time.monotonic() - started, 3),
            **audit_fields(operation, payload, result),
        }
        if error is not None:
            record["error"] = truncate_utf8(f"{type(error).__name__}: {error}")[0]
        self.audit.record(record)

    def abort(self, lease: LeaseRecord) -> None:
        # Serialize with release/reassignment, but never wait on the exec lock.
        # Otherwise an abort racing release could hit the next slot occupant.
        with lease.interrupt_lock:
            self.sandbox_for(lease).abort()

    def cancel_command(self, lease: LeaseRecord, command_id: str) -> dict[str, Any]:
        # Same locking as abort().
        with lease.interrupt_lock:
            return self.sandbox_for(lease).cancel_command(command_id)

    def kill_processes(self, lease: LeaseRecord) -> dict[str, Any]:
        # Same locking as abort(): an active exec does not delay the kill.
        started = time.monotonic()
        try:
            with lease.interrupt_lock:
                result = self.sandbox_for(lease).kill_processes()
        except Exception as error:
            self.audit_operation(lease, "kill-processes", {}, started, error=error)
            raise
        self.audit_operation(lease, "kill-processes", {}, started, result=result)
        return result

    @property
    def free_slots(self) -> int:
        return int(self.capacity()["free_slots"])

    def close(self) -> None:
        self.stop_event.set()
        self.prepare_wakeup.set()
        for thread in self.prepare_threads:
            thread.join(timeout=2)
        with self.lock:
            sandboxes = [
                slot.sandbox
                for slot in self.prepared_slots.values()
                if slot.sandbox is not None
            ]
            self.leases.clear()
        if self.egress is not None:
            self.egress.close()
        for sandbox in sandboxes:
            with contextlib.suppress(Exception):
                sandbox.close()

    def new_sandbox(self) -> SlurmStepSandbox:
        return SlurmStepSandbox(
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            gpus=self.gpus,
            gpu_type=self.gpu_type,
            apptainer_binary=self.apptainer_binary,
            worker_python=self.worker_python,
            cache_dir=self.cache_dir,
            tmp_dir=self.tmp_dir,
            max_storage_mb=self.max_storage_mb,
            network_mode=self.network_mode,
            gpu_container_mode=self.gpu_container_mode,
            # This worker's sandboxes share its forwarder's unauthenticated slots.
            relay_max_pending=relay_pending_limit(self.slots),
            readonly_mount_roots=self.readonly_mount_roots,
            agent_readonly_mount_roots=self.agent_readonly_mount_roots,
        )


class AgentHandler(JsonHandler):
    @property
    def state(self) -> AgentState:
        return self.server.state  # type: ignore[attr-defined, no-any-return]

    def _control_auth(self) -> None:
        self.require_token(self.state.service_token)

    def _lease(self, lease_id: str) -> LeaseRecord:
        return self.state.get_lease(lease_id, self.bearer_token())

    def _worker_end_time(self) -> dict[str, float]:
        """Optional ``worker_end_time`` for start/restart/status responses."""
        end = self.state.job_end_time
        return {} if end is None else {"worker_end_time": end}

    @staticmethod
    def _exec(
        sandbox: SlurmStepSandbox,
        operation: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if operation == "exec-env":
            options["env"] = payload["env"]
        if payload.get("command_id") is not None:
            options["command_id"] = str(payload["command_id"])
        return sandbox.run_command(
            str(payload["command"]), payload.get("timeout"), **options
        )

    @staticmethod
    def _optional_start_options(payload: dict[str, Any]) -> dict[str, Any]:
        """Disk and agent-phase mount options; old clients send neither key."""
        options: dict[str, Any] = {}
        if "storage_mb" in payload:
            options["storage_mb"] = payload["storage_mb"]
            options["image_sha256"] = payload["image_sha256"]
        if "readonly_mounts" in payload:
            options["readonly_mounts"] = validate_mount_request(
                payload["readonly_mounts"]
            )
        return options

    @staticmethod
    def _operate(
        sandbox: SlurmStepSandbox,
        operation: str,
        payload: dict[str, Any],
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        if operation in {"start", "restart"}:
            image = str(payload["image"]).strip()
            if not image:
                raise ValueError("image is required")
            flags = payload.get("extra_start_flags", [])
            if not isinstance(flags, list) or not all(
                isinstance(flag, str) for flag in flags
            ):
                raise ValueError("extra_start_flags must be a list of strings")
            method = sandbox.start if operation == "start" else sandbox.restart
            return method(
                image=image,
                command_timeout=int(payload.get("timeout", 1800)),
                pwd=str(payload.get("pwd", "/workspace")),
                extra_start_flags=flags,
                proxy_url=proxy_url,
                **AgentHandler._optional_start_options(payload),
            )
        if operation in {"exec", "exec-env"}:
            return AgentHandler._exec(sandbox, operation, payload)
        if operation == "add-mounts":
            # Restarts the instance, so it holds the lease lock like restart.
            return sandbox.add_readonly_mounts(
                validate_mount_request(payload["mounts"])
            )
        if operation == "write-file":
            chunk = (
                {
                    "offset": payload["offset"],
                    "truncate": payload.get("truncate", False),
                }
                if "offset" in payload
                else {}
            )
            sandbox.write_file(
                str(payload["path"]), decode_bytes(str(payload["content_b64"])), **chunk
            )
            return {"written": True}
        if operation == "read-file":
            chunk = (
                {"offset": payload["offset"], "size": payload["size"]}
                if "offset" in payload
                else {}
            )
            return {
                "content_b64": encode_bytes(
                    sandbox.read_file(str(payload["path"]), **chunk)
                )
            }
        if operation == "put-archive":
            sandbox.put_archive(
                str(payload["root"]), decode_bytes(str(payload["content_b64"]))
            )
            return {"extracted": True}
        raise KeyError(operation)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parts = self.path_parts()
            if parts == [API_VERSION, "health"]:
                self._control_auth()
                self.send_json(200, {"ok": True, **self.state.capacity()})
                return
            if (
                len(parts) == 4
                and parts[:2] == [API_VERSION, "leases"]
                and parts[3] == "status"
            ):
                lease = self._lease(parts[2])
                with lease.lock:
                    sandbox = self.state.sandbox_for(lease)
                    status = sandbox.status()
                self.send_json(
                    200,
                    {
                        "lease_id": lease.lease_id,
                        "internet": lease.proxy_url is not None,
                        "started": bool(status["instance_name"]),
                        **status,
                        **self._worker_end_time(),
                    },
                )
                return
            raise KeyError(self.path)
        except Exception as error:
            self.send_error_json(error)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parts = self.path_parts()
            if parts == [API_VERSION, "control", "leases"]:
                self._control_auth()
                payload = self.read_json()
                self.send_json(
                    201, self.state.reserve(internet=payload.get("internet", False))
                )
                return
            if len(parts) != 4 or parts[:2] != [API_VERSION, "leases"]:
                raise KeyError(self.path)
            lease = self._lease(parts[2])
            operation = parts[3]
            payload = self.read_json()
            if operation == "abort":
                # Do not take lease.lock: exec holds it while waiting on this step.
                self.state.abort(lease)
                self.send_json(200, {"interrupted": True})
                return
            if operation == "exec-cancel":
                # Like abort, must not take lease.lock.
                result = self.state.cancel_command(lease, str(payload["command_id"]))
                self.send_json(200, result)
                return
            if operation == "kill-processes":
                # Like abort, must not take lease.lock.
                self.send_json(200, self.state.kill_processes(lease))
                return
            with lease.lock:
                sandbox = self.state.sandbox_for(lease)
                started = time.monotonic()
                try:
                    result = self._operate(sandbox, operation, payload, lease.proxy_url)
                except Exception as error:
                    self.state.mark_broken(lease, error)
                    self.state.audit_operation(
                        lease, operation, payload, started, error=error
                    )
                    raise
                self.state.audit_operation(
                    lease, operation, payload, started, result=result
                )
            if operation in {"start", "restart"}:
                result = {**result, **self._worker_end_time()}
            self.send_json(200, result)
        except Exception as error:
            self.send_error_json(error)

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            parts = self.path_parts()
            if len(parts) == 4 and parts[:3] == [API_VERSION, "control", "leases"]:
                self._control_auth()
                self.send_json(200, self.state.release(parts[3]))
                return
            raise KeyError(self.path)
        except Exception as error:
            self.send_error_json(error)


def advertise_host(explicit: str | None, mode: str, controller_url: str) -> str:
    if explicit:
        return explicit
    if mode == "fqdn":
        return socket.getfqdn()
    if mode == "hostname":
        return socket.gethostname()
    target = urlsplit(controller_url)
    if not target.hostname:
        raise ValueError("controller URL has no hostname")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.connect((target.hostname, target.port or 80))
        return str(connection.getsockname()[0])


def current_cgroup_path() -> str:
    paths = [
        line.partition("::")[2]
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        if "::" in line
    ]
    if len(paths) != 1 or not paths[0]:
        raise RuntimeError("Sandfleet requires one unified cgroup v2 hierarchy")
    return str(Path("/sys/fs/cgroup") / paths[0].lstrip("/"))


def register_agent(
    controller_url: str,
    token: str,
    payload: dict[str, Any],
    *,
    timeout: float = 300,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request_json(
                discover_controller(controller_url),
                f"/{API_VERSION}/agents/register",
                method="POST",
                token=token,
                payload=payload,
                timeout=15,
            )
            return
        except SandfleetError as error:
            if error.status is not None and error.status < 500:
                raise
            last_error = error
            time.sleep(2)
    raise RuntimeError(
        f"Could not register with Sandfleet controller: {last_error}"
    ) from last_error


def heartbeat_loop(
    stop: threading.Event,
    controller_url: str,
    token: str,
    agent_id: str,
    state: AgentState,
    registration: dict[str, Any],
    controller_loss_timeout_seconds: float = 3600,
) -> None:
    """Heartbeat until stopped; exit if the controller stays unreachable.

    Leases are only renewed through the controller, so once a lease TTL passes
    without contact no lease on this worker can still be valid. Exiting then
    frees the allocation instead of holding it until the Slurm time limit.
    """
    last_contact = time.monotonic()
    while not stop.wait(5):
        try:
            result = request_json(
                discover_controller(controller_url),
                f"/{API_VERSION}/agents/{agent_id}/heartbeat",
                method="POST",
                token=token,
                payload=state.capacity(),
                timeout=10,
            )
            if not result["ok"]:
                raise RuntimeError(
                    f"Controller rejected agent heartbeat: {result['status']}"
                )
            last_contact = time.monotonic()
        except SandfleetError as error:
            if error.status == 404:
                register_agent(
                    discover_controller(controller_url),
                    token,
                    {**registration, **state.capacity()},
                    timeout=30,
                )
                last_contact = time.monotonic()
            elif error.status is not None and error.status < 500:
                raise
        lost_for = time.monotonic() - last_contact
        if 0 < controller_loss_timeout_seconds < lost_for:
            raise RuntimeError(
                f"No successful controller heartbeat for {lost_for:.0f}s; exiting"
            )


def step_permit_acquirer(
    controller_url: str, token: str, stop: threading.Event
) -> Callable[[], None]:
    """Return a callback that obtains one globally rate-limited srun permit."""

    def acquire() -> None:
        outage_deadline: float | None = None
        while not stop.is_set():
            try:
                result = request_json(
                    discover_controller(controller_url),
                    f"/{API_VERSION}/step-permits",
                    method="POST",
                    token=token,
                    payload={},
                    timeout=10,
                )
            except SandfleetError as error:
                if not transient(error):
                    raise
                # An ambiguous permit response can waste a token, never launch
                # a step without a fresh grant or bypass the global rate limit.
                now = time.monotonic()
                if outage_deadline is None:
                    outage_deadline = now + 300
                if now >= outage_deadline:
                    raise
                stop.wait(min(2, outage_deadline - now))
                continue
            # A healthy denial is rate limiting, not controller downtime.
            outage_deadline = None
            if result["granted"]:
                return
            stop.wait(max(0.05, float(result["retry_after_seconds"])))
        raise RuntimeError("Agent stopped while waiting for a step permit")

    return acquire


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-url", required=True)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--memory-mb", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--gpu-type")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--advertise-host")
    parser.add_argument(
        "--advertise-mode",
        choices=("fqdn", "hostname", "route-ip"),
        default="fqdn",
    )
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
    parser.add_argument(
        "--controller-loss-timeout-seconds",
        type=float,
        default=3600,
        help="Exit after this long without a successful heartbeat (0 disables)",
    )
    parser.add_argument(
        "--worker-python",
        default=os.environ.get("SANDFLEET_WORKER_PYTHON") or sys.executable,
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--tmp-dir")
    parser.add_argument("--max-storage-mb", type=int, default=0)
    parser.add_argument(
        "--readonly-mount-root",
        action="append",
        default=[],
        dest="readonly_mount_roots",
        help="Host directory whose subdirectories leases may add as read-only mounts",
    )
    parser.add_argument(
        "--agent-readonly-mount-root",
        action="append",
        default=[],
        dest="agent_readonly_mount_roots",
        help="Host directory whose subdirectories leases may mount read-only at start",
    )
    parser.add_argument("--start-concurrency", type=int, default=2)
    parser.add_argument("--start-jitter-seconds", type=float, default=5.0)
    parser.add_argument(
        "--audit-log-dir",
        help="Append a JSONL record of every sandbox command and file write here",
    )
    parser.add_argument("--audit-log-max-mb", type=int, default=DEFAULT_AUDIT_MAX_MB)
    add_egress_arguments(parser, "internet")
    return parser


def load_service_token(parser: argparse.ArgumentParser) -> str:
    try:
        token = secret_from_env("SANDFLEET_TOKEN")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not token:
        parser.error("SANDFLEET_TOKEN must be set")
    return token


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (
        args.slots < 1
        or args.cpus < 1
        or args.memory_mb < 1
        or args.gpus < 0
        or args.start_concurrency < 1
        or args.max_storage_mb < 0
        or args.audit_log_max_mb < 1
    ):
        parser.error(
            "slots, cpus, memory-mb, start-concurrency and audit-log-max-mb must be "
            "positive"
        )
    if (args.gpus == 0) != (args.gpu_type is None):
        parser.error("gpus and gpu-type must be configured together")
    try:
        validate_phase_roots(args.agent_readonly_mount_roots, args.readonly_mount_roots)
    except ValueError as error:
        parser.error(str(error))
    if args.internet != "off" and not env_enabled("SANDFLEET_SANDBOX_DROP_NET_CAPS"):
        parser.error(
            "--internet requires SANDFLEET_SANDBOX_DROP_NET_CAPS: with CAP_NET_ADMIN "
            "a sandbox could route around the egress forwarder"
        )


def build_egress(args: argparse.Namespace) -> EgressForwarder | None:
    settings = egress_settings_from_args(args, "internet")
    if not settings.enabled:
        return None
    # Step workers on this node relay internet sandboxes' 127.0.0.1 proxy port
    # to the forwarder from the host namespace, so it binds the host's loopback
    # and no other machine can reach it.
    host = "127.0.0.1"
    forwarder = EgressForwarder(
        settings, host=host, log=lambda message: print(message, flush=True)
    )
    forwarder.start()
    print(
        f"Sandfleet egress forwarder ({args.internet}) at {host}:{forwarder.port}",
        flush=True,
    )
    return forwarder


def build_audit(args: argparse.Namespace, agent_id: str) -> AuditLog | None:
    if not args.audit_log_dir:
        return None
    return AuditLog(
        Path(args.audit_log_dir) / f"audit-{agent_id}.jsonl",
        max_bytes=args.audit_log_max_mb * 1024 * 1024,
        context={
            "pool": args.pool,
            "agent_id": agent_id,
            "job_id": os.getenv("SLURM_JOB_ID"),
            "node": os.getenv("SLURMD_NODENAME") or socket.gethostname(),
        },
    )


def build_state(args: argparse.Namespace, token: str, agent_id: str) -> AgentState:
    return AgentState(
        service_token=token,
        slots=args.slots,
        cpus=args.cpus,
        memory_mb=args.memory_mb,
        gpus=args.gpus,
        gpu_type=args.gpu_type,
        apptainer_binary=args.apptainer_binary,
        network_mode=args.network_mode,
        gpu_container_mode=args.gpu_container_mode,
        worker_python=args.worker_python,
        cache_dir=args.cache_dir,
        tmp_dir=args.tmp_dir,
        max_storage_mb=args.max_storage_mb,
        start_concurrency=args.start_concurrency,
        start_jitter_seconds=args.start_jitter_seconds,
        allocation_cpus=frozenset(os.sched_getaffinity(0)),
        allocation_cgroup=current_cgroup_path(),
        egress=build_egress(args),
        audit=build_audit(args, agent_id),
        readonly_mount_roots=tuple(args.readonly_mount_roots),
        agent_readonly_mount_roots=tuple(args.agent_readonly_mount_roots),
        job_end_time=job_end_time(),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    token = load_service_token(parser)
    validate_args(parser, args)
    agent_id = f"{os.getenv('SLURM_JOB_ID', 'local')}-{uuid.uuid4().hex[:8]}"
    state = build_state(args, token, agent_id)
    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    server.state = state  # type: ignore[attr-defined]
    server.log_callback = lambda message: print(message, flush=True)  # type: ignore[attr-defined]
    port = int(server.server_address[1])
    host = advertise_host(args.advertise_host, args.advertise_mode, args.controller_url)
    url = f"http://{host}:{port}"
    registration = {
        "agent_id": agent_id,
        "pool": args.pool,
        "url": url,
        "slots": args.slots,
        "ready_slots": 0,
        "free_slots": 0,
        "preparing_slots": 0,
        "failed_slots": 0,
        "gpus": args.gpus,
        "gpu_type": args.gpu_type,
        "internet": state.egress is not None,
        "version": VERSION,
        "job_id": os.getenv("SLURM_JOB_ID"),
        "array_job_id": os.getenv("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "node": os.getenv("SLURMD_NODENAME") or socket.gethostname(),
    }
    register_agent(args.controller_url, token, registration)
    print(
        f"Sandfleet agent {agent_id} ready at {url} with {args.slots} slots", flush=True
    )

    stop = threading.Event()
    fatal_errors: list[Exception] = []

    def report_fatal(error: Exception) -> None:
        fatal_errors.append(error)
        stop.set()
        state.stop_event.set()
        server.shutdown()

    def supervised_heartbeat() -> None:
        try:
            heartbeat_loop(
                stop,
                args.controller_url,
                token,
                agent_id,
                state,
                registration,
                args.controller_loss_timeout_seconds,
            )
        except Exception as error:
            report_fatal(error)
            raise

    heartbeat = threading.Thread(
        target=supervised_heartbeat,
        daemon=True,
    )
    heartbeat.start()
    state.start_prewarmer(
        step_permit_acquirer(args.controller_url, token, state.stop_event),
        report_fatal,
    )

    def shutdown(_signum: int, _frame: object) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        stop.set()
        state.close()
        server.server_close()
        heartbeat.join(timeout=10)
    if fatal_errors:
        raise fatal_errors[0]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
