"""Python client for the Sandfleet controller and node-agent data plane."""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Any

from sandfleet.errors import SandfleetError, SandboxLostError
from sandfleet.environment import validate_command_env
from sandfleet.http import API_VERSION, decode_bytes, encode_bytes, request_json
from sandfleet.resources import GPURequest
from sandfleet.recovery import controller_request

# The worker bounds the restart behind add-mounts at 900 s; the rest is transport.
ADD_MOUNTS_TIMEOUT = 960


@dataclass(frozen=True)
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    output_truncated: bool = False
    cancelled: bool = False
    timed_out: bool | None = None


def validate_disk_status(
    storage: dict[str, Any] | None, *, storage_mb: int, image_sha256: str | None
) -> None:
    """Require the requested writable-volume guarantee, including on version skew."""
    if (
        not storage
        or storage["mode"] != "disk-overlay"
        or storage["storage_mb"] != storage_mb
        or storage["policy"] != "nominal-image-mib"
    ):
        raise RuntimeError("Worker did not establish the requested disk volume")
    if (
        not storage["writable_paths_verified"]
        or storage["image_sha256"] != image_sha256
    ):
        raise RuntimeError("Worker image or writable mount verification failed")


def validate_start_mounts(
    status: dict[str, Any], requested: list[dict[str, str]]
) -> None:
    """Require every requested start-time mount in the worker's status.

    A pre-0.14 agent ignores ``readonly_mounts`` and starts without them.
    """
    if not requested:
        return
    reported = {
        (item["source"], item["target"])
        for item in status.get("readonly_mounts") or []
        if item.get("read_only") is True
    }
    missing = [
        item["target"]
        for item in requested
        if (item["source"], item["target"]) not in reported
    ]
    if missing:
        raise RuntimeError(
            f"Sandfleet worker did not mount {missing} read-only at start "
            "(it may predate Sandfleet 0.14)"
        )


def _lease_selection(
    pool: str | None,
    cpus: int | None,
    memory_mb: int | None,
    gpu: int | None,
    gpu_type: list[str] | tuple[str, ...] | None,
) -> tuple[dict[str, Any], str]:
    resource_mode = any(value is not None for value in (cpus, memory_mb, gpu, gpu_type))
    if (pool is not None) == resource_mode:
        raise ValueError("Set either pool or cpus and memory_mb")
    if resource_mode and (cpus is None or memory_mb is None):
        raise ValueError("Resource requests require cpus and memory_mb")
    if (gpu is None) != (gpu_type is None):
        raise ValueError("GPU resource requests require both gpu and gpu_type")
    if not resource_mode:
        return {"pool": pool}, f"pool {pool!r}"
    resources: dict[str, Any] = {"cpus": cpus, "memory_mb": memory_mb}
    if gpu is not None and gpu_type is not None:
        resources.update(GPURequest.from_values(gpu, list(gpu_type)).to_payload())
    return {"resources": resources}, f"resources({resources!r})"


class SandfleetClient:
    def __init__(
        self, url: str, token: str, *, timeout: float = 60, registry: str | None = None
    ):
        if not url or not token:
            raise ValueError("Sandfleet URL and token are required")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.registry = registry

    def _controller_request(self, url: str, path: str, **kwargs: Any) -> dict[str, Any]:
        return controller_request(
            request_json, url, path, registry=self.registry, **kwargs
        )

    def health(self) -> dict[str, Any]:
        return self._controller_request(
            self.url, f"/{API_VERSION}/health", token=self.token, timeout=self.timeout
        )

    def metrics(self) -> dict[str, Any]:
        """Return the controller's machine-readable operational snapshot."""
        return self._controller_request(
            self.url, f"/{API_VERSION}/metrics", token=self.token, timeout=self.timeout
        )

    def gpu_capabilities(self) -> dict[str, Any]:
        """Return configured GPU types and cached, advisory scheduler availability."""

        return request_json(
            self.url,
            f"/{API_VERSION}/capabilities/gpus",
            token=self.token,
            timeout=self.timeout,
        )

    def create_pool(
        self,
        name: str,
        *,
        desired_capacity: int | None = None,
        min_capacity: int = 0,
        max_capacity: int | None = None,
        warm_capacity: int | None = None,
        slots_per_job: int,
        cpus: int = 1,
        memory_mb: int = 4096,
        idle_timeout_seconds: int = 600,
    ) -> dict[str, Any]:
        if desired_capacity is None and max_capacity is None:
            raise ValueError("desired_capacity or max_capacity is required")
        if desired_capacity is not None and max_capacity is not None:
            raise ValueError("desired_capacity and max_capacity are mutually exclusive")
        payload: dict[str, Any] = {
            "name": name,
            "slots_per_job": slots_per_job,
            "cpus": cpus,
            "memory_mb": memory_mb,
            "idle_timeout_seconds": idle_timeout_seconds,
        }
        if desired_capacity is not None:
            payload["desired_capacity"] = desired_capacity
        else:
            payload.update(
                {
                    "min_capacity": min_capacity,
                    "max_capacity": max_capacity,
                    "warm_capacity": min_capacity
                    if warm_capacity is None
                    else warm_capacity,
                }
            )
        return self._controller_request(
            self.url,
            f"/{API_VERSION}/pools",
            method="POST",
            token=self.token,
            payload=payload,
            timeout=self.timeout,
        )

    def pool(self, name: str) -> dict[str, Any]:
        return self._controller_request(
            self.url,
            f"/{API_VERSION}/pools/{name}",
            token=self.token,
            timeout=self.timeout,
        )

    def wait_for_pool(
        self, name: str, *, capacity: int | None = None, timeout: float = 600
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        delay = 1.0
        while True:
            status = self.pool(name)
            target = int(status["desired_capacity"]) if capacity is None else capacity
            if int(status["ready_capacity"]) >= target:
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Pool {name!r} did not reach capacity {target} within {timeout:.0f}s; "
                    f"last status={status}"
                )
            time.sleep(
                min(
                    delay * random.uniform(0.9, 1.1),
                    max(0.0, deadline - time.monotonic()),
                )
            )
            delay = min(10.0, delay * 1.5)

    def delete_pool(self, name: str) -> dict[str, Any]:
        return self._controller_request(
            self.url,
            f"/{API_VERSION}/pools/{name}",
            method="DELETE",
            token=self.token,
            timeout=self.timeout,
        )

    def resume_pool(self, name: str) -> dict[str, Any]:
        return self._controller_request(
            self.url,
            f"/{API_VERSION}/pools/{name}/resume",
            method="POST",
            token=self.token,
            payload={},
            timeout=self.timeout,
        )

    def acquire(
        self,
        pool: str | None = None,
        *,
        cpus: int | None = None,
        memory_mb: int | None = None,
        gpu: int | None = None,
        gpu_type: list[str] | tuple[str, ...] | None = None,
        timeout: float = 900,
        cancel_event: threading.Event | None = None,
        internet: bool = False,
    ) -> "SandboxLease":
        selection, target = _lease_selection(pool, cpus, memory_mb, gpu, gpu_type)
        request_payload = {
            **selection,
            "timeout_seconds": max(1, int(timeout)),
            # Requires an operator-enabled controller; offline is the default.
            **({"internet": True} if internet else {}),
        }
        deadline = time.monotonic() + timeout
        result = self._controller_request(
            self.url,
            f"/{API_VERSION}/lease-requests",
            method="POST",
            token=self.token,
            payload=request_payload,
            timeout=self.timeout,
            retry_seconds=min(300.0, max(0.001, deadline - time.monotonic())),
            stop=cancel_event,
        )
        request_id = str(result["request_id"])
        delay = max(0.0, float(result["poll_after_seconds"]))
        try:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Lease acquisition cancelled")
            while result["status"] == "pending":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for a sandbox from {target}")
                base_delay = max(delay, float(result["poll_after_seconds"]))
                wait = min(base_delay * random.uniform(0.9, 1.1), remaining)
                if cancel_event is not None:
                    if cancel_event.wait(wait):
                        raise InterruptedError("Lease acquisition cancelled")
                else:
                    time.sleep(wait)
                delay = min(10.0, max(0.1, base_delay * 1.5))
                result = self._controller_request(
                    self.url,
                    f"/{API_VERSION}/lease-requests/{request_id}",
                    token=self.token,
                    timeout=self.timeout,
                    retry_seconds=min(300.0, max(0.001, deadline - time.monotonic())),
                    stop=cancel_event,
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Lease acquisition cancelled")
            if result["status"] != "assigned":
                raise TimeoutError(
                    result["error"] or f"Could not acquire a sandbox from {target}"
                )
            lease = result["lease"]
        except Exception as error:
            try:
                self._controller_request(
                    self.url,
                    f"/{API_VERSION}/lease-requests/{request_id}",
                    method="DELETE",
                    token=self.token,
                    timeout=self.timeout,
                )
            except Exception as cleanup_error:
                error.add_note(
                    f"Could not cancel lease request {request_id}: {cleanup_error}"
                )
            raise
        return SandboxLease(
            controller=self,
            lease_id=lease["lease_id"],
            lease_token=lease["lease_token"],
            agent_url=lease["agent_url"],
            pool=str(lease["pool"]),
            ttl_seconds=int(lease["ttl_seconds"]),
        )

    def release(self, lease_id: str) -> None:
        self._controller_request(
            self.url,
            f"/{API_VERSION}/leases/{lease_id}",
            method="DELETE",
            token=self.token,
            timeout=max(self.timeout, 120),
        )


class SandboxLease:
    def __init__(
        self,
        *,
        controller: SandfleetClient,
        lease_id: str,
        lease_token: str,
        agent_url: str,
        pool: str,
        ttl_seconds: int = 0,
    ):
        self.controller = controller
        self.lease_id = lease_id
        self.lease_token = lease_token
        self.agent_url = agent_url.rstrip("/")
        self.pool = pool
        self.closed = False
        self.image: str | None = None
        self.command_timeout = 1800
        self.pwd = "/workspace"
        self.extra_start_flags: list[str] = []
        self.storage_mb: int | None = None
        self.image_sha256: str | None = None
        self.readonly_mounts: list[dict[str, str]] = []
        self.ttl_seconds = ttl_seconds
        self._renew_stop = threading.Event()
        self._renew_thread: threading.Thread | None = None
        self._renewal_error: Exception | None = None
        if ttl_seconds > 0:
            self._renew_thread = threading.Thread(
                target=self._renew_loop,
                name=f"sandfleet-renew-{lease_id[:8]}",
                daemon=True,
            )
            self._renew_thread.start()

    def _renew_loop(self) -> None:
        interval = max(0.1, min(30.0, self.ttl_seconds / 3))
        while not self._renew_stop.wait(interval * random.uniform(0.9, 1.1)):
            try:
                self.controller._controller_request(
                    self.controller.url,
                    f"/{API_VERSION}/leases/{self.lease_id}/renew",
                    method="POST",
                    token=self.controller.token,
                    payload={},
                    timeout=min(10, self.controller.timeout),
                    retry_seconds=min(300.0, self.ttl_seconds / 2),
                    stop=self._renew_stop,
                )
            except InterruptedError:
                return
            except (SandfleetError, OSError, ValueError, KeyError) as error:
                self._renewal_error = error
                return

    @property
    def path(self) -> str:
        return f"/{API_VERSION}/leases/{self.lease_id}"

    def _request(
        self,
        suffix: str,
        *,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: float = 60,
    ) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("Sandbox lease has been released")
        if self._renewal_error is not None:
            raise SandboxLostError(
                f"Lease renewal failed: {self._renewal_error}",
                error_type="SandboxLostError",
            ) from self._renewal_error
        try:
            return request_json(
                self.agent_url,
                self.path + suffix,
                method=method,
                token=self.lease_token,
                payload=payload,
                timeout=timeout,
            )
        except SandfleetError as error:
            if error.status is not None:
                raise
            reason = str(error)
            try:
                status = self.controller._controller_request(
                    self.controller.url,
                    self.path,
                    token=self.controller.token,
                    timeout=self.controller.timeout,
                )
                reason = status.get("lost_reason") or reason
            except Exception as controller_error:
                error.add_note(
                    f"Controller could not explain the lost agent connection: {controller_error}"
                )
            raise SandboxLostError(reason, error_type="SandboxLostError") from error

    def start(
        self,
        image: str,
        *,
        command_timeout: int = 1800,
        pwd: str = "/workspace",
        extra_start_flags: list[str] | None = None,
        storage_mb: int | None = None,
        image_sha256: str | None = None,
        readonly_mounts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Start the sandbox; ``restart()`` reuses these settings.

        ``readonly_mounts`` (feature ``lease-start-readonly-mounts-v1``) is a
        list of ``{"source": host_dir, "target": sandbox_dir}`` mounted read-only
        from the start, for inputs the agent may read. Each source must resolve
        inside one of the controller's ``agent_readonly_mount_roots``.
        """
        self.image = image
        self.command_timeout = command_timeout
        self.pwd = pwd
        self.extra_start_flags = extra_start_flags or []
        self.storage_mb = storage_mb
        self.image_sha256 = image_sha256
        self.readonly_mounts = [dict(item) for item in readonly_mounts or []]
        return self._start_instance("/start")

    def restart(self, image: str | None = None) -> dict[str, Any]:
        selected_image = image or self.image
        if selected_image is None:
            raise RuntimeError("Sandbox has not been started")
        self.image = selected_image
        return self._start_instance("/restart")

    def _start_instance(self, operation: str) -> dict[str, Any]:
        result = self._request(
            operation,
            payload={
                "image": self.image,
                "timeout": self.command_timeout,
                "pwd": self.pwd,
                "extra_start_flags": self.extra_start_flags,
                **self._disk_payload(),
                # Only sent when used, so agents without the field keep working.
                **(
                    {"readonly_mounts": self.readonly_mounts}
                    if self.readonly_mounts
                    else {}
                ),
            },
            timeout=max(1800, self.command_timeout + 60),
        )
        if self.storage_mb is not None:
            validate_disk_status(
                result["storage"],
                storage_mb=self.storage_mb,
                image_sha256=self.image_sha256,
            )
        validate_start_mounts(result, self.readonly_mounts)
        return result

    def _disk_payload(self) -> dict[str, Any]:
        if self.storage_mb is None:
            return {}
        return {"storage_mb": self.storage_mb, "image_sha256": self.image_sha256}

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
        *,
        env: dict[str, str] | None = None,
        command_id: str | None = None,
    ) -> ExecutionResult:
        effective_timeout = self.command_timeout if timeout is None else timeout
        payload: dict[str, Any] = {"command": command, "timeout": timeout}
        if env is not None:
            payload["env"] = validate_command_env(env)
        if command_id is not None:
            payload["command_id"] = command_id
        result = self._request(
            "/exec-env" if env is not None else "/exec",
            payload=payload,
            timeout=effective_timeout + 30,
        )
        return ExecutionResult(
            stdout=str(result["stdout"]),
            stderr=str(result["stderr"]),
            exit_code=int(result["exit_code"]),
            output_truncated=bool(result.get("output_truncated", False)),
            cancelled=bool(result.get("cancelled", False)),
            timed_out=result.get("timed_out"),
        )

    def cancel_command(self, command_id: str) -> dict[str, Any]:
        """Cancel one command by id, concurrently with its ``run_command``.

        Returns the worker state: ``pending``, ``running`` or ``already_finished``.
        """
        return self._request(
            "/exec-cancel", payload={"command_id": command_id}, timeout=30
        )

    def kill_processes(self) -> dict[str, Any]:
        """SIGKILL every process in the sandbox except its init (PID 1).

        Also stops a concurrent ``run_command``, which then reports
        ``cancelled``. Returns ``killed`` (processes found and killed) and
        ``cancelled_commands`` (IDs of commands that were running). Raises if
        any process survives; the sandbox is not stopped or released.
        """
        return self._request("/kill-processes", payload={}, timeout=120)

    def add_readonly_mounts(self, mounts: list[dict[str, str]]) -> dict[str, Any]:
        """Add operator-allowlisted host directories as read-only mounts.

        ``mounts`` is a list of ``{"source": host_dir, "target": sandbox_dir}``;
        each source must resolve inside one of the controller's
        ``readonly_mount_roots``. Meant for after an agent phase and before
        verification. Needs a disk-overlay sandbox: the worker kills every
        sandbox process, stops the instance and restarts it from the same
        overlay with the binds, so files persist and background processes do
        not. Returns the new status plus ``added_mounts`` and
        ``stopped_processes``. If the restart fails the sandbox is gone; release
        the lease.
        """
        return self._request(
            "/add-mounts", payload={"mounts": mounts}, timeout=ADD_MOUNTS_TIMEOUT
        )

    def write_file(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode("utf-8")
        self._request(
            "/write-file",
            payload={"path": path, "content_b64": encode_bytes(content)},
            timeout=120,
        )

    def write_file_chunk(
        self, path: str, content: bytes, *, offset: int, truncate: bool = False
    ) -> None:
        self._request(
            "/write-file",
            payload={
                "path": path,
                "content_b64": encode_bytes(content),
                "offset": offset,
                "truncate": truncate,
            },
            timeout=120,
        )

    def read_file_chunk(self, path: str, *, offset: int, size: int = 262144) -> bytes:
        result = self._request(
            "/read-file",
            payload={"path": path, "offset": offset, "size": size},
            timeout=120,
        )
        return decode_bytes(result["content_b64"])

    def abort(self) -> None:
        """Interrupt only this lease's step; it must subsequently be released."""
        self._request("/abort", payload={}, timeout=30)

    def read_file(self, path: str, *, binary: bool = False) -> str | bytes:
        result = self._request("/read-file", payload={"path": path}, timeout=120)
        content = decode_bytes(result["content_b64"])
        return content if binary else content.decode("utf-8", errors="replace")

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        self._request(
            "/put-archive",
            payload={"root": root, "content_b64": encode_bytes(tar_bytes)},
            timeout=300,
        )

    def status(self) -> dict[str, Any]:
        return self._request("/status", method="GET", timeout=30)

    def close(self) -> None:
        if self.closed:
            return
        self._renew_stop.set()
        if self._renew_thread is not None:
            self._renew_thread.join(timeout=11)
        self.controller.release(self.lease_id)
        self.closed = True

    def __enter__(self) -> "SandboxLease":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        error: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            self.close()
        except Exception as close_error:
            if error is None:
                raise
            error.add_note(f"Sandbox release also failed: {close_error}")
