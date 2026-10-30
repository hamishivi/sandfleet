"""Persistent Slurm-step client used by a Sandfleet node agent."""

from __future__ import annotations

import contextlib
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any

from sandfleet.errors import SandfleetError, SandboxOOMError
from sandfleet.mounts import validate_phase_roots
from sandfleet.network import validate_network_config
from sandfleet.step_worker import KILL_PROCESSES_BUDGET_SECONDS


CONTROL_CONNECT_TIMEOUT_SECONDS = 5.0
# The worker bounds a kill (lock wait, kill loop, fallback interrupts) by
# KILL_PROCESSES_BUDGET_SECONDS; the extra covers the control-channel transport.
KILL_PROCESSES_CONTROL_TIMEOUT_SECONDS = KILL_PROCESSES_BUDGET_SECONDS + 15.0
# Adding read-only mounts kills the sandbox's processes, stops the instance and
# starts it again (Apptainer's own start is bounded at 300 s).
ADD_READONLY_MOUNTS_TIMEOUT_SECONDS = 900


def parse_cpuset(value: str) -> frozenset[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        if not item:
            continue
        start, separator, end = item.partition("-")
        cpus.update(range(int(start), int(end) + 1) if separator else [int(start)])
    return frozenset(cpus)


class SlurmStepSandbox:
    """Own one persistent worker step and replace Apptainer inside it."""

    OOM_RETURN_CODES = {-9, 137, 253}

    def __init__(
        self,
        *,
        cpus: int,
        memory_mb: int,
        gpus: int = 0,
        gpu_type: str | None = None,
        apptainer_binary: str = "apptainer",
        worker_python: str | None = None,
        cache_dir: str | None = None,
        tmp_dir: str | None = None,
        srun_binary: str = "srun",
        node: str | None = None,
        start_timeout: int = 120,
        srun_extra_args: tuple[str, ...] = (),
        max_storage_mb: int = 0,
        network_mode: str = "loopback",
        gpu_container_mode: str = "auto",
        relay_max_pending: int = 1,
        readonly_mount_roots: tuple[str, ...] = (),
        agent_readonly_mount_roots: tuple[str, ...] = (),
    ):
        validate_network_config(network_mode)
        if cpus < 1 or memory_mb < 1:
            raise ValueError("cpus and memory_mb must be positive")
        if gpus < 0 or (gpus == 0) != (gpu_type is None):
            raise ValueError("GPU count and type must be configured together")
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.gpus = gpus
        self.gpu_type = gpu_type
        self.apptainer_binary = apptainer_binary
        self.worker_python = worker_python or sys.executable
        self.cache_dir = cache_dir
        self.tmp_dir = tmp_dir
        self.srun_binary = srun_binary
        self.node = (
            node
            or os.getenv("SLURMD_NODENAME")
            or socket.gethostname().split(".", 1)[0]
        )
        self.start_timeout = start_timeout
        self.srun_extra_args = srun_extra_args
        self.max_storage_mb = max_storage_mb
        self.storage_status: dict[str, Any] | None = None
        self.network_mode = network_mode
        self.gpu_container_mode = gpu_container_mode
        self.relay_max_pending = relay_max_pending
        self.agent_readonly_mount_roots, self.readonly_mount_roots = (
            validate_phase_roots(agent_readonly_mount_roots, readonly_mount_roots)
        )
        self.readonly_mounts: list[dict[str, Any]] = []

        self.process: subprocess.Popen | None = None
        self.connection: Connection | None = None
        self.control_connection: Connection | None = None
        self.worker_dir: str | None = None
        self.worker_log_path: str | None = None
        self.worker_log_handle = None
        self.metadata: dict[str, Any] = {}
        self.network_metadata: dict[str, Any] = {}
        self.instance_name: str | None = None
        self._lock = threading.Lock()
        self._control_lock = threading.Lock()
        self._interrupted = threading.Event()

    @property
    def storage_lost(self) -> bool:
        """Whether the step's private directory vanished, e.g. with the job's /tmp."""
        return self.worker_dir is not None and not os.path.isdir(self.worker_dir)

    @property
    def healthy(self) -> bool:
        """Whether the isolated Slurm step is alive, connected, and has its storage."""
        return (
            self.process is not None
            and self.process.poll() is None
            and self.connection is not None
            and not self.storage_lost
        )

    def _log_tail(self, max_bytes: int = 16_384) -> str:
        if not self.worker_log_path:
            return ""
        try:
            with open(self.worker_log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                return handle.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _srun_command(self, *worker_args: str) -> list[str]:
        gres = "none" if not self.gpus else f"gpu:{self.gpu_type}:{self.gpus}"
        return [
            self.srun_binary,
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={self.node}",
            f"--cpus-per-task={self.cpus}",
            f"--mem={self.memory_mb}M",
            "--kill-on-bad-exit=1",
            "--unbuffered",
            f"--gres={gres}",
            "--exact",
            "--exclusive",
            *self.srun_extra_args,
            self.worker_python,
            "-m",
            "sandfleet.step_worker",
            "--gpus",
            str(self.gpus),
            *worker_args,
        ]

    def _launch_worker(self) -> None:
        self._interrupted.clear()
        if shutil.which(self.srun_binary) is None:
            raise RuntimeError(f"srun binary {self.srun_binary!r} is not available")
        if not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("Sandfleet agent must run inside a Slurm allocation")

        self.worker_dir = tempfile.mkdtemp(prefix="sandfleet-step-", dir="/tmp")
        os.chmod(self.worker_dir, 0o700)
        socket_path = os.path.join(self.worker_dir, "worker.sock")
        control_socket_path = os.path.join(self.worker_dir, "control.sock")
        self.worker_log_path = os.path.join(self.worker_dir, "worker.log")
        authkey = secrets.token_bytes(32)
        command = self._srun_command(
            "--socket",
            socket_path,
            "--authkey",
            authkey.hex(),
            "--control-socket",
            control_socket_path,
            *(
                [
                    "--storage-root",
                    self.worker_dir,
                    "--max-storage-mb",
                    str(self.max_storage_mb),
                ]
                if self.max_storage_mb
                else []
            ),
            *(
                argument
                for root in self.readonly_mount_roots
                for argument in ("--readonly-mount-root", root)
            ),
            *(
                argument
                for root in self.agent_readonly_mount_roots
                for argument in ("--agent-readonly-mount-root", root)
            ),
        )
        self.worker_log_handle = open(self.worker_log_path, "w+b")  # noqa: SIM115
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self.worker_log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.start_timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                detail = self._log_tail()
                self._terminate_worker()
                raise RuntimeError(
                    f"srun worker exited during startup with code {return_code}: {detail}"
                )
            if os.path.exists(socket_path):
                try:
                    self.connection = Client(
                        socket_path, family="AF_UNIX", authkey=authkey
                    )
                except (ConnectionError, OSError) as error:
                    last_error = error
                    time.sleep(0.05)
                    continue
                self._establish_control(control_socket_path, authkey)
                return
            time.sleep(0.05)
        detail = self._log_tail() or (
            "srun produced no output; the allocation may lack free step resources"
        )
        self._terminate_worker()
        raise TimeoutError(
            f"srun worker startup timed out (last_error={last_error!r}, log={detail!r})"
        )

    def _raise_remote_error(self, response: dict[str, Any]) -> None:
        error_type = response["error_type"]
        message = response["message"]
        remote_traceback = response["traceback"]
        detail = (
            f"{message}\nRemote traceback:\n{remote_traceback}"
            if remote_traceback
            else message
        )
        if error_type == "SandboxOOMError":
            raise SandboxOOMError(detail, error_type=error_type)
        if error_type == "FileNotFoundError":
            raise FileNotFoundError(message)
        if error_type == "IsADirectoryError":
            raise IsADirectoryError(message)
        if error_type == "ValueError":
            raise ValueError(detail)
        raise SandfleetError(detail, error_type=error_type)

    def _rpc(self, operation: str, *, response_timeout: float, **payload: Any) -> Any:
        connection = self.connection
        process = self.process
        if connection is None or process is None:
            raise RuntimeError("Slurm step worker is not running")
        with self._lock:
            try:
                connection.send({"operation": operation, **payload})
                response = self._receive(connection, response_timeout)
            except TimeoutError as cause:
                detail = self._log_tail()
                error = TimeoutError(
                    f"Step operation {operation!r} timed out: {detail}"
                )
                self._cleanup_after_failure(error)
                raise error from cause
            except (EOFError, OSError) as cause:
                return_code = process.poll()
                detail = self._log_tail()
                if return_code in self.OOM_RETURN_CODES:
                    error = SandboxOOMError(
                        f"Sandbox step exited after an apparent OOM (code={return_code}): {detail}"
                    )
                else:
                    error = SandfleetError(
                        f"Lost connection to sandbox step (code={return_code}): {detail}"
                    )
                self._cleanup_after_failure(error)
                raise error from cause
        if not response["ok"]:
            self._raise_remote_error(response)
        return response["result"]

    def _cleanup_after_failure(self, error: Exception) -> None:
        try:
            self._terminate_worker()
        except Exception as cleanup_error:
            error.add_note(f"Step cleanup retained for retry: {cleanup_error}")

    def _receive(self, connection: Connection, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while not connection.poll(min(1.0, max(0.0, deadline - time.monotonic()))):
            if self._interrupted.is_set():
                raise ConnectionAbortedError("Lease execution explicitly interrupted")
            if time.monotonic() >= deadline:
                raise TimeoutError("Step operation deadline exceeded")
        return connection.recv()

    def abort(self) -> None:
        """Wake an in-flight RPC and interrupt its srun; never submit a new step."""
        self._interrupted.set()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def _establish_control(self, control_socket_path: str, authkey: bytes) -> None:
        # The control listener starts just after the main one. It is required.
        deadline = time.monotonic() + CONTROL_CONNECT_TIMEOUT_SECONDS
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if os.path.exists(control_socket_path):
                try:
                    self.control_connection = Client(
                        control_socket_path, family="AF_UNIX", authkey=authkey
                    )
                    return
                except (ConnectionError, OSError) as error:
                    last_error = error
            time.sleep(0.05)
        detail = self._log_tail()
        self._terminate_worker()
        raise RuntimeError(
            f"Worker control channel did not come up "
            f"(last_error={last_error!r}, log={detail!r})"
        )

    def cancel_command(self, command_id: str) -> dict[str, Any]:
        """Cancel one command over the control channel, even while an exec is running."""
        return self._control("cancel", 30, command_id=command_id)

    def kill_processes(self) -> dict[str, Any]:
        """Kill every sandbox process except init, even while an exec is running."""
        return self._control("kill_processes", KILL_PROCESSES_CONTROL_TIMEOUT_SECONDS)

    def _control(self, operation: str, timeout: float, **payload: Any) -> Any:
        connection = self.control_connection
        if connection is None:
            raise SandfleetError(
                f"Sandbox control channel is unavailable; cannot {operation}",
                error_type="SandfleetError",
            )
        with self._control_lock:
            try:
                connection.send({"operation": operation, **payload})
                if not connection.poll(timeout):
                    raise TimeoutError(f"{operation} control operation timed out")
                response = connection.recv()
            except (EOFError, OSError, TimeoutError) as cause:
                self.control_connection = None
                raise SandfleetError(
                    f"Lost sandbox control channel during {operation}: {cause}",
                    error_type="SandfleetError",
                ) from cause
        if not response["ok"]:
            self._raise_remote_error(response)
        return response["result"]

    def _config(
        self,
        *,
        image: str,
        command_timeout: int,
        pwd: str,
        extra_start_flags: list[str] | None,
        storage_mb: int | None = None,
        image_sha256: str | None = None,
        proxy_url: str | None = None,
    ) -> dict[str, Any]:
        config = {
            "image": image,
            "timeout": command_timeout,
            "pwd": pwd,
            "cache_dir": self.cache_dir,
            "tmp_dir": self.tmp_dir,
            "extra_start_flags": extra_start_flags or [],
            "apptainer_binary": self.apptainer_binary,
            "gpu_enabled": self.gpus > 0,
            "network_mode": self.network_mode,
            "gpu_container_mode": self.gpu_container_mode,
            "proxy_url": proxy_url,
            "relay_max_pending": self.relay_max_pending,
        }
        if storage_mb is not None:
            config.update(storage_mb=storage_mb, image_sha256=image_sha256)
        return config

    @property
    def cpuset(self) -> frozenset[int]:
        return parse_cpuset(str(self.metadata.get("cpuset.cpus.effective", "")))

    @property
    def cgroup_path(self) -> str:
        return str(self.metadata.get("path", ""))

    @property
    def gpu_devices(self) -> frozenset[str]:
        gpu = self.metadata.get("gpu", {})
        return frozenset(str(item["path"]) for item in gpu.get("accessible", []))

    def _record_status(self, result: dict[str, Any]) -> None:
        metadata = result["cgroup"]
        memory_limit = int(metadata["memory.max.effective"])
        if memory_limit > self.memory_mb * 1024 * 1024:
            raise RuntimeError(
                f"Slurm step memory limit is {memory_limit} bytes; expected at most {self.memory_mb} MiB"
            )
        cpuset = parse_cpuset(metadata["cpuset.cpus.effective"])
        if not cpuset:
            raise RuntimeError("Slurm step has an empty CPU set")
        gpu = metadata.get("gpu", {})
        gpu_errors = gpu.get("errors", [])
        if gpu_errors:
            raise RuntimeError(f"Could not probe Slurm step GPU devices: {gpu_errors}")
        accessible = gpu.get("accessible", [])
        if len(accessible) != self.gpus:
            raise RuntimeError(
                f"Slurm step can access {len(accessible)} GPU devices; expected exactly {self.gpus}"
            )
        if self.gpus:
            identities = gpu.get("identities", [])
            if len(identities) != self.gpus:
                raise RuntimeError(
                    f"Slurm step reports {len(identities)} GPU identities; expected exactly {self.gpus}"
                )
            visible = str(gpu.get("cuda_visible_devices") or "")
            visible_count = len([item for item in visible.split(",") if item])
            if visible_count != self.gpus:
                raise RuntimeError(
                    "Slurm step CUDA_VISIBLE_DEVICES does not match its GPU allocation"
                )
        metadata["cpuset.cpu_count"] = len(cpuset)
        self.metadata = metadata
        self.instance_name = result["instance_name"]
        if self.max_storage_mb:
            if result["max_storage_mb"] != self.max_storage_mb:
                raise RuntimeError("Step worker disk policy/version mismatch")
            self.storage_status = result["storage"]
        self.network_metadata = result["network"]
        self._record_mounts(result)

    def _record_mounts(self, result: dict[str, Any]) -> None:
        if result["readonly_mount_roots"] != list(
            self.readonly_mount_roots
        ) or result.get("agent_readonly_mount_roots") != list(
            self.agent_readonly_mount_roots
        ):
            raise RuntimeError("Step worker read-only mount policy/version mismatch")
        self.readonly_mounts = result["readonly_mounts"]

    def start(self, *, image: str, **options: Any) -> dict[str, Any]:
        return self._start_instance("start", image=image, **options)

    def restart(self, *, image: str, **options: Any) -> dict[str, Any]:
        return self._start_instance("restart", image=image, **options)

    def _start_instance(
        self,
        operation: str,
        *,
        image: str,
        command_timeout: int = 1800,
        pwd: str = "/workspace",
        extra_start_flags: list[str] | None = None,
        storage_mb: int | None = None,
        image_sha256: str | None = None,
        proxy_url: str | None = None,
        readonly_mounts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        config = self._config(
            image=image,
            command_timeout=command_timeout,
            pwd=pwd,
            extra_start_flags=extra_start_flags,
            storage_mb=storage_mb,
            image_sha256=image_sha256,
            proxy_url=proxy_url,
        )
        if readonly_mounts is not None:
            # Agent-phase mounts; the worker checks them against its roots.
            config["readonly_mounts"] = readonly_mounts
        result = self._rpc(
            operation,
            response_timeout=max(1800, command_timeout + 60),
            config=config,
        )
        self._record_status(result)
        return self.status()

    def prepare(self) -> dict[str, Any]:
        """Create the resource-limited Slurm step without starting Apptainer."""
        if not self.healthy:
            self.discard()
            self._launch_worker()
        result = self._rpc("status", response_timeout=30)
        self._record_status(result)
        return self.status()

    def add_readonly_mounts(self, mounts: list[dict[str, str]]) -> dict[str, Any]:
        """Restart the instance from its overlay with extra read-only mounts."""
        result = self._rpc(
            "add_readonly_mounts",
            response_timeout=ADD_READONLY_MOUNTS_TIMEOUT_SECONDS,
            mounts=mounts,
        )
        self._record_status(result)
        return {
            **self.status(),
            "added_mounts": result["added_mounts"],
            "stopped_processes": result["stopped_processes"],
        }

    def recycle(self) -> dict[str, Any]:
        """Stop the current container while retaining its Slurm step."""
        if not self.healthy:
            raise RuntimeError("Slurm step worker is not running")
        result = self._rpc("stop", response_timeout=60)
        self._record_status(result)
        return self.status()

    def run_command(
        self,
        command: str,
        timeout: int | None = None,
        *,
        env: dict[str, str] | None = None,
        command_id: str | None = None,
    ) -> dict[str, Any]:
        effective_timeout = 1800 if timeout is None else timeout
        return self._rpc(
            "run_command_env" if env is not None else "run_command",
            response_timeout=effective_timeout + 30,
            command=command,
            timeout=timeout,
            **({"env": env} if env is not None else {}),
            **({"command_id": command_id} if command_id is not None else {}),
        )

    def write_file(self, path: str, content: bytes, **chunk: Any) -> None:
        self._rpc(
            "write_file", response_timeout=120, path=path, content=content, **chunk
        )

    def read_file(self, path: str, **chunk: Any) -> bytes:
        return self._rpc("read_file", response_timeout=120, path=path, **chunk)

    def put_archive(self, root: str, tar_bytes: bytes) -> None:
        self._rpc("put_archive", response_timeout=300, root=root, tar_bytes=tar_bytes)

    def status(self) -> dict[str, Any]:
        return {
            "instance_name": self.instance_name,
            "cgroup": self.metadata,
            "network": self.network_metadata,
            "worker_pid": None if self.process is None else self.process.pid,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "gpus": self.gpus,
            "gpu_type": self.gpu_type,
            "healthy": self.healthy,
            "storage": self.storage_status,
            "max_storage_mb": self.max_storage_mb,
            "readonly_mounts": self.readonly_mounts,
            "readonly_mount_roots": list(self.readonly_mount_roots),
            "agent_readonly_mount_roots": list(self.agent_readonly_mount_roots),
        }

    def _terminate_worker(self) -> None:
        if self.max_storage_mb:
            self._terminate_disk_worker()
            return
        control, self.control_connection = self.control_connection, None
        if control is not None:
            with contextlib.suppress(Exception):
                control.close()
        connection, self.connection = self.connection, None
        if connection is not None:
            with contextlib.suppress(Exception):
                connection.close()
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            with contextlib.suppress(Exception):
                process.terminate()
                process.wait(timeout=10)
            if process.poll() is None:
                with contextlib.suppress(Exception):
                    process.kill()
                    process.wait(timeout=5)
        if self.worker_log_handle is not None:
            with contextlib.suppress(Exception):
                self.worker_log_handle.close()
            self.worker_log_handle = None
        worker_dir, self.worker_dir = self.worker_dir, None
        if worker_dir:
            with contextlib.suppress(OSError):
                shutil.rmtree(worker_dir)
        self.worker_log_path = None

    def _terminate_disk_worker(self) -> None:
        # Unlike legacy best-effort discard, mounted disk files must survive an
        # uncertain shutdown so a later release can retry without losing state.
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        path = self.metadata.get("path")
        if path:
            events = Path(path) / "cgroup.events"
            if events.exists():
                values = dict(line.split() for line in events.read_text().splitlines())
                if values["populated"] != "0":
                    raise RuntimeError(
                        "Disk step still populated; retaining storage for cleanup retry"
                    )
        elif self.worker_dir and list(Path(self.worker_dir).glob("disk-overlay-*")):
            raise RuntimeError("Cannot establish disk step death; retaining storage")
        self._close_disk_resources()

    def _close_disk_resources(self) -> None:
        if self.control_connection is not None:
            with contextlib.suppress(Exception):
                self.control_connection.close()
            self.control_connection = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.worker_log_handle is not None:
            self.worker_log_handle.close()
            self.worker_log_handle = None
        if self.worker_dir is not None:
            # Already gone when Slurm tears down the job's /tmp under the step.
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self.worker_dir)
            self.worker_dir = None
        self.process = None
        self.worker_log_path = None

    def discard(self) -> None:
        """Force-discard a dead or unusable step without a shutdown RPC."""
        self._terminate_worker()
        self.instance_name = None
        self.metadata = {}
        self.storage_status = None

    def close(self) -> None:
        if self.process is None:
            return
        try:
            self._rpc("shutdown", response_timeout=30)
        except BaseException as error:
            try:
                self.discard()
            except Exception as cleanup_error:
                error.add_note(f"Step discard failed: {cleanup_error}")
            raise
        self.discard()

    def __enter__(self) -> "SlurmStepSandbox":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
