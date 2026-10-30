"""Policy-bounded declarative sandbox resource shapes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


DYNAMIC_POOL_PREFIX = "sfres-"
# Apptainer GPU integration. `nvccli` delegates to nvidia-container-cli; it is
# needed where `--nv` cannot build a contained /dev, e.g. GB200/GB300 hosts whose
# /dev/nvidia-caps-imex-channels is a directory. `auto` picks per worker host.
GPU_CONTAINER_MODES = ("auto", "nv", "nvccli")
GPU_MODEL_NAME = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*_[1-9][0-9]*gb")
SLURM_GPU_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
MAX_GPU_TYPE_FALLBACKS = 32


@dataclass(frozen=True)
class GPURequest:
    """A homogeneous GPU count and an ordered list of acceptable types."""

    gpu: int
    type: tuple[str, ...]

    @classmethod
    def from_values(cls, gpu: Any, gpu_types: Any) -> "GPURequest":
        if type(gpu) is not int or gpu < 1:
            raise ValueError("gpu must be a positive JSON integer")
        if not isinstance(gpu_types, list) or not gpu_types:
            raise ValueError("type must be a non-empty JSON array")
        if len(gpu_types) > MAX_GPU_TYPE_FALLBACKS:
            raise ValueError(
                f"type may contain at most {MAX_GPU_TYPE_FALLBACKS} fallbacks"
            )
        if any(
            not isinstance(gpu_type, str) or GPU_MODEL_NAME.fullmatch(gpu_type) is None
            for gpu_type in gpu_types
        ):
            raise ValueError(
                "GPU types must be lowercase memory-qualified names such as a100_80gb"
            )
        if len(set(gpu_types)) != len(gpu_types):
            raise ValueError("GPU type fallbacks must not contain duplicates")
        return cls(gpu=gpu, type=tuple(gpu_types))

    def to_payload(self) -> dict[str, Any]:
        return {"gpu": self.gpu, "type": list(self.type)}

    @property
    def pool_key(self) -> str:
        if len(self.type) == 1:
            return f"gpu{self.gpu}-{self.type[0]}"
        encoded = json.dumps(self.to_payload(), separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:12]
        return f"gpu{self.gpu}-fallback-{digest}"


@dataclass(frozen=True)
class GPUPlacement:
    """One operator-approved client type mapped to a Slurm GRES type."""

    logical_type: str
    slurm_type: str
    memory_mb: int


@dataclass(frozen=True)
class GPUTypeSpec:
    """Public capability metadata plus its private Slurm GRES mapping."""

    slurm_type: str
    memory_mb: int


@dataclass(frozen=True)
class GPUTypeCatalog:
    """Trusted mapping from public GPU names to site-specific Slurm types."""

    types: dict[str, GPUTypeSpec]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GPUTypeCatalog":
        if not isinstance(payload, dict):
            raise ValueError("GPU type catalog must be a JSON object")
        unexpected = payload.keys() - {"types"}
        if unexpected:
            raise ValueError(
                "Unknown GPU type catalog fields: " + ", ".join(sorted(unexpected))
            )
        raw_types = payload.get("types")
        if not isinstance(raw_types, dict) or not raw_types:
            raise ValueError("GPU type catalog requires a non-empty types object")
        types: dict[str, GPUTypeSpec] = {}
        slurm_types: set[str] = set()
        for logical_type, raw_spec in raw_types.items():
            if (
                not isinstance(logical_type, str)
                or GPU_MODEL_NAME.fullmatch(logical_type) is None
            ):
                raise ValueError(
                    "GPU catalog names must be lowercase memory-qualified names such as a100_80gb"
                )
            if not isinstance(raw_spec, dict) or raw_spec.keys() != {
                "slurm_type",
                "memory_mb",
            }:
                raise ValueError(
                    f"GPU catalog entry {logical_type!r} requires exactly slurm_type and memory_mb"
                )
            slurm_type = raw_spec["slurm_type"]
            memory_mb = raw_spec["memory_mb"]
            if (
                not isinstance(slurm_type, str)
                or SLURM_GPU_TYPE.fullmatch(slurm_type) is None
            ):
                raise ValueError(
                    f"GPU catalog entry {logical_type!r} has an invalid Slurm GRES type"
                )
            if type(memory_mb) is not int or memory_mb < 1:
                raise ValueError(
                    f"GPU catalog entry {logical_type!r} has an invalid memory_mb"
                )
            if slurm_type in slurm_types:
                raise ValueError(
                    "Each logical GPU type must map to a distinct Slurm GRES type so "
                    "memory-qualified requests remain enforceable"
                )
            slurm_types.add(slurm_type)
            types[logical_type] = GPUTypeSpec(
                slurm_type=slurm_type, memory_mb=memory_mb
            )
        return cls(types=types)

    def resolve(self, request: GPURequest) -> GPUPlacement:
        for requested_type in request.type:
            if requested_type in self.types:
                return GPUPlacement(
                    logical_type=requested_type,
                    slurm_type=self.types[requested_type].slurm_type,
                    memory_mb=self.types[requested_type].memory_mb,
                )
        raise ValueError(
            "No requested GPU type is enabled by this controller's operator catalog"
        )


@dataclass(frozen=True)
class SandboxResources:
    cpus: int
    memory_mb: int
    gpu_request: GPURequest | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SandboxResources":
        unexpected = payload.keys() - {"cpus", "memory_mb", "gpu", "type"}
        if unexpected:
            raise ValueError(
                "Unknown resource fields: " + ", ".join(sorted(unexpected))
            )
        if not {"cpus", "memory_mb"} <= payload.keys():
            raise ValueError("Resource requests require cpus and memory_mb")
        values = (payload["cpus"], payload["memory_mb"])
        if any(type(value) is not int for value in values):
            raise ValueError("cpus and memory_mb must be JSON integers")
        has_gpu, has_type = "gpu" in payload, "type" in payload
        if has_gpu != has_type:
            raise ValueError("GPU resource requests require both gpu and type")
        gpu_request = (
            GPURequest.from_values(payload["gpu"], payload["type"]) if has_gpu else None
        )
        return cls(cpus=values[0], memory_mb=values[1], gpu_request=gpu_request)

    def validate(
        self,
        *,
        max_cpus_per_sandbox: int,
        max_memory_mb_per_sandbox: int,
        max_gpus_per_sandbox: int = 0,
        gpu_type_catalog: GPUTypeCatalog | None = None,
    ) -> None:
        if not 1 <= self.cpus <= max_cpus_per_sandbox:
            raise ValueError(f"cpus must be between 1 and {max_cpus_per_sandbox}")
        if not 64 <= self.memory_mb <= max_memory_mb_per_sandbox:
            raise ValueError(
                f"memory_mb must be between 64 and {max_memory_mb_per_sandbox}"
            )
        if self.gpu_request is not None:
            if gpu_type_catalog is None:
                raise ValueError(
                    "GPU sandboxes are disabled until an operator type catalog and Slurm device isolation are configured"
                )
            if self.gpu_request.gpu > max_gpus_per_sandbox:
                raise ValueError(
                    f"gpu must be between 1 and {max_gpus_per_sandbox} for this controller"
                )
            gpu_type_catalog.resolve(self.gpu_request)

    @property
    def pool_name(self) -> str:
        gpu_key = "gpu-none" if self.gpu_request is None else self.gpu_request.pool_key
        return f"{DYNAMIC_POOL_PREFIX}c{self.cpus}-m{self.memory_mb}-{gpu_key}"


@dataclass(frozen=True)
class DynamicPoolPolicy:
    max_cpus_per_worker: int
    max_memory_mb_per_worker: int
    slots_per_job: int = 16
    max_capacity: int = 512
    warm_capacity: int = 0
    idle_timeout_seconds: int = 600
    max_pools: int = 8
    cpu_allocation_granularity: int = 1
    memory_headroom_mb: int = 512
    max_gpus_per_worker: int = 0

    def validate(self, *, controller_max_capacity: int, max_pools: int) -> None:
        if not 1 <= self.max_capacity <= controller_max_capacity:
            raise ValueError(
                "dynamic pool max capacity must fit within controller max capacity"
            )
        if not 1 <= self.slots_per_job <= self.max_capacity:
            raise ValueError(
                "dynamic pool slots per job must be between 1 and its max capacity"
            )
        if not 0 <= self.warm_capacity <= self.max_capacity:
            raise ValueError(
                "dynamic pool warm capacity must be between 0 and its max capacity"
            )
        if self.idle_timeout_seconds < 0:
            raise ValueError("dynamic pool idle timeout must be non-negative")
        if not 1 <= self.max_pools <= max_pools:
            raise ValueError(
                "dynamic pool limit must be between 1 and controller max pools"
            )
        if self.max_cpus_per_worker < 1:
            raise ValueError("dynamic worker CPU limit must be positive")
        if self.max_memory_mb_per_worker <= self.memory_headroom_mb:
            raise ValueError(
                "dynamic worker memory limit must exceed agent memory headroom"
            )
        if self.cpu_allocation_granularity < 1:
            raise ValueError("CPU allocation granularity must be positive")
        if self.max_gpus_per_worker < 0:
            raise ValueError("dynamic worker GPU limit must be non-negative")

    def slots_for(self, resources: SandboxResources) -> int:
        rounded_cpus = (
            (resources.cpus + self.cpu_allocation_granularity - 1)
            // self.cpu_allocation_granularity
        ) * self.cpu_allocation_granularity
        cpu_slots = self.max_cpus_per_worker // rounded_cpus
        memory_slots = (
            self.max_memory_mb_per_worker - self.memory_headroom_mb
        ) // resources.memory_mb
        gpu_slots = self.slots_per_job
        if resources.gpu_request is not None:
            gpu_slots = self.max_gpus_per_worker // resources.gpu_request.gpu
        slots = min(self.slots_per_job, cpu_slots, memory_slots, gpu_slots)
        if slots < 1:
            raise ValueError(
                "Requested sandbox resources do not fit within one dynamic worker"
            )
        return slots
