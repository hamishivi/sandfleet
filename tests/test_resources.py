from __future__ import annotations

import pytest

from sandfleet.resources import (
    DynamicPoolPolicy,
    GPURequest,
    GPUTypeCatalog,
    SandboxResources,
)


def test_gpu_request_preserves_ordered_fallbacks():
    request = GPURequest.from_values(2, ["h100_80gb", "a100_80gb"])
    assert request.to_payload() == {
        "gpu": 2,
        "type": ["h100_80gb", "a100_80gb"],
    }
    assert (
        request.pool_key
        != GPURequest.from_values(2, ["a100_80gb", "h100_80gb"]).pool_key
    )


def test_gpu_catalog_resolves_operator_mapping_without_exposing_slurm_names():
    catalog = GPUTypeCatalog.from_payload(
        {
            "types": {
                "h100_80gb": {"slurm_type": "h100-80gb", "memory_mb": 81920},
                "a100_40gb": {"slurm_type": "a100-40gb", "memory_mb": 40960},
            }
        }
    )
    placement = catalog.resolve(GPURequest.from_values(2, ["h200_141gb", "a100_40gb"]))
    assert placement.logical_type == "a100_40gb"
    assert placement.slurm_type == "a100-40gb"
    assert placement.memory_mb == 40960


def test_gpu_catalog_requires_distinct_scheduler_types():
    with pytest.raises(ValueError, match="distinct Slurm GRES"):
        GPUTypeCatalog.from_payload(
            {
                "types": {
                    "a100_80gb": {"slurm_type": "a100", "memory_mb": 81920},
                    "a100_40gb": {"slurm_type": "a100", "memory_mb": 40960},
                }
            }
        )


def test_gpu_count_bounds_worker_packing():
    resources = SandboxResources.from_payload(
        {"cpus": 2, "memory_mb": 1024, "gpu": 2, "type": ["h100_80gb"]}
    )
    policy = DynamicPoolPolicy(
        max_cpus_per_worker=64,
        max_memory_mb_per_worker=131072,
        max_gpus_per_worker=8,
        slots_per_job=16,
    )
    assert policy.slots_for(resources) == 4


@pytest.mark.parametrize(
    ("gpu", "gpu_types", "message"),
    [
        (0, ["a100_80gb"], "positive JSON integer"),
        (1, [], "non-empty JSON array"),
        (1, ["a100"], "memory-qualified"),
        (1, ["any"], "memory-qualified"),
        (1, ["a100_80gb", "a100_80gb"], "duplicates"),
    ],
)
def test_gpu_request_rejects_ambiguous_selectors(gpu, gpu_types, message):
    with pytest.raises(ValueError, match=message):
        GPURequest.from_values(gpu, gpu_types)
