from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from sandfleet.controller import ControllerState, check_agent_version
from sandfleet.egress import EgressSettings
from sandfleet.errors import SandboxOOMError, SandfleetError
from sandfleet.resources import DynamicPoolPolicy, GPURequest, GPUTypeCatalog
from sandfleet.slurm import SlurmSnapshot, SlurmSubmission
from sandfleet.version import VERSION


class FakeLauncher:
    def __init__(self):
        self.cancelled: list[str] = []
        self.submitted: list[dict] = []
        self.states: dict[str, dict[str, str]] = {}
        self.snapshot_calls = 0
        self.gpu_inventory_calls = 0
        self.gpu_probe_calls: list[str] = []
        self.gpu_inventory: dict[str, dict] = {}
        self.gpu_quota = None
        self.next_id = 100

    def submit_worker_batch(self, **kwargs):
        submission_id = str(self.next_id)
        self.next_id += 1
        self.submitted.append(kwargs)
        count = int(kwargs["worker_count"])
        self.states[submission_id] = {
            (f"{submission_id}_{index}" if count > 1 else submission_id): "PENDING"
            for index in range(count)
        }
        return SlurmSubmission(submission_id=submission_id, worker_count=count)

    def snapshot(self, submission_ids):
        self.snapshot_calls += 1
        return SlurmSnapshot(
            ok=True,
            states={
                submission_id: dict(self.states.get(submission_id, {}))
                for submission_id in submission_ids
            },
        )

    def cancel(self, job_ids):
        self.cancelled.extend(job_ids)
        for job_id in job_ids:
            base = job_id.split("_", 1)[0]
            if job_id == base:
                self.states.pop(base, None)
            else:
                self.states.get(base, {}).pop(job_id, None)

    def discover_gpu_inventory(self, gpu_types):
        self.gpu_inventory_calls += 1
        return {
            gpu_type: dict(
                self.gpu_inventory.get(
                    gpu_type,
                    {
                        "configured": 8,
                        "allocated": 0,
                        "free_estimate": 8,
                        "largest_node_free_estimate": 8,
                        "schedulable_nodes": 1,
                        "unavailable_nodes": 0,
                    },
                )
            )
            for gpu_type in gpu_types
        }

    def probe_gpu_access(self, gpu_type):
        self.gpu_probe_calls.append(gpu_type)
        return {"ok": True, "detail": "Job 1 to start at ..."}

    def discover_gpu_quota(self):
        return self.gpu_quota


def new_state(launcher, **kwargs):
    return ControllerState(
        token="secret",
        client_token="client-secret",
        metrics_token="metrics-secret",
        launcher=launcher,
        max_capacity=4096,
        max_cpus_per_sandbox=64,
        max_memory_mb_per_sandbox=262144,
        min_submit_interval_seconds=0,
        min_scheduler_poll_seconds=0,
        submission_grace_seconds=0,
        scale_up_debounce_seconds=0,
        **kwargs,
    )


def test_delete_pool_waits_for_an_inflight_submission(tmp_path, monkeypatch):
    launcher = FakeLauncher()
    state_path = tmp_path / "state.json"
    state = new_state(launcher, state_file=str(state_path))
    state.create_pool({"name": "example", "desired_capacity": 1, "slots_per_job": 1})
    submitting, finish_submit, deleting, deleted = (threading.Event() for _ in range(4))
    failures = []
    submit = launcher.submit_worker_batch

    def blocked_submit(**kwargs):
        submitting.set()
        assert finish_submit.wait(timeout=5)
        return submit(**kwargs)

    def reconcile():
        try:
            state.reconcile_once()
        except BaseException as error:
            failures.append(error)

    def delete():
        deleting.set()
        try:
            state.delete_pool("example")
        except BaseException as error:
            failures.append(error)
        finally:
            deleted.set()

    monkeypatch.setattr(launcher, "submit_worker_batch", blocked_submit)
    reconciler = threading.Thread(target=reconcile)
    deleter = threading.Thread(target=delete)
    reconciler.start()
    try:
        assert submitting.wait(timeout=2)
        deleter.start()
        assert deleting.wait(timeout=2)
        deleted_during_submit = deleted.wait(timeout=0.1)
    finally:
        finish_submit.set()
        reconciler.join(timeout=5)
        if deleter.ident is not None:
            deleter.join(timeout=5)
    assert not reconciler.is_alive() and not deleter.is_alive()
    assert not failures
    assert not deleted_during_submit
    assert launcher.cancelled == ["100"]
    assert not state.pools and not state.submissions
    restored = new_state(launcher, state_file=str(state_path))
    restored.reconcile_once()
    assert not restored.pools and not restored.submissions


def test_delete_pool_releases_an_inflight_assignment(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(launcher)
    state.create_pool({"name": "example", "desired_capacity": 1, "slots_per_job": 1})
    state.reconcile_once()
    state.request_lease("example", timeout_seconds=30)
    state.register_agent(
        {
            "agent_id": "example-agent",
            "pool": "example",
            "url": "http://agent:9000",
            "slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "node.example",
            "version": VERSION,
        }
    )
    reserving, finish_reserve, deleting, deleted = (threading.Event() for _ in range(4))
    failures, releases = [], []

    def request(_url, path, **kwargs):
        if path == "/v1/control/leases":
            reserving.set()
            assert finish_reserve.wait(timeout=5)
            return {
                "lease_id": "example-lease",
                "lease_token": "synthetic",
                "free_slots": 0,
            }
        assert path == "/v1/control/leases/example-lease"
        assert kwargs["method"] == "DELETE"
        releases.append(path)
        return {"released": True, "free_slots": 1}

    def assign():
        try:
            state._assign_pending()
        except BaseException as error:
            failures.append(error)

    def delete():
        deleting.set()
        try:
            state.delete_pool("example")
        except BaseException as error:
            failures.append(error)
        finally:
            deleted.set()

    monkeypatch.setattr("sandfleet.controller.request_json", request)
    assigner = threading.Thread(target=assign)
    deleter = threading.Thread(target=delete)
    assigner.start()
    try:
        assert reserving.wait(timeout=2)
        deleter.start()
        assert deleting.wait(timeout=2)
        deleted_during_reserve = deleted.wait(timeout=0.1)
    finally:
        finish_reserve.set()
        assigner.join(timeout=5)
        if deleter.ident is not None:
            deleter.join(timeout=5)
    assert not assigner.is_alive() and not deleter.is_alive()
    assert not failures
    assert not deleted_during_reserve
    assert releases == ["/v1/control/leases/example-lease"]
    assert launcher.cancelled == ["100"]
    assert not state.pools and not state.leases and not state.submissions


def orphaned_state(tmp_path, launcher):
    """A late submission persisted after an older controller deleted its pool."""
    state_path = tmp_path / "state.json"
    state = new_state(launcher, state_file=str(state_path))
    state.create_pool({"name": "example", "desired_capacity": 1, "slots_per_job": 1})
    state.reconcile_once()
    with state.lock:
        state.pools.pop("example")
        state._save_locked(force=True)
    return state_path, new_state(launcher, state_file=str(state_path))


def test_restart_reclaims_an_orphan_whose_worker_has_finished(tmp_path):
    launcher = FakeLauncher()
    state_path, restored = orphaned_state(tmp_path, launcher)
    launcher.states.clear()
    restored.reconcile_once()
    assert not restored.submissions
    assert not launcher.cancelled
    again = new_state(launcher, state_file=str(state_path))
    assert not again.pools and not again.submissions


def test_restart_cancels_only_the_orphaned_submission(tmp_path):
    launcher = FakeLauncher()
    state_path, restored = orphaned_state(tmp_path, launcher)
    restored.create_pool(
        {
            "name": "survivor",
            "min_capacity": 0,
            "warm_capacity": 0,
            "max_capacity": 1,
            "slots_per_job": 1,
        }
    )
    launcher.states["unrelated"] = {"unrelated": "RUNNING"}
    restored.reconcile_once()
    assert launcher.cancelled == ["100"]
    assert launcher.states == {"unrelated": {"unrelated": "RUNNING"}}
    assert set(restored.pools) == {"survivor"}
    assert not restored.submissions
    again = new_state(launcher, state_file=str(state_path))
    assert not again.submissions


def test_orphan_cancellation_failure_retains_durable_ownership(tmp_path, monkeypatch):
    launcher = FakeLauncher()
    state_path, restored = orphaned_state(tmp_path, launcher)
    cancel = launcher.cancel

    def fail_cancel(_ids):
        raise RuntimeError("synthetic cancellation failure")

    monkeypatch.setattr(launcher, "cancel", fail_cancel)
    with pytest.raises(RuntimeError, match="synthetic cancellation failure"):
        restored.reconcile_once()
    assert set(restored.submissions) == {"100"}
    again = new_state(launcher, state_file=str(state_path))
    assert set(again.submissions) == {"100"}
    monkeypatch.setattr(launcher, "cancel", cancel)
    again.reconcile_once()
    assert launcher.cancelled == ["100"]
    assert not again.submissions


def test_pool_name_cannot_adopt_a_worker_awaiting_orphan_cleanup(tmp_path):
    launcher = FakeLauncher()
    _state_path, restored = orphaned_state(tmp_path, launcher)
    with pytest.raises(ValueError, match="cleanup"):
        restored.create_pool(
            {"name": "example", "desired_capacity": 1, "slots_per_job": 1}
        )
    restored.reconcile_once()
    restored.create_pool({"name": "example", "desired_capacity": 1, "slots_per_job": 1})


def test_scheduler_failure_does_not_drop_or_cancel_orphans(tmp_path, monkeypatch):
    launcher = FakeLauncher()
    _state_path, restored = orphaned_state(tmp_path, launcher)
    monkeypatch.setattr(
        launcher,
        "snapshot",
        lambda _ids: SlurmSnapshot(ok=False, states={}, error="unavailable"),
    )
    restored.reconcile_once()
    assert set(restored.submissions) == {"100"}
    assert not launcher.cancelled


def test_elastic_requests_are_coalesced_into_one_worker_batch():
    launcher = FakeLauncher()
    state = new_state(launcher)
    created = state.create_pool(
        {
            "name": "rollouts",
            "min_capacity": 0,
            "warm_capacity": 0,
            "max_capacity": 512,
            "slots_per_job": 16,
            "cpus": 1,
            "memory_mb": 4096,
        }
    )
    assert created["target_capacity"] == 0
    assert launcher.snapshot_calls == 0  # API status never invokes Slurm.

    requests = [state.request_lease("rollouts", timeout_seconds=60) for _ in range(40)]
    assert all(request["status"] == "pending" for request in requests)
    assert state.pool_status("rollouts")["target_capacity"] == 40

    state.reconcile_once()
    assert launcher.submitted == [
        {
            "pool": "rollouts",
            "worker_count": 3,
            "slots": 16,
            "cpus": 1,
            "memory_mb": 4096,
        }
    ]

    # The three pending array tasks satisfy desired capacity. Reconciliation
    # does not submit duplicates while Slurm is busy scheduling them.
    state.reconcile_once()
    assert len(launcher.submitted) == 1


def test_declarative_resources_create_and_reuse_one_bounded_pool():
    launcher = FakeLauncher()
    state = new_state(
        launcher,
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=64,
            max_memory_mb_per_worker=131072,
            slots_per_job=8,
            max_capacity=64,
            warm_capacity=2,
            idle_timeout_seconds=30,
            max_pools=4,
        ),
    )

    first = state.request_lease(resources={"cpus": 3, "memory_mb": 6144})
    second = state.request_lease(resources={"cpus": 3, "memory_mb": 6144})

    assert first["pool"] == second["pool"] == "sfres-c3-m6144-gpu-none"
    assert first["resources"] == {"cpus": 3, "memory_mb": 6144}
    pool = state.pool_status(first["pool"])
    assert pool["dynamic"]
    assert pool["slots_per_job"] == 8
    assert pool["max_capacity"] == 64
    assert pool["warm_capacity"] == 2
    assert pool["target_capacity"] == 4
    assert len(state.pools) == 1

    state.reconcile_once()
    assert launcher.submitted == [
        {
            "pool": "sfres-c3-m6144-gpu-none",
            "worker_count": 1,
            "slots": 8,
            "cpus": 3,
            "memory_mb": 6144,
        }
    ]


def test_declarative_resources_are_explicit_and_policy_bounded():
    state = new_state(
        FakeLauncher(),
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=64,
            max_memory_mb_per_worker=131072,
            max_capacity=32,
            max_pools=1,
        ),
    )

    with pytest.raises(ValueError, match="exactly one"):
        state.request_lease()
    with pytest.raises(ValueError, match="exactly one"):
        state.request_lease("named", resources={"cpus": 1, "memory_mb": 512})
    with pytest.raises(ValueError, match="require cpus and memory_mb"):
        state.request_lease(resources={"cpus": 1})
    with pytest.raises(ValueError, match="require both gpu and type"):
        state.request_lease(resources={"cpus": 1, "memory_mb": 512, "gpu": 1})
    with pytest.raises(ValueError, match="GPU sandboxes are disabled"):
        state.request_lease(
            resources={
                "cpus": 1,
                "memory_mb": 512,
                "gpu": 1,
                "type": ["a100_80gb", "a100_40gb"],
            }
        )
    with pytest.raises(ValueError, match="memory-qualified"):
        state.request_lease(
            resources={
                "cpus": 1,
                "memory_mb": 512,
                "gpu": 1,
                "type": ["any"],
            }
        )
    with pytest.raises(ValueError, match="must be JSON integers"):
        state.request_lease(resources={"cpus": True, "memory_mb": 512})
    with pytest.raises(ValueError, match="cpus must be between"):
        state.request_lease(resources={"cpus": 65, "memory_mb": 512})
    state.request_lease(resources={"cpus": 1, "memory_mb": 512})
    with pytest.raises(ValueError, match="Dynamic pool limit"):
        state.request_lease(resources={"cpus": 2, "memory_mb": 512})


def test_gpu_request_uses_first_currently_free_exact_type_and_scales_gpu_pool():
    launcher = FakeLauncher()
    launcher.gpu_inventory = {
        "h100-80gb": {
            "configured": 8,
            "allocated": 8,
            "free_estimate": 0,
            "largest_node_free_estimate": 0,
            "schedulable_nodes": 1,
            "unavailable_nodes": 0,
        },
        "a100-40gb": {
            "configured": 4,
            "allocated": 2,
            "free_estimate": 2,
            "largest_node_free_estimate": 2,
            "schedulable_nodes": 1,
            "unavailable_nodes": 0,
        },
    }
    launcher.gpu_quota = {"limit": 304, "allocated": 27, "free_estimate": 277}
    catalog = GPUTypeCatalog.from_payload(
        {
            "types": {
                "h100_80gb": {"slurm_type": "h100-80gb", "memory_mb": 81920},
                "a100_40gb": {"slurm_type": "a100-40gb", "memory_mb": 40960},
            }
        }
    )
    state = new_state(
        launcher,
        max_gpus_per_sandbox=2,
        gpu_type_catalog=catalog,
        gpu_device_isolation_verified=True,
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=16,
            max_memory_mb_per_worker=65536,
            max_gpus_per_worker=4,
            slots_per_job=4,
            max_capacity=16,
            max_pools=4,
        ),
    )

    assert state.gpu_capabilities()["checked_at"] is None
    request = GPURequest(1, ("h100_80gb", "a100_40gb"))
    # Before discovery, placement falls back to the client's order.
    assert state._resolve_gpu_placement(request).logical_type == "h100_80gb"

    state.reconcile_once()
    state.gpu_capability_refresh_thread.join()
    capabilities = state.gpu_capabilities()
    assert capabilities["types"]["h100_80gb"]["inventory"]["free_estimate"] == 0
    assert capabilities["types"]["a100_40gb"]["scheduler_request_valid"] is True
    assert capabilities["qos_gpu_quota"]["free_estimate"] == 277
    assert launcher.gpu_inventory_calls == 1
    state.reconcile_once()
    assert launcher.gpu_inventory_calls == 1
    state.request_gpu_capability_refresh()
    state.reconcile_once()
    state.gpu_capability_refresh_thread.join()
    assert launcher.gpu_inventory_calls == 2

    request = state.request_lease(
        resources={
            "cpus": 2,
            "memory_mb": 4096,
            "gpu": 1,
            "type": ["h100_80gb", "a100_40gb"],
        }
    )
    assert request["resources"] == {
        "cpus": 2,
        "memory_mb": 4096,
        "gpu": 1,
        "type": ["a100_40gb"],
    }
    state.reconcile_once()
    assert launcher.submitted[-1] == {
        "pool": "sfres-c2-m4096-gpu1-a100_40gb",
        "worker_count": 1,
        "slots": 4,
        "cpus": 2,
        "memory_mb": 4096,
        "gpus": 1,
        "gpu_type": "a100-40gb",
    }


def gpu_state(launcher, **kwargs):
    catalog = GPUTypeCatalog.from_payload(
        {"types": {"a100_40gb": {"slurm_type": "a100-40gb", "memory_mb": 40960}}}
    )
    return new_state(
        launcher,
        max_gpus_per_sandbox=1,
        gpu_type_catalog=catalog,
        gpu_device_isolation_verified=True,
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=16,
            max_memory_mb_per_worker=65536,
            max_gpus_per_worker=4,
            slots_per_job=4,
            max_capacity=16,
            max_pools=4,
        ),
        **kwargs,
    )


def test_malformed_gpu_discovery_does_not_stop_the_controller():
    launcher = FakeLauncher()

    def malformed_inventory(gpu_types):
        raise KeyError("nodes")

    launcher.discover_gpu_inventory = malformed_inventory
    state = gpu_state(launcher)

    state.reconcile_once()
    state.gpu_capability_refresh_thread.join()

    assert "KeyError" in state.gpu_capabilities()["error"]
    request = state.request_lease(
        resources={"cpus": 2, "memory_mb": 4096, "gpu": 1, "type": ["a100_40gb"]}
    )
    assert request["resources"]["type"] == ["a100_40gb"]


def test_saved_gpu_pool_is_not_used_after_gpus_are_disabled(tmp_path):
    state_path = tmp_path / "state.json"
    launcher = FakeLauncher()
    state = gpu_state(launcher, state_file=str(state_path))
    request = state.request_lease(
        resources={"cpus": 2, "memory_mb": 4096, "gpu": 1, "type": ["a100_40gb"]}
    )
    pool_name = request["pool"]
    state._save(force=True)

    restored = new_state(
        launcher,
        state_file=str(state_path),
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=16,
            max_memory_mb_per_worker=65536,
            slots_per_job=4,
            max_capacity=16,
            max_pools=4,
        ),
    )
    submitted = len(launcher.submitted)
    restored.reconcile_once()

    assert len(launcher.submitted) == submitted
    assert "not enabled" in restored.pool_status(pool_name)["last_scale_error"]
    with pytest.raises(ValueError, match="not enabled"):
        restored.request_lease(pool_name)


@pytest.mark.parametrize("changed_field", ["gpu_count", "gpu_type", "slurm_gpu_type"])
def test_existing_resource_pool_rejects_gpu_identity_change(changed_field):
    state = new_state(
        FakeLauncher(),
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=8,
            max_memory_mb_per_worker=16384,
            slots_per_job=2,
            max_capacity=4,
            max_pools=2,
        ),
    )
    resources = {"cpus": 2, "memory_mb": 4096}
    pool_name = state._pool_for_resources(resources)
    setattr(
        state.pools[pool_name],
        changed_field,
        1 if changed_field == "gpu_count" else "other",
    )
    with pytest.raises(ValueError, match="Resource pool name collision"):
        state._pool_for_resources(resources)


def test_declarative_resources_must_be_enabled():
    state = new_state(FakeLauncher())
    with pytest.raises(ValueError, match="--enable-dynamic-resources"):
        state.request_lease(resources={"cpus": 1, "memory_mb": 512})


def test_dynamic_pool_packing_respects_worker_cpu_and_memory_limits():
    state = new_state(
        FakeLauncher(),
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=16,
            max_memory_mb_per_worker=32768,
            slots_per_job=16,
            max_capacity=32,
            max_pools=2,
            cpu_allocation_granularity=2,
        ),
    )

    cpu_limited = state.request_lease(resources={"cpus": 3, "memory_mb": 512})
    assert state.pool_status(cpu_limited["pool"])["slots_per_job"] == 4

    memory_limited = state.request_lease(resources={"cpus": 1, "memory_mb": 16384})
    assert state.pool_status(memory_limited["pool"])["slots_per_job"] == 1


def test_queued_lease_is_assigned_after_agent_registers(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(launcher)
    state.create_pool(
        {
            "name": "rollouts",
            "min_capacity": 0,
            "warm_capacity": 0,
            "max_capacity": 4,
            "slots_per_job": 2,
        }
    )
    request = state.request_lease("rollouts", timeout_seconds=60)
    assert request["status"] == "pending"
    state.reconcile_once()

    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "rollouts",
            "url": "http://agent:9000",
            "slots": 2,
            "free_slots": 2,
            "job_id": "101",
            "array_job_id": "100",
            "array_task_id": "0",
            "node": "n1",
            "version": VERSION,
        }
    )

    calls = []

    def fake_request(url, path, **kwargs):
        calls.append((url, path, kwargs))
        return {"lease_id": "lease-1", "lease_token": "lease-secret", "free_slots": 1}

    monkeypatch.setattr("sandfleet.controller.request_json", fake_request)
    state.reconcile_once()
    assigned = state.lease_request_status(request["request_id"])
    assert assigned["status"] == "assigned"
    assert assigned["lease"]["agent_url"] == "http://agent:9000"
    assert calls[0][1] == "/v1/control/leases"


def test_lost_worker_marks_lease_and_submits_bounded_replacement(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(launcher, agent_stale_seconds=1, agent_lost_seconds=2)
    state.create_pool(
        {
            "name": "fixed",
            "desired_capacity": 2,
            "slots_per_job": 2,
        }
    )
    state.reconcile_once()
    assert launcher.submitted[0]["worker_count"] == 1
    launcher.states["100"] = {"100": "RUNNING"}
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "fixed",
            "url": "http://agent:9000",
            "slots": 2,
            "free_slots": 2,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def fake_request(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {
                "lease_id": "lease-1",
                "lease_token": "lease-secret",
                "free_slots": 1,
            }
        return {"released": True, "free_slots": 2}

    monkeypatch.setattr("sandfleet.controller.request_json", fake_request)
    assert state.request_lease("fixed")["status"] == "assigned"
    state.agents["agent-1"].last_heartbeat = time.time() - 10

    state.reconcile_once()
    assert state.lease_status("lease-1")["status"] == "lost"
    assert "heartbeat" in state.lease_status("lease-1")["lost_reason"]
    assert "100" in launcher.cancelled
    assert len(launcher.submitted) == 2
    assert launcher.submitted[1]["worker_count"] == 1


def test_preempted_worker_resets_its_leases_without_contacting_agent(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(
        launcher,
        lease_ttl_seconds=10,
        agent_stale_seconds=1,
        agent_lost_seconds=2,
    )
    state.create_pool({"name": "preempt", "desired_capacity": 1, "slots_per_job": 1})
    state.reconcile_once()
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "preempt",
            "url": "http://agent:9000",
            "slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def reserve_without_releasing_dead_worker(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {
                "lease_id": "lease-1",
                "lease_token": "lease-secret",
                "free_slots": 0,
            }
        raise AssertionError("release called on a lost worker")

    monkeypatch.setattr(
        "sandfleet.controller.request_json", reserve_without_releasing_dead_worker
    )
    assert state.request_lease("preempt")["status"] == "assigned"
    state.leases["lease-1"].expires_at = time.time() - 1
    launcher.states.pop("100")
    state.agents["agent-1"].last_heartbeat = time.time() - 10

    def reject_scancel(_job_ids):
        raise AssertionError("scancel called for a worker absent from squeue")

    launcher.cancel = reject_scancel
    state.reconcile_once()

    agent = state.agents["agent-1"]
    assert agent.status == "lost"
    assert agent.free_slots == agent.ready_slots == 0
    assert not agent.cleanup_pending
    assert agent.cleanup_attempts == 0
    assert state.lease_status("lease-1")["status"] == "lost"
    assert "heartbeat" in state.lease_status("lease-1")["lost_reason"]
    assert state.pool_status("preempt")["active_leases"] == 0
    assert state.metrics["worker_preemptions_absorbed"] == 1
    assert len(launcher.submitted) == 2


def test_restarted_agent_does_not_cancel_its_live_worker():
    launcher = FakeLauncher()
    state = new_state(launcher)
    state.create_pool({"name": "restart", "desired_capacity": 1, "slots_per_job": 1})
    state.reconcile_once()
    launcher.states["100"] = {"100": "RUNNING"}
    registration = {
        "pool": "restart",
        "url": "http://agent:9000",
        "slots": 1,
        "free_slots": 1,
        "job_id": "100",
        "node": "n1",
        "version": VERSION,
    }
    state.register_agent({"agent_id": "old", **registration})
    state.register_agent({"agent_id": "new", **registration})

    state.reconcile_once()

    assert state.agents["old"].status == "lost"
    assert not state.agents["old"].cleanup_pending
    assert state.agents["new"].status == "active"
    assert launcher.cancelled == []


def test_failed_lost_worker_cleanup_is_persisted_and_retried(tmp_path):
    state_path = tmp_path / "controller-state.json"
    launcher = FakeLauncher()
    state = new_state(
        launcher,
        state_file=str(state_path),
        agent_stale_seconds=1,
        agent_lost_seconds=2,
    )
    state.create_pool({"name": "cleanup", "desired_capacity": 1, "slots_per_job": 1})
    state.reconcile_once()
    launcher.states["100"] = {"100": "RUNNING"}
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "cleanup",
            "url": "http://agent:9000",
            "slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )
    state.agents["agent-1"].last_heartbeat = time.time() - 10
    cancel = launcher.cancel
    attempts = 0

    def fail_once(job_ids):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("scheduler temporarily unavailable")
        cancel(job_ids)

    launcher.cancel = fail_once
    state.reconcile_once()

    agent = state.agents["agent-1"]
    assert agent.status == "lost"
    assert agent.cleanup_pending
    assert agent.cleanup_attempts == 1
    assert agent.last_cleanup_error == "scheduler temporarily unavailable"
    assert state.metrics["worker_cleanup_failures"] == 1

    agent.next_cleanup_at = 0
    state._save(force=True)
    restored = new_state(
        launcher,
        state_file=str(state_path),
        agent_stale_seconds=1,
        agent_lost_seconds=2,
    )
    restored.reconcile_once()

    restored_agent = restored.agents["agent-1"]
    assert not restored_agent.cleanup_pending
    assert restored_agent.cleanup_attempts == 2
    assert restored_agent.last_cleanup_error is None
    assert launcher.cancelled == ["100"]
    assert restored.metrics["worker_cleanup_failures"] == 1
    assert restored.metrics["worker_cleanups_completed"] == 1


def test_controller_termination_counter_survives_restart(tmp_path):
    state_path = tmp_path / "controller-state.json"
    state = new_state(FakeLauncher(), state_file=str(state_path))
    error = RuntimeError("unexpected reconciler failure")

    state._record_reconcile_termination(error)

    assert state.reconcile_error is error
    assert state.metrics["controller_terminations"] == 1
    restored = new_state(FakeLauncher(), state_file=str(state_path))
    assert restored.metrics["controller_terminations"] == 1


def test_idle_surplus_worker_is_drained_and_cancelled():
    launcher = FakeLauncher()
    state = new_state(launcher)
    state.create_pool(
        {
            "name": "scale-zero",
            "min_capacity": 0,
            "warm_capacity": 0,
            "max_capacity": 8,
            "slots_per_job": 8,
            "idle_timeout_seconds": 0,
        }
    )
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "scale-zero",
            "url": "http://agent:9000",
            "slots": 8,
            "free_slots": 8,
            "job_id": "55",
            "node": "n1",
            "version": VERSION,
        }
    )
    state.reconcile_once()
    assert launcher.cancelled == ["55"]
    assert state.agents["agent-1"].status == "draining"


def test_failed_scale_down_restores_worker_placement_state(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(launcher)
    state.create_pool(
        {
            "name": "scale-zero",
            "min_capacity": 0,
            "warm_capacity": 0,
            "max_capacity": 8,
            "slots_per_job": 8,
            "idle_timeout_seconds": 0,
        }
    )
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "scale-zero",
            "url": "http://agent:9000",
            "slots": 8,
            "free_slots": 8,
            "job_id": "55",
            "node": "n1",
            "version": VERSION,
        }
    )

    def fail_cancel(_job_ids):
        raise RuntimeError("busy")

    monkeypatch.setattr(launcher, "cancel", fail_cancel)

    with pytest.raises(RuntimeError, match="busy"):
        state._scale_down(time.time())

    agent = state.agents["agent-1"]
    assert agent.status == "active"
    assert agent.free_slots == 8


def test_durable_desired_state_survives_controller_restart(tmp_path):
    state_path = tmp_path / "controller-state.json"
    launcher = FakeLauncher()
    first = new_state(launcher, state_file=str(state_path))
    first.create_pool(
        {
            "name": "durable",
            "min_capacity": 2,
            "warm_capacity": 2,
            "max_capacity": 16,
            "slots_per_job": 2,
        }
    )

    restored = new_state(FakeLauncher(), state_file=str(state_path))
    status = restored.pool_status("durable")
    assert status["min_capacity"] == 2
    assert status["max_capacity"] == 16
    assert status["target_capacity"] == 2


def test_corrupt_primary_recovers_previous_controller_generation(tmp_path):
    state_path = tmp_path / "controller-state.json"
    first = new_state(FakeLauncher(), state_file=str(state_path))
    first.create_pool({"name": "stable", "desired_capacity": 1, "slots_per_job": 1})
    first.create_pool({"name": "newest", "desired_capacity": 1, "slots_per_job": 1})
    state_path.write_bytes(b"x" * 1_048_576)

    restored = new_state(FakeLauncher(), state_file=str(state_path))

    assert restored.pool_status("stable")["target_capacity"] == 1
    assert "newest" not in restored.pools
    assert restored.metrics["state_recoveries"] == 1
    assert restored.state_recovery is not None
    quarantined = restored.state_recovery["quarantined"]
    assert quarantined is not None
    assert Path(quarantined).read_bytes() == b"x" * 1_048_576
    assert restored.metrics_snapshot()["persistence"]["last_recovery"] is not None
    assert (
        json.loads(state_path.read_text())["durable_metrics"]["state_recoveries"] == 1
    )


def test_version_four_state_migrates_state_recovery_metric(tmp_path):
    state_path = tmp_path / "controller-state.json"
    first = new_state(FakeLauncher(), state_file=str(state_path))
    first.create_pool({"name": "legacy", "desired_capacity": 1, "slots_per_job": 1})
    payload = json.loads(state_path.read_text())
    payload["version"] = 4
    payload["durable_metrics"].pop("state_recoveries")
    state_path.write_text(json.dumps(payload))

    restored = new_state(FakeLauncher(), state_file=str(state_path))

    assert restored.pool_status("legacy")["target_capacity"] == 1
    assert restored.metrics["state_recoveries"] == 0
    restored._save(force=True)
    assert (
        json.loads(state_path.read_text())["version"] == ControllerState.STATE_VERSION
    )


def test_version_two_state_migrates_operator_pools(tmp_path):
    state_path = tmp_path / "controller-state.json"
    first = new_state(FakeLauncher(), state_file=str(state_path))
    first.create_pool({"name": "legacy", "desired_capacity": 1, "slots_per_job": 1})
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["version"] = 2
    for pool in payload["pools"]:
        pool.pop("dynamic")
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = new_state(FakeLauncher(), state_file=str(state_path))
    assert not restored.pool_status("legacy")["dynamic"]


def test_version_three_state_migrates_lost_worker_cleanup(tmp_path):
    state_path = tmp_path / "controller-state.json"
    first = new_state(FakeLauncher(), state_file=str(state_path))
    first.create_pool({"name": "legacy", "desired_capacity": 1, "slots_per_job": 1})
    first.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "legacy",
            "url": "http://agent:9000",
            "slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )
    first._mark_agent_lost_locked(
        first.agents["agent-1"], "preempted", cleanup_worker=True
    )
    first._save(force=True)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["version"] = 3
    payload.pop("durable_metrics")
    for agent in payload["agents"]:
        for field in (
            "cleanup_pending",
            "cleanup_attempts",
            "next_cleanup_at",
            "last_cleanup_error",
        ):
            agent.pop(field)
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = new_state(FakeLauncher(), state_file=str(state_path))
    assert restored.agents["agent-1"].cleanup_pending


def gate_registration(**overrides):
    return {
        "agent_id": "agent-1",
        "pool": "legacy",
        "url": "http://agent:9000",
        "slots": 1,
        "free_slots": 1,
        "job_id": "100",
        "node": "n1",
        "version": VERSION,
    } | overrides


@pytest.mark.parametrize("version", [None, "0.8.0", "0.8", "0.8.9rc1", "dev", 9])
def test_agents_older_than_the_network_policy_are_rejected(version):
    state = new_state(FakeLauncher())
    state.create_pool({"name": "legacy", "desired_capacity": 1, "slots_per_job": 1})
    payload = gate_registration(version=version)
    if version is None:
        payload.pop("version")
    with pytest.raises(ValueError, match="older than 0.9.0"):
        state.register_agent(payload)
    assert state.agents == {}
    state.register_agent(gate_registration(version="0.10.0"))
    assert state.agents["agent-1"].status == "active"


@pytest.mark.parametrize("version", [VERSION, "0.9", "0.9.0rc1", "0.9.1.dev3", "1"])
def test_current_and_suffixed_agent_versions_pass_the_gate(version):
    check_agent_version(version)


@pytest.mark.parametrize("version", [7, ControllerState.STATE_VERSION])
def test_agents_recovered_from_pre_gate_state_are_lost(tmp_path, version):
    state_path = tmp_path / "controller-state.json"
    first = new_state(FakeLauncher(), state_file=str(state_path))
    first.create_pool({"name": "legacy", "desired_capacity": 1, "slots_per_job": 1})
    first.register_agent(gate_registration())
    first._save(force=True)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["version"] = version
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = new_state(FakeLauncher(), state_file=str(state_path))

    agent = restored.agents["agent-1"]
    if version < 8:
        assert (agent.status, agent.cleanup_pending) == ("lost", True)
        assert restored.heartbeat("agent-1", {}) == {"ok": False, "status": "lost"}
    else:
        assert (agent.status, agent.cleanup_pending) == ("stale", False)


def test_scheduler_polling_is_rate_limited_even_when_reconcile_is_woken():
    launcher = FakeLauncher()
    state = ControllerState(
        token="secret",
        client_token="client-secret",
        metrics_token="metrics-secret",
        launcher=launcher,
        max_capacity=64,
        max_cpus_per_sandbox=8,
        max_memory_mb_per_sandbox=65536,
        min_scheduler_poll_seconds=60,
    )
    state.create_pool(
        {
            "name": "quiet",
            "min_capacity": 0,
            "warm_capacity": 0,
            "max_capacity": 8,
            "slots_per_job": 8,
        }
    )
    state.reconcile_once()
    state.reconcile_once()
    state.reconcile_once()
    assert launcher.snapshot_calls == 1


def test_repeated_worker_boot_failures_open_circuit_until_resumed():
    launcher = FakeLauncher()
    state = new_state(launcher, max_scale_failures=1)
    state.create_pool({"name": "broken", "desired_capacity": 2, "slots_per_job": 2})
    state.reconcile_once()
    assert len(launcher.submitted) == 1

    # The array vanished without any agent ever registering.
    launcher.states.pop("100")
    state.reconcile_once()
    assert len(launcher.submitted) == 1
    assert state.pool_status("broken")["scaling_suspended"]

    state.resume_pool("broken")
    state.reconcile_once()
    assert len(launcher.submitted) == 2


def test_step_start_permits_are_globally_rate_limited():
    state = new_state(FakeLauncher(), step_start_rate=1, step_start_burst=2)
    assert state.acquire_step_permit()["granted"]
    assert state.acquire_step_permit()["granted"]
    delayed = state.acquire_step_permit()
    assert not delayed["granted"]
    assert 0 < delayed["retry_after_seconds"] <= 1
    assert state.metrics["step_permits_granted"] == 2
    assert state.metrics["step_permits_delayed"] == 1


def test_metrics_report_prepared_capacity_not_configured_slots():
    state = new_state(FakeLauncher())
    state.create_pool({"name": "observed", "desired_capacity": 4, "slots_per_job": 4})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "observed",
            "url": "http://agent:9000",
            "slots": 4,
            "ready_slots": 2,
            "free_slots": 2,
            "preparing_slots": 2,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )
    metrics = state.metrics_snapshot()
    assert metrics["version"] == VERSION
    assert metrics["totals"]["ready_capacity"] == 2
    assert metrics["pools"][0]["ready_capacity"] == 2
    assert (
        'sandfleet_pool_ready_capacity{pool="observed"} 2' in state.prometheus_metrics()
    )


def test_expired_lease_is_released_and_recycled(monkeypatch):
    state = new_state(FakeLauncher(), lease_ttl_seconds=10)
    state.create_pool({"name": "ttl", "desired_capacity": 1, "slots_per_job": 1})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "ttl",
            "url": "http://agent:9000",
            "slots": 1,
            "ready_slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def fake_request(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {
                "lease_id": "lease-ttl",
                "lease_token": "lease-secret",
                "free_slots": 0,
            }
        return {"released": True, "free_slots": 1}

    monkeypatch.setattr("sandfleet.controller.request_json", fake_request)
    assert state.request_lease("ttl")["status"] == "assigned"
    state.leases["lease-ttl"].expires_at = time.time() - 1
    state._reap_agents_and_requests(time.time(), scheduler_ok=True)
    assert "lease-ttl" not in state.leases
    assert state.metrics["leases_expired"] == 1
    assert state.agents["agent-1"].free_slots == 1


def test_expired_lease_transport_failure_does_not_stop_reconciliation(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(
        launcher,
        lease_ttl_seconds=10,
        agent_stale_seconds=1,
        agent_lost_seconds=2,
    )
    state.create_pool({"name": "ttl", "desired_capacity": 1, "slots_per_job": 1})
    state.create_pool({"name": "other", "desired_capacity": 1, "slots_per_job": 1})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "ttl",
            "url": "http://agent:9000",
            "slots": 1,
            "ready_slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def reserve_then_disconnect(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {
                "lease_id": "lease-ttl",
                "lease_token": "lease-secret",
                "free_slots": 0,
            }
        raise SandfleetError("agent unavailable")

    monkeypatch.setattr("sandfleet.controller.request_json", reserve_then_disconnect)
    assert state.request_lease("ttl")["status"] == "assigned"
    state.leases["lease-ttl"].expires_at = time.time() - 1
    state.request_lease("other")

    state.reconcile_once()

    assert state.lease_status("lease-ttl")["status"] == "active"
    assert state.metrics["leases_expired"] == 0
    assert state.metrics["lease_release_transport_failures"] == 1
    assert launcher.submitted[-1]["pool"] == "other"

    state.agents["agent-1"].last_heartbeat = time.time() - 3
    state.reconcile_once()

    assert state.lease_status("lease-ttl")["status"] == "lost"
    assert state.metrics["worker_cleanups_completed"] == 1


@pytest.mark.parametrize("status", [500, 503])
def test_expired_lease_protocol_failure_remains_fatal(monkeypatch, status):
    state = new_state(FakeLauncher(), lease_ttl_seconds=10)
    state.create_pool({"name": "ttl", "desired_capacity": 1, "slots_per_job": 1})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "ttl",
            "url": "http://agent:9000",
            "slots": 1,
            "ready_slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def reserve_then_reject(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {
                "lease_id": "lease-ttl",
                "lease_token": "lease-secret",
                "free_slots": 0,
            }
        raise SandfleetError("agent rejected cleanup", status=status)

    monkeypatch.setattr("sandfleet.controller.request_json", reserve_then_reject)
    assert state.request_lease("ttl")["status"] == "assigned"
    state.leases["lease-ttl"].expires_at = time.time() - 1

    with pytest.raises(SandfleetError, match="agent rejected cleanup"):
        state.reconcile_once()

    assert state.lease_status("lease-ttl")["status"] == "active"
    assert state.metrics["lease_release_transport_failures"] == 0


@pytest.mark.parametrize(
    "cleanup_error",
    [
        SandfleetError(
            "execution still draining", status=503, error_type="LeaseCleanupPending"
        ),
        SandboxOOMError(
            "Sandbox exceeded its Slurm step memory limit",
            status=500,
            error_type="SandboxOOMError",
        ),
    ],
)
def test_expired_lease_cleanup_pending_retries_without_losing_assignment(
    monkeypatch, cleanup_error
):
    launcher = FakeLauncher()
    state = new_state(launcher, lease_ttl_seconds=10)
    state.create_pool({"name": "ttl", "desired_capacity": 1, "slots_per_job": 1})
    state.create_pool({"name": "other", "desired_capacity": 1, "slots_per_job": 1})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "ttl",
            "url": "http://agent:9000",
            "slots": 1,
            "ready_slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )
    pending = True

    def reserve_then_cleanup(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {
                "lease_id": "lease-ttl",
                "lease_token": "lease-secret",
                "free_slots": 0,
            }
        if pending:
            raise cleanup_error
        return {"released": True, "free_slots": 1}

    monkeypatch.setattr("sandfleet.controller.request_json", reserve_then_cleanup)
    request = state.request_lease("ttl")
    state.leases["lease-ttl"].expires_at = time.time() - 1
    for attempt in range(1, 4):
        state.reconcile_once()
        assert state.lease_status("lease-ttl")["status"] == "active"
        assert state.lease_requests[request["request_id"]].status == "assigned"
        assert state.metrics["lease_release_cleanup_pending"] == attempt
        assert state.metrics["leases_expired"] == 0
        assert state.metrics["lease_release_transport_failures"] == 0
    assert any(item["pool"] == "other" for item in launcher.submitted)
    pending = False
    state.reconcile_once()
    assert "lease-ttl" not in state.leases
    assert state.lease_requests[request["request_id"]].status == "released"
    assert state.metrics["leases_expired"] == 1


def test_pool_mode_is_explicit():
    state = new_state(FakeLauncher())
    with pytest.raises(ValueError, match="exactly one"):
        state.create_pool({"name": "ambiguous", "slots_per_job": 1})
    with pytest.raises(ValueError, match="exactly one"):
        state.create_pool(
            {
                "name": "ambiguous",
                "desired_capacity": 1,
                "max_capacity": 2,
                "slots_per_job": 1,
            }
        )


def test_failed_agent_cleanup_keeps_lease_releasable(monkeypatch):
    state = new_state(FakeLauncher())
    state.create_pool({"name": "release", "desired_capacity": 1, "slots_per_job": 1})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "release",
            "url": "http://agent:9000",
            "slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def reserve_then_fail(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {"lease_id": "lease-1", "lease_token": "secret", "free_slots": 0}
        assert state.lease_status("lease-1")["status"] == "active"
        raise SandfleetError("agent unavailable")

    monkeypatch.setattr("sandfleet.controller.request_json", reserve_then_fail)
    assert state.request_lease("release")["status"] == "assigned"
    with pytest.raises(SandfleetError, match="agent unavailable"):
        state.release("lease-1")
    assert state.lease_status("lease-1")["status"] == "active"


def test_failed_request_cancellation_remains_retryable(monkeypatch):
    state = new_state(FakeLauncher())
    state.create_pool({"name": "cancel", "desired_capacity": 1, "slots_per_job": 1})
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "cancel",
            "url": "http://agent:9000",
            "slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )
    cleanup_attempts = 0

    def reserve_then_retry_cleanup(_url, _path, **kwargs):
        nonlocal cleanup_attempts
        if kwargs["method"] == "POST":
            return {"lease_id": "lease-1", "lease_token": "secret", "free_slots": 0}
        cleanup_attempts += 1
        if cleanup_attempts == 1:
            raise SandfleetError("agent unavailable")
        return {"released": True, "free_slots": 1}

    monkeypatch.setattr("sandfleet.controller.request_json", reserve_then_retry_cleanup)
    request = state.request_lease("cancel")

    with pytest.raises(SandfleetError, match="agent unavailable"):
        state.cancel_lease_request(request["request_id"])
    assert state.lease_request_status(request["request_id"])["status"] == "assigned"
    assert state.lease_status("lease-1")["status"] == "active"

    assert state.cancel_lease_request(request["request_id"]) == {"cancelled": True}
    assert state.lease_request_status(request["request_id"])["status"] == "cancelled"
    with pytest.raises(KeyError):
        state.lease_status("lease-1")


def test_cleanup_cancels_every_pool_despite_an_unreachable_agent(monkeypatch):
    launcher = FakeLauncher()
    state = new_state(launcher)
    for name in ("dead", "live"):
        state.create_pool({"name": name, "desired_capacity": 1, "slots_per_job": 1})
    state._scale_up_once(time.time())
    state.last_submission_at = 0
    state._scale_up_once(time.time())
    assert len(state.submissions) == 2
    state.register_agent(
        {
            "agent_id": "agent-1",
            "pool": "dead",
            "url": "http://gone:9000",
            "slots": 1,
            "ready_slots": 1,
            "free_slots": 1,
            "job_id": "100",
            "node": "n1",
            "version": VERSION,
        }
    )

    def reserve_then_refuse(_url, _path, **kwargs):
        if kwargs["method"] == "POST":
            return {"lease_id": "lease-1", "lease_token": "secret", "free_slots": 0}
        raise SandfleetError("unreachable") from ConnectionRefusedError()

    monkeypatch.setattr("sandfleet.controller.request_json", reserve_then_refuse)
    assert state.request_lease("dead")["status"] == "assigned"
    with pytest.raises(SandfleetError, match="unreachable"):
        state.close(cleanup_jobs=True)
    assert sorted(launcher.cancelled) == ["100", "101"]
    assert "live" not in state.pools
    assert "dead" in state.pools


def test_internet_leases_are_opt_in_and_operator_enabled(monkeypatch):
    launcher = FakeLauncher()
    launcher.settings = SimpleNamespace(internet=EgressSettings())
    state = new_state(launcher)
    state.create_pool(
        {"name": "rollouts", "min_capacity": 0, "max_capacity": 2, "slots_per_job": 2}
    )
    with pytest.raises(ValueError, match="disabled"):
        state.request_lease("rollouts", internet=True)
    with pytest.raises(ValueError, match="boolean"):
        state.request_lease("rollouts", internet="true")

    launcher.settings.internet = EgressSettings(mode="allowlist")
    state.register_agent(agent_payload("agent-1", internet=True))
    payloads = []

    def fake_request(_url, _path, **kwargs):
        payloads.append(kwargs["payload"])
        index = len(payloads)
        return {"lease_id": f"lease-{index}", "lease_token": "t", "free_slots": 1}

    monkeypatch.setattr("sandfleet.controller.request_json", fake_request)
    online = state.request_lease("rollouts", internet=True)
    offline = state.request_lease("rollouts")

    assert payloads == [{"internet": True}, {"internet": False}]
    assert online["internet"] and online["lease"]["internet"]
    assert not offline["internet"] and not offline["lease"]["internet"]
    assert state.metrics["internet_leases_created"] == 1


def agent_payload(agent_id, *, internet, job_id="101"):
    payload = {
        "agent_id": agent_id,
        "pool": "rollouts",
        "url": f"http://{agent_id}:9000",
        "slots": 2,
        "free_slots": 2,
        "job_id": job_id,
        "array_job_id": None,
        "array_task_id": None,
        "node": "n1",
        "version": VERSION,
    }
    return payload if internet is None else {**payload, "internet": internet}


def test_internet_leases_are_placed_only_on_capable_agents(monkeypatch):
    launcher = FakeLauncher()
    launcher.settings = SimpleNamespace(internet=EgressSettings(mode="any"))
    state = new_state(launcher)
    state.create_pool(
        {"name": "rollouts", "min_capacity": 0, "max_capacity": 4, "slots_per_job": 2}
    )
    # A pre-0.8 agent omits the capability and must not receive internet leases.
    state.register_agent(agent_payload("old", internet=None, job_id="101"))
    state.register_agent(agent_payload("new", internet=True, job_id="102"))
    with pytest.raises(ValueError, match="boolean"):
        state.register_agent(agent_payload("bad", internet="yes", job_id="103"))
    assert not state.agents["old"].internet
    urls = []

    def fake_request(url, _path, **kwargs):
        urls.append((url, kwargs["payload"]["internet"]))
        free = 2 - sum(seen == url for seen, _ in urls)
        return {
            "lease_id": f"lease-{len(urls)}",
            "lease_token": "t",
            "free_slots": free,
        }

    monkeypatch.setattr("sandfleet.controller.request_json", fake_request)
    for _ in range(2):
        assert state.request_lease("rollouts", internet=True)["status"] == "assigned"
    assert state.request_lease("rollouts", internet=True)["status"] == "pending"
    assert state.request_lease("rollouts")["status"] == "assigned"
    assert urls == [
        ("http://new:9000", True),
        ("http://new:9000", True),
        ("http://old:9000", False),
    ]


def test_agent_rejection_fails_one_request_without_blocking_others(monkeypatch):
    launcher = FakeLauncher()
    launcher.settings = SimpleNamespace(internet=EgressSettings(mode="any"))
    state = new_state(launcher)
    state.create_pool(
        {"name": "rollouts", "min_capacity": 0, "max_capacity": 2, "slots_per_job": 2}
    )
    online = state.request_lease("rollouts", internet=True)
    offline = state.request_lease("rollouts")

    def fake_request(_url, _path, **kwargs):
        if kwargs["payload"]["internet"]:
            raise SandfleetError("not enabled on this worker", status=400)
        return {"lease_id": "lease-1", "lease_token": "t", "free_slots": 1}

    monkeypatch.setattr("sandfleet.controller.request_json", fake_request)
    state.register_agent(agent_payload("agent-1", internet=True))
    state.reconcile_once()

    failed = state.lease_request_status(online["request_id"])
    assert failed["status"] == "failed"
    assert "not enabled" in failed["error"]
    assert state.lease_request_status(offline["request_id"])["status"] == "assigned"


def test_version_six_state_migrates_leases_offline(tmp_path):
    state_path = tmp_path / "controller-state.json"
    first = new_state(FakeLauncher(), state_file=str(state_path))
    first.create_pool({"name": "legacy", "desired_capacity": 1, "slots_per_job": 1})
    first.request_lease("legacy")
    first._save(force=True)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["version"] = 6
    for request in payload["lease_requests"]:
        request.pop("internet")
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = new_state(FakeLauncher(), state_file=str(state_path))
    assert [request.internet for request in restored.lease_requests.values()] == [False]
