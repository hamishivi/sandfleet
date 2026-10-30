from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from sandfleet import recovery
from sandfleet.errors import SandfleetError
from sandfleet.controller import write_registry
from sandfleet.recovery import controller_request, controller_url
from sandfleet.resources import DynamicPoolPolicy
from test_controller import FakeLauncher, new_state


def test_renew_rediscovers_moved_controller(tmp_path, monkeypatch):
    registry = tmp_path / "endpoint.json"
    write_registry(str(registry), "http://old")
    seen = []

    def transport(url, path, **kwargs):
        seen.append(url)
        if url == "http://old":
            write_registry(str(registry), "http://new")
            raise SandfleetError("unavailable") from ConnectionRefusedError()
        return {"renewed": True}

    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    assert controller_request(
        transport,
        "http://old",
        "/v1/leases/x/renew",
        registry=str(registry),
        method="POST",
    ) == {"renewed": True}
    assert seen == ["http://old", "http://new"]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_permanent_errors_never_retry(status):
    calls = []

    def transport(*args, **kwargs):
        calls.append(args)
        raise SandfleetError("strict failure", status=status)

    with pytest.raises(SandfleetError):
        controller_request(transport, "http://old", "/v1/leases/x/renew", method="POST")
    assert len(calls) == 1


def test_ambiguous_acquisition_is_not_replayed():
    calls = []

    def transport(*args, **kwargs):
        calls.append(args)
        raise SandfleetError("lost response") from ConnectionResetError()

    with pytest.raises(SandfleetError):
        controller_request(transport, "http://old", "/v1/lease-requests", method="POST")
    assert len(calls) == 1


def test_retry_is_bounded_and_interruptible(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        recovery.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt)
    )
    calls = []

    def transport(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise SandfleetError("down") from ConnectionRefusedError()

    with pytest.raises(SandfleetError):
        controller_request(
            transport,
            "http://old",
            "/v1/leases/x/renew",
            method="POST",
            retry_seconds=2,
        )
    assert now[0] == 2
    assert max(calls) <= 2
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError):
        controller_request(transport, "http://old", "/v1/health", stop=stop)


def test_registry_missing_is_transient_but_malformed_is_not(tmp_path):
    path = tmp_path / "endpoint.json"
    assert controller_url("http://old", str(path)) == "http://old"
    path.write_text("{")
    with pytest.raises(ValueError):
        controller_url("http://old", str(path))
    path.write_text(json.dumps({"url": "http://user:password@elsewhere"}))
    with pytest.raises(ValueError):
        controller_url("http://old", str(path))


def test_restored_dynamic_pool_reuses_saved_spec_under_new_defaults(tmp_path):
    path = tmp_path / "state.json"
    old_policy = DynamicPoolPolicy(
        max_cpus_per_worker=96,
        max_memory_mb_per_worker=393216,
        slots_per_job=48,
        max_capacity=640,
    )
    state = new_state(
        FakeLauncher(), state_file=str(path), dynamic_pool_policy=old_policy
    )
    first = state.request_lease(resources={"cpus": 2, "memory_mb": 8192})
    saved_slots = state.pool_status(first["pool"])["slots_per_job"]
    state.close(cleanup_jobs=False)
    policy = DynamicPoolPolicy(
        max_cpus_per_worker=128,
        max_memory_mb_per_worker=524288,
        slots_per_job=64,
        max_capacity=2048,
    )
    restored = new_state(
        FakeLauncher(), state_file=str(path), dynamic_pool_policy=policy
    )
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(
                    lambda _: restored.request_lease(
                        resources={"cpus": 2, "memory_mb": 8192}
                    ),
                    range(16),
                )
            )
        assert {r["pool"] for r in results} == {first["pool"]}
        assert len(restored.pools) == 1
        pool = restored.pool_status(first["pool"])
        assert pool["max_capacity"] == 640
        assert pool["slots_per_job"] == saved_slots
        restored.pools[first["pool"]].dynamic = False
        with pytest.raises(ValueError, match="collision"):
            restored.request_lease(resources={"cpus": 2, "memory_mb": 8192})
    finally:
        restored.close(cleanup_jobs=False)


def test_worker_heartbeat_discovers_replacement(tmp_path, monkeypatch):
    from sandfleet import agent
    from unittest.mock import Mock

    registry = tmp_path / "endpoint.json"
    write_registry(str(registry), "http://new")
    monkeypatch.setenv("SANDFLEET_REGISTRY", str(registry))
    stop = Mock()
    stop.wait.side_effect = [False, True]
    calls = []

    def request(url, path, **kwargs):
        calls.append((url, path))
        return {"ok": True}

    monkeypatch.setattr(agent, "request_json", request)
    state = Mock()
    state.capacity.return_value = {}
    agent.heartbeat_loop(stop, "http://old", "secret", "worker", state, {})
    assert calls == [("http://new", "/v1/agents/worker/heartbeat")]


def run_heartbeats(monkeypatch, outcomes, timeout_seconds):
    """Drive heartbeat_loop over scripted outcomes, advancing a fake clock 600s each."""
    from unittest.mock import Mock

    from sandfleet import agent

    clock = iter(range(0, 10**6, 600))
    monkeypatch.setattr(agent.time, "monotonic", lambda: next(clock))
    outcomes = iter(outcomes)

    def request(url, path, **kwargs):
        if next(outcomes) == "down":
            raise SandfleetError("down") from ConnectionRefusedError()
        return {"ok": True}

    monkeypatch.setattr(agent, "request_json", request)
    stop = Mock()
    stop.wait.side_effect = lambda _: False
    state = Mock()
    state.capacity.return_value = {}
    agent.heartbeat_loop(
        stop, "http://controller", "secret", "worker", state, {}, timeout_seconds
    )


def test_worker_exits_when_controller_is_unreachable_for_a_lease_ttl(monkeypatch):
    with pytest.raises(RuntimeError, match="No successful controller heartbeat"):
        run_heartbeats(monkeypatch, ["ok"] + ["down"] * 20, timeout_seconds=3600)


def test_worker_survives_outages_shorter_than_the_timeout(monkeypatch):
    # Each outage is shorter than an hour, then a heartbeat succeeds and resets the clock.
    outcomes = (["down"] * 4 + ["ok"]) * 3
    with pytest.raises(StopIteration):  # ran out of scripted outcomes without exiting
        run_heartbeats(monkeypatch, outcomes, timeout_seconds=3600)


def test_zero_controller_loss_timeout_never_exits(monkeypatch):
    with pytest.raises(StopIteration):
        run_heartbeats(monkeypatch, ["down"] * 30, timeout_seconds=0)


def test_step_permit_outage_retries_without_bypassing_grant(tmp_path, monkeypatch):
    from sandfleet import agent

    registry = tmp_path / "endpoint.json"
    write_registry(str(registry), "http://old")
    monkeypatch.setenv("SANDFLEET_REGISTRY", str(registry))
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda _: False)
    seen = []

    def request(url, path, **kwargs):
        seen.append(url)
        if len(seen) == 1:
            write_registry(str(registry), "http://new")
            raise SandfleetError("down") from ConnectionRefusedError()
        return {"granted": True}

    monkeypatch.setattr(agent, "request_json", request)
    agent.step_permit_acquirer("http://old", "secret", stop)()
    assert seen == ["http://old", "http://new"]


def test_refused_acquisition_can_retry_before_any_delivery(monkeypatch):
    calls = []

    def transport(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise SandfleetError("refused") from ConnectionRefusedError()
        return {"request_id": "only-one"}

    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    assert controller_request(
        transport, "http://old", "/v1/lease-requests", method="POST"
    ) == {"request_id": "only-one"}
    assert len(calls) == 2


@pytest.mark.parametrize("payload", [[], {}, {"url": 1}, {"url": None}])
def test_registry_schema_errors_are_explicit(tmp_path, payload):
    path = tmp_path / "endpoint.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="string URL"):
        controller_url("http://old", str(path))


def test_permit_queue_wait_does_not_consume_outage_budget(monkeypatch):
    from sandfleet import agent

    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(agent.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))
    calls = []

    def request(*args, **kwargs):
        calls.append(now[0])
        if len(calls) in {1, 3}:
            return {"granted": False, "retry_after_seconds": 301}
        if len(calls) in {2, 4}:
            raise SandfleetError("down") from ConnectionRefusedError()
        return {"granted": True}

    monkeypatch.setattr(agent, "request_json", request)
    agent.step_permit_acquirer("http://controller", "token", stop)()
    assert calls == [0, 301, 303, 604, 606]


def test_continuous_permit_outage_still_expires(monkeypatch):
    from sandfleet import agent

    now = [0.0]
    stop = threading.Event()
    monkeypatch.setattr(agent.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(stop, "wait", lambda dt: now.__setitem__(0, now[0] + dt))

    def request(*args, **kwargs):
        raise SandfleetError("down") from ConnectionRefusedError()

    monkeypatch.setattr(agent, "request_json", request)
    with pytest.raises(SandfleetError, match="down"):
        agent.step_permit_acquirer("http://controller", "token", stop)()
    assert now[0] == 300
