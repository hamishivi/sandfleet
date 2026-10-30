from __future__ import annotations

import pytest
import threading

from sandfleet.client import SandboxLease, SandfleetClient


@pytest.fixture
def lease():
    return SandboxLease(
        controller=SandfleetClient("http://controller", "client"),
        lease_id="owned-lease",
        lease_token="private",
        agent_url="http://agent",
        pool="test",
    )


@pytest.mark.parametrize("operation", ["start", "restart"])
@pytest.mark.parametrize(
    "storage",
    [None, {"mode": "writable-tmpfs"}, {"mode": "disk-overlay", "storage_mb": 4096}],
)
def test_disk_client_rejects_downgrade_and_retains_lease(
    monkeypatch, lease, operation, storage
):
    calls = []

    def request(path, **kwargs):
        calls.append(path)
        return {"storage": storage}

    monkeypatch.setattr(lease, "_request", request)
    lease.storage_mb = 2048
    lease.image_sha256 = "a" * 64
    with pytest.raises(RuntimeError, match="disk volume"):
        if operation == "start":
            lease.start("/image.sif", storage_mb=2048, image_sha256="a" * 64)
        else:
            lease.restart("/image.sif")
    assert calls == [f"/{operation}"]
    assert not lease.closed


def test_disk_client_requires_storage_echo_but_legacy_client_does_not(
    monkeypatch, lease
):
    monkeypatch.setattr(lease, "_request", lambda *_args, **_kwargs: {})
    assert lease.start("/image.sif") == {}
    assert lease.restart() == {}
    with pytest.raises(KeyError, match="storage"):
        lease.start("/image.sif", storage_mb=2048, image_sha256="a" * 64)
    with pytest.raises(KeyError, match="storage"):
        lease.restart()


def test_disk_client_checks_restart_sha_and_mount_verification(monkeypatch, lease):
    storage = {
        "mode": "disk-overlay",
        "storage_mb": 2048,
        "policy": "nominal-image-mib",
        "writable_paths_verified": True,
        "image_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        lease, "_request", lambda *_args, **_kwargs: {"storage": storage}
    )
    assert (
        lease.start("/image.sif", storage_mb=2048, image_sha256="a" * 64)["storage"]
        == storage
    )
    assert lease.restart()["storage"] == storage
    storage["image_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="verification"):
        lease.restart()
    storage["image_sha256"] = "a" * 64
    storage["writable_paths_verified"] = False
    with pytest.raises(RuntimeError, match="verification"):
        lease.restart()


@pytest.mark.parametrize("assigned", [False, True])
def test_cancelled_acquire_cancels_pending_or_late_assigned_request(
    monkeypatch, assigned
):
    cancelled = threading.Event()
    calls = []

    def request(_base, path, **kwargs):
        calls.append((path, kwargs.get("method")))
        cancelled.set()
        return {
            "request_id": "owned-request",
            "status": "assigned" if assigned else "pending",
            "poll_after_seconds": 1,
        }

    monkeypatch.setattr("sandfleet.client.request_json", request)
    with pytest.raises(InterruptedError, match="cancelled"):
        SandfleetClient("http://controller", "client").acquire(
            cpus=1, memory_mb=1024, cancel_event=cancelled
        )
    assert calls == [
        ("/v1/lease-requests", "POST"),
        ("/v1/lease-requests/owned-request", "DELETE"),
    ]


def test_acquire_waits_on_controller_queue_without_slurm_polling(monkeypatch):
    responses = iter(
        [
            {"request_id": "request-1", "status": "pending", "poll_after_seconds": 0},
            {
                "request_id": "request-1",
                "status": "assigned",
                "lease": {
                    "lease_id": "lease-1",
                    "lease_token": "lease-token",
                    "agent_url": "http://agent:9000",
                    "pool": "rollouts",
                    "ttl_seconds": 0,
                },
            },
        ]
    )
    calls = []

    def fake_request(base_url, path, **kwargs):
        calls.append((base_url, path, kwargs))
        return next(responses)

    monkeypatch.setattr("sandfleet.client.request_json", fake_request)
    client = SandfleetClient("http://controller:8765", "secret")
    lease = client.acquire("rollouts", timeout=5)
    assert lease.lease_id == "lease-1"
    assert [path for _, path, _ in calls] == [
        "/v1/lease-requests",
        "/v1/lease-requests/request-1",
    ]


def test_acquire_sends_declarative_resources(monkeypatch):
    calls = []

    def fake_request(base_url, path, **kwargs):
        calls.append((base_url, path, kwargs))
        return {
            "request_id": "request-1",
            "status": "assigned",
            "poll_after_seconds": 2,
            "lease": {
                "lease_id": "lease-1",
                "lease_token": "lease-token",
                "agent_url": "http://agent:9000",
                "pool": "sfres-c4-m8192-gpu-none",
                "ttl_seconds": 0,
            },
        }

    monkeypatch.setattr("sandfleet.client.request_json", fake_request)
    client = SandfleetClient("http://controller:8765", "secret")
    lease = client.acquire(cpus=4, memory_mb=8192, timeout=5)

    assert lease.pool == "sfres-c4-m8192-gpu-none"
    assert calls[0][2]["payload"] == {
        "resources": {"cpus": 4, "memory_mb": 8192},
        "timeout_seconds": 5,
    }
    client.acquire(cpus=4, memory_mb=8192, timeout=5, internet=True)
    assert calls[1][2]["payload"]["internet"] is True


def test_acquire_sends_ordered_gpu_type_fallbacks(monkeypatch):
    calls = []

    def fake_request(base_url, path, **kwargs):
        calls.append((base_url, path, kwargs))
        return {
            "request_id": "request-1",
            "status": "failed",
            "poll_after_seconds": 2,
            "error": "GPU sandboxes are not supported",
        }

    monkeypatch.setattr("sandfleet.client.request_json", fake_request)
    client = SandfleetClient("http://controller:8765", "secret")
    with pytest.raises(TimeoutError, match="GPU sandboxes"):
        client.acquire(
            cpus=4,
            memory_mb=8192,
            gpu=1,
            gpu_type=["a100_80gb", "a100_40gb"],
            timeout=5,
        )
    assert calls[0][2]["payload"]["resources"] == {
        "cpus": 4,
        "memory_mb": 8192,
        "gpu": 1,
        "type": ["a100_80gb", "a100_40gb"],
    }


def test_gpu_capabilities_is_a_cache_only_client_get(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "sandfleet.client.request_json",
        lambda base_url, path, **kwargs: (
            calls.append((base_url, path, kwargs))
            or {"enabled": True, "types": {"h100_80gb": {}}}
        ),
    )
    result = SandfleetClient("http://controller:8765", "secret").gpu_capabilities()
    assert result["enabled"]
    assert calls[0][1] == "/v1/capabilities/gpus"
    assert calls[0][2]["token"] == "secret"


def test_acquire_requires_exactly_one_resource_mode():
    client = SandfleetClient("http://controller:8765", "secret")
    with pytest.raises(ValueError, match="either pool"):
        client.acquire()
    with pytest.raises(ValueError, match="either pool"):
        client.acquire("rollouts", cpus=1, memory_mb=512)
    with pytest.raises(ValueError, match="require cpus and memory_mb"):
        client.acquire(cpus=1)
    with pytest.raises(ValueError, match="both gpu and gpu_type"):
        client.acquire(cpus=1, memory_mb=512, gpu=1)


def test_run_command_with_id_sends_it_and_surfaces_cancelled(monkeypatch, lease):
    seen = {}

    def request(path, *, payload, timeout):
        seen["path"] = path
        seen["payload"] = payload
        return {"stdout": "", "stderr": "", "exit_code": 137, "cancelled": True}

    monkeypatch.setattr(lease, "_request", request)
    result = lease.run_command("sleep 30", command_id="job-1")
    assert seen["path"] == "/exec"
    assert seen["payload"]["command_id"] == "job-1"
    assert result.cancelled is True
    assert result.exit_code == 137


def test_run_command_without_id_omits_the_field(monkeypatch, lease):
    seen = {}

    def request(path, *, payload, timeout):
        seen["payload"] = payload
        return {"stdout": "", "stderr": "", "exit_code": 0}

    monkeypatch.setattr(lease, "_request", request)
    result = lease.run_command("echo hi")
    assert "command_id" not in seen["payload"]
    assert result.cancelled is False


def test_cancel_command_hits_exec_cancel_route(monkeypatch, lease):
    seen = {}

    def request(path, *, payload, timeout):
        seen["path"] = path
        seen["payload"] = payload
        return {"cancelled": True, "interrupted": True, "state": "running"}

    monkeypatch.setattr(lease, "_request", request)
    result = lease.cancel_command("job-1")
    assert seen["path"] == "/exec-cancel"
    assert seen["payload"] == {"command_id": "job-1"}
    assert result == {"cancelled": True, "interrupted": True, "state": "running"}


def test_kill_processes_hits_kill_processes_route(monkeypatch, lease):
    seen = {}

    def request(path, *, payload, timeout):
        seen.update(path=path, payload=payload, timeout=timeout)
        return {"killed": 2, "cancelled_commands": []}

    monkeypatch.setattr(lease, "_request", request)
    assert lease.kill_processes() == {"killed": 2, "cancelled_commands": []}
    assert seen["path"] == "/kill-processes" and seen["payload"] == {}
    assert seen["timeout"] > 90


@pytest.mark.parametrize("timed_out", [True, False, None])
def test_run_command_retains_optional_timeout_outcome(monkeypatch, lease, timed_out):
    result = {"stdout": "", "stderr": "", "exit_code": 0}
    if timed_out is not None:
        result["timed_out"] = timed_out
    monkeypatch.setattr(lease, "_request", lambda *args, **kwargs: result)
    assert lease.run_command("true", command_id="deadline").timed_out is timed_out
