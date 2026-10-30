from __future__ import annotations

import socket
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from sandfleet.egress import EgressSettings
from sandfleet.controller import ControllerHandler, ControllerState
from sandfleet.errors import SandfleetError
from sandfleet.http import request_json
from sandfleet.resources import DynamicPoolPolicy


@pytest.mark.parametrize("max_storage_mb", [0, 2048])
def test_dashboard_and_metrics_api_have_separate_auth(monkeypatch, max_storage_mb):
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: "localhost")
    state = ControllerState(
        token="admin-secret",
        client_token="client-secret",
        metrics_token="metrics-secret",
        launcher=SimpleNamespace(
            settings=SimpleNamespace(
                max_storage_mb=max_storage_mb,
                internet=EgressSettings(),
                readonly_mount_roots=("/models",),
                agent_readonly_mount_roots=("/assets",),
                time_limit="1-12:00:00",
            )
        ),
        max_capacity=16,
        max_cpus_per_sandbox=4,
        max_memory_mb_per_sandbox=16384,
        dynamic_pool_policy=DynamicPoolPolicy(
            max_cpus_per_worker=8,
            max_memory_mb_per_worker=16384,
            slots_per_job=2,
            max_capacity=4,
            max_pools=2,
        ),
    )
    state.create_pool({"name": "web", "desired_capacity": 1, "slots_per_job": 1})
    assert state.token_is_admin("admin-secret")
    assert not state.token_is_client("admin-secret")
    assert not state.token_is_metrics("admin-secret")
    assert state.token_is_client("client-secret")
    assert not state.token_is_metrics("client-secret")
    assert state.token_is_metrics("metrics-secret")
    server = ThreadingHTTPServer(("127.0.0.1", 0), ControllerHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = request_json(url, "/v1/health", token="client-secret", timeout=2)
        assert (
            health["max_storage_mb"],
            health["internet"],
            health["readonly_mount_roots"],
            health["agent_readonly_mount_roots"],
            health["worker_time_limit_seconds"],
        ) == (max_storage_mb, "off", ["/models"], ["/assets"], 36 * 3600)
        assert set(health["features"]) == {
            "command-env-v1",
            "command-cancel-v1",
            "disk-overlay-v1",
            "file-chunks-v1",
            "lease-abort-v1",
            "lease-internet-v1",
            "lease-kill-processes-v1",
            "lease-start-readonly-mounts-v1",
            "lease-readonly-mounts-v1",
        }
        with urlopen(url + "/", timeout=2) as response:  # noqa: S310 - local test server
            assert b"Slurm sandbox fleet control plane" in response.read()

        try:
            urlopen(url + "/v1/metrics", timeout=2)  # noqa: S310 - local test server
        except HTTPError as error:
            assert error.code == 401
        else:
            raise AssertionError("metrics endpoint accepted an unauthenticated request")

        metrics = request_json(url, "/v1/metrics", token="metrics-secret", timeout=2)
        assert metrics["totals"]["pools"] == 1
        request = Request(
            url + "/metrics", headers={"Authorization": "Bearer metrics-secret"}
        )
        with urlopen(request, timeout=2) as response:  # noqa: S310 - local test server
            assert b"sandfleet_up" in response.read()

        try:
            request_json(
                url, "/v1/pools", method="POST", token="client-secret", payload={}
            )
        except Exception as error:
            assert getattr(error, "status", None) == 401
        else:
            raise AssertionError("client token was accepted for an admin route")

        dynamic = request_json(
            url,
            "/v1/lease-requests",
            method="POST",
            token="client-secret",
            payload={
                "resources": {"cpus": 2, "memory_mb": 2048},
                "timeout_seconds": 30,
            },
        )
        assert dynamic["pool"] == "sfres-c2-m2048-gpu-none"
        assert dynamic["resources"] == {"cpus": 2, "memory_mb": 2048}
        assert_gpu_capabilities_disabled(url)
        del state.launcher.settings.max_storage_mb
        with pytest.raises(SandfleetError, match="max_storage_mb") as failure:
            request_json(url, "/v1/health", token="client-secret", timeout=2)
        assert failure.value.status == 500
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def assert_gpu_capabilities_disabled(url):
    gpu_capabilities = request_json(
        url,
        "/v1/capabilities/gpus",
        token="client-secret",
        timeout=2,
    )
    assert gpu_capabilities["enabled"] is False
    assert gpu_capabilities["types"] == {}
