"""Agent-phase read-only mounts given when a sandbox starts (0.14)."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sandfleet import mounts, step_worker
from sandfleet.agent import AgentHandler, AgentState, build_parser, validate_args
from sandfleet.audit import AuditLog
from sandfleet.http import request_json
from sandfleet.mounts import ReadonlyMount
from sandfleet.slurm import SlurmLauncher
from sandfleet.step_worker import ApptainerInstance, WorkerSession

from test_agent import FakeSandbox
from test_client import lease  # noqa: F401 - fixture
from test_slurm import settings

AGENT_MOUNT = ReadonlyMount("/assets/data", "/assets/data", "/assets/data", "agent")


# --- per-phase roots ----------------------------------------------------------


def test_agent_roots_may_not_contain_verifier_roots(tmp_path):
    # An agent root inside a verifier root is fine: verifiers may read agent inputs.
    assert mounts.validate_phase_roots(["/assets/public"], ["/assets"]) == (
        ("/assets/public",),
        ("/assets",),
    )
    for agent_roots in (["/assets"], ["/assets/hidden"]):
        with pytest.raises(ValueError, match="contains the verifier-only root"):
            mounts.validate_phase_roots(agent_roots, ["/assets/hidden"])
    with pytest.raises(ValueError, match="verifier-only"):
        replace(
            settings(tmp_path),
            readonly_mount_roots=("/assets/hidden",),
            agent_readonly_mount_roots=("/assets",),
        )
    parser = build_parser()
    base = ["--controller-url", "http://c:1", "--pool", "p", "--slots", "1"]
    base += ["--cpus", "1", "--memory-mb", "64", "--readonly-mount-root", "/a/hidden"]
    with pytest.raises(SystemExit):
        validate_args(
            parser, parser.parse_args(base + ["--agent-readonly-mount-root", "/a"])
        )


def test_worker_submission_passes_agent_roots(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_: seen.append(command)
        or SimpleNamespace(returncode=0, stdout="1\n", stderr=""),
    )
    configured = replace(
        settings(tmp_path),
        readonly_mount_roots=("/hidden",),
        agent_readonly_mount_roots=("/assets",),
    )
    SlurmLauncher(
        configured, service_token_file="/t", controller_url="http://c:1"
    ).submit_worker_batch(pool="p", worker_count=1, slots=1, cpus=1, memory_mb=64)
    assert "--readonly-mount-root /hidden" in seen[0][-1]
    assert "--agent-readonly-mount-root /assets" in seen[0][-1]


# --- worker session: resolution against the agent roots ----------------------


@pytest.fixture
def phase_tree(tmp_path):
    agent_root = tmp_path / "assets"
    (agent_root / "data").mkdir(parents=True)
    verifier_root = tmp_path / "hidden"
    (verifier_root / "model").mkdir(parents=True)
    # A link inside the agent root to verifier-only files.
    (agent_root / "leak").symlink_to(verifier_root / "model")
    return SimpleNamespace(agent=str(agent_root), verifier=str(verifier_root))


def start_session(monkeypatch, agent_roots, verifier_roots):
    built = []
    monkeypatch.setattr(
        step_worker,
        "ApptainerInstance",
        lambda **config: built.append(config) or Mock(name="instance"),
    )
    session = WorkerSession(
        readonly_mount_roots=verifier_roots, agent_readonly_mount_roots=agent_roots
    )
    monkeypatch.setattr(session, "_status", lambda _request: {})

    def start(source, target="/assets/x"):
        config = {"image": "/i.sif", "gpu_enabled": False}
        config["readonly_mounts"] = [{"source": source, "target": target}]
        return session.dispatch({"operation": "start", "config": config})

    return session, start, built


def test_start_mounts_resolve_against_the_agent_roots_only(monkeypatch, phase_tree):
    _session, start, built = start_session(
        monkeypatch, (phase_tree.agent,), (phase_tree.verifier,)
    )
    for source in (f"{phase_tree.verifier}/model", f"{phase_tree.agent}/leak"):
        with pytest.raises(PermissionError, match="for the agent phase"):
            start(source)
    assert built == []
    start(f"{phase_tree.agent}/data")
    [[mount]] = [config["readonly_mounts"] for config in built]
    assert mount.phase == "agent"
    assert mount.host_path == os.path.realpath(f"{phase_tree.agent}/data")


def test_start_mounts_need_agent_roots(monkeypatch, phase_tree):
    _session, start, built = start_session(monkeypatch, (), (phase_tree.verifier,))
    with pytest.raises(ValueError, match="agent-phase mounts are not enabled"):
        start(f"{phase_tree.verifier}/model")
    assert built == []


def test_roots_that_resolve_into_each_other_are_refused(
    monkeypatch, tmp_path, phase_tree
):
    alias = tmp_path / "alias"
    alias.symlink_to(phase_tree.verifier)  # configured apart, the same directory
    _session, start, built = start_session(
        monkeypatch, (str(alias),), (phase_tree.verifier,)
    )
    with pytest.raises(ValueError, match="verifier-only root"):
        start(f"{alias}/model")
    assert built == []


# --- ApptainerInstance with start-time mounts --------------------------------


def instance(monkeypatch, tmp_path, *, disk, seccomp="1", pwd="/workspace", **extra):
    monkeypatch.setattr("sandfleet.step_worker.image_digest", lambda *_a: "a" * 64)
    monkeypatch.setenv("SANDFLEET_SANDBOX_SECCOMP", seccomp)
    return ApptainerInstance(
        image="/image.sif",
        timeout=60,
        pwd=pwd,
        cache_dir=None,
        tmp_dir=None,
        extra_start_flags=[],
        apptainer_binary="apptainer",
        **({"storage_mb": 64, "image_sha256": "a" * 64} if disk else {}),
        storage_root=str(tmp_path) if disk else None,
        **extra,
    )


def test_writable_tmpfs_start_binds_each_mount_read_only(monkeypatch, tmp_path):
    value = instance(monkeypatch, tmp_path, disk=False, readonly_mounts=[AGENT_MOUNT])
    flags = value.start_flags
    index = flags.index("/assets/data:/assets/data:ro")
    assert flags[index - 1] == "--bind"
    assert value.readonly_mounts_status == [AGENT_MOUNT.describe()]


def test_disk_start_creates_the_mount_points_in_its_fresh_overlay(
    monkeypatch, tmp_path
):
    value = instance(monkeypatch, tmp_path, disk=True, readonly_mounts=[AGENT_MOUNT])
    value.overlay = Mock()
    value.overlay.create.return_value = tmp_path / "writable.img"
    flags = value._disk_start_flags()
    value.overlay.create.assert_called_once_with(create_dirs=["/assets/data"])
    assert "/assets/data:/assets/data:ro" in flags


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"seccomp": "0"}, RuntimeError, "seccomp profile that denies"),
        ({"pwd": "/assets/data/work"}, ValueError, "would cover the writable path"),
    ],
)
def test_start_mounts_are_refused_before_any_start(
    monkeypatch, tmp_path, options, error, message
):
    with pytest.raises(error, match=message):
        instance(
            monkeypatch, tmp_path, disk=True, readonly_mounts=[AGENT_MOUNT], **options
        )


def test_start_mounts_are_verified_from_the_host_after_launch(monkeypatch, tmp_path):
    value = instance(monkeypatch, tmp_path, disk=False, readonly_mounts=[AGENT_MOUNT])
    calls = []
    monkeypatch.setattr(
        step_worker.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0)
    )
    monkeypatch.setattr(value, "_start_network", lambda: calls.append("network"))
    monkeypatch.setattr(value, "_bootstrap", lambda: calls.append("bootstrap"))
    monkeypatch.setattr(
        value, "_verify_readonly_mounts", lambda: calls.append("verify mounts")
    )
    value._launch(reuse_overlay=False)
    assert calls == ["network", "bootstrap", "verify mounts"]


def test_added_mounts_may_not_overlap_start_mounts(monkeypatch, tmp_path):
    value = instance(monkeypatch, tmp_path, disk=True, readonly_mounts=[AGENT_MOUNT])
    value.name = "inst"
    value.overlay = Mock()
    nested = ReadonlyMount("/hidden/m", "/assets/data/m", "/hidden/m")
    with pytest.raises(ValueError, match="equal or nested"):
        value.add_readonly_mounts([nested])


# --- agent and client -----------------------------------------------------------


@pytest.fixture
def agent_lease(monkeypatch, tmp_path):
    log = AuditLog(tmp_path / "w.jsonl", max_bytes=1 << 20, context={"pool": "p"})
    state = AgentState(
        service_token="service-secret",
        slots=1,
        cpus=1,
        memory_mb=4096,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
        audit=log,
        agent_readonly_mount_roots=("/assets",),
    )
    sandbox = FakeSandbox()
    sandbox.start = Mock(return_value={"instance_name": "fake-1"})
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    granted = request_json(
        url, "/v1/control/leases", method="POST", token="service-secret", payload={}
    )

    def start(payload):
        return request_json(
            url,
            f"/v1/leases/{granted['lease_id']}/start",
            method="POST",
            token=granted["lease_token"],
            payload={"image": "/i.sif", **payload},
            timeout=5,
        )

    try:
        yield SimpleNamespace(start=start, sandbox=sandbox, log=tmp_path / "w.jsonl")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_agent_forwards_and_audits_start_mounts(agent_lease):
    request = [{"source": "/assets/data", "target": "/assets/data"}]
    agent_lease.start({"readonly_mounts": request})
    assert agent_lease.sandbox.start.call_args.kwargs["readonly_mounts"] == request
    with pytest.raises(Exception, match="reserved"):
        agent_lease.start(
            {"readonly_mounts": [{"source": "/assets/d", "target": "/etc/d"}]}
        )
    agent_lease.start({})  # an old client sends no key
    assert "readonly_mounts" not in agent_lease.sandbox.start.call_args.kwargs
    records = [json.loads(line) for line in agent_lease.log.read_text().splitlines()]
    assert [record["op"] for record in records] == ["start"] * 3
    assert records[0]["readonly_mounts"] == request and "error" not in records[0]
    assert (
        "/etc/d" in records[1]["readonly_mounts"] and "reserved" in records[1]["error"]
    )
    assert "readonly_mounts" not in records[2]


def test_client_sends_start_mounts_and_requires_the_worker_to_report_them(
    monkeypatch,
    lease,  # noqa: F811
):
    request = [{"source": "/assets/data", "target": "/assets/data"}]
    reported = [{**request[0], "read_only": True, "phase": "agent"}]
    payloads = []

    def respond(path, *, payload, timeout):
        payloads.append((path, payload))
        return {"readonly_mounts": reported}

    monkeypatch.setattr(lease, "_request", respond)
    lease.start("/i.sif", readonly_mounts=request)
    lease.restart()
    assert [(path, payload["readonly_mounts"]) for path, payload in payloads] == [
        ("/start", request),
        ("/restart", request),
    ]
    # A pre-0.14 agent ignores the field and reports no mounts.
    reported = []
    with pytest.raises(RuntimeError, match="did not mount"):
        lease.start("/i.sif", readonly_mounts=request)
    lease.start("/i.sif")
    assert "readonly_mounts" not in payloads[-1][1]
