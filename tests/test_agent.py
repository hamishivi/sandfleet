from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from unittest.mock import Mock

import pytest

from sandfleet import agent
from sandfleet.audit import AuditLog
from sandfleet.errors import NoCapacityError, SandfleetError
from sandfleet.agent import (
    AgentHandler,
    AgentState,
    LeaseRecord,
    advertise_host,
    current_cgroup_path,
)
from sandfleet.http import encode_bytes, request_json


class FakeSandbox:
    def __init__(
        self,
        cpuset=frozenset({0}),
        cgroup_path="/job/step/task",
        gpu_devices=frozenset(),
    ):
        self.instance = 0
        self.files: dict[str, bytes] = {}
        self.closed = False
        self.prepared = False
        self.cpuset = cpuset
        self.cgroup_path = cgroup_path
        self.gpu_devices = gpu_devices
        self.storage_lost = False

    @property
    def healthy(self):
        return self.prepared and not self.closed and not self.storage_lost

    def prepare(self):
        self.prepared = True
        return self.status()

    def recycle(self):
        self.instance = 0
        return self.status()

    def start(self, **_kwargs):
        self.instance += 1
        return self.status()

    def restart(self, **_kwargs):
        self.instance += 1
        return self.status()

    def run_command(self, command, timeout=None):
        return {"stdout": command, "stderr": "", "exit_code": 0}

    def write_file(self, path, content):
        self.files[path] = content

    def read_file(self, path):
        return self.files[path]

    def put_archive(self, root, tar_bytes):
        self.files[root] = tar_bytes

    def status(self):
        return {
            "instance_name": f"fake-{self.instance}",
            "cgroup": {"memory.max.effective": "4096"},
        }

    def close(self):
        self.closed = True

    def discard(self):
        self.close()


@pytest.mark.parametrize("outcome", ["drained", "abort_error", "still_running"])
def test_busy_release_interrupts_and_requires_confirmed_drain(monkeypatch, outcome):
    state = AgentState(
        service_token="service",
        slots=1,
        cpus=1,
        memory_mb=1024,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
    )
    sandbox = FakeSandbox()
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    monkeypatch.setattr(state, "_retry_at", lambda failures: 0)
    monkeypatch.setattr(agent, "RELEASE_IDLE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(agent, "RELEASE_DRAIN_SECONDS", 0.05)
    assert state.prepare_one(lambda: None)
    lease_id = state.reserve()["lease_id"]
    lease = state.leases[lease_id]
    busy, drain = threading.Event(), threading.Event()

    def execute():
        with lease.lock:
            busy.set()
            drain.wait(timeout=5)

    def abort():
        if outcome == "abort_error":
            raise RuntimeError("abort failed")
        if outcome == "drained":
            sandbox.closed = True
            drain.set()

    sandbox.abort = Mock(side_effect=abort)
    thread = threading.Thread(target=execute)
    thread.start()
    assert busy.wait(timeout=1)
    try:
        if outcome == "drained":
            assert state.release(lease_id)["released"]
            assert lease_id not in state.leases
            assert state.free_slots == 0  # killed step is not marked reusable
            assert state.prepare_wakeup.is_set()
            replacement = FakeSandbox()
            replacement.abort = Mock()
            monkeypatch.setattr(state, "new_sandbox", lambda: replacement)
            assert state.prepare_one(lambda: None)
            replacement_id = state.reserve()["lease_id"]
            assert not state.release(lease_id)["released"]
            assert state.release(replacement_id)["released"]
            assert state.prepared_slots[0].sandbox is replacement
            replacement.abort.assert_not_called()
        else:
            with pytest.raises((RuntimeError, TimeoutError)):
                state.release(lease_id)
            assert state.leases[lease_id] is lease
            assert state.prepared_slots[lease.slot_id].lease_id == lease_id
            assert state.free_slots == 0
        sandbox.abort.assert_called_once()
    finally:
        drain.set()
        thread.join(timeout=1)


def test_release_interrupt_lock_contention_is_retryable(monkeypatch):
    state = AgentState(
        service_token="service",
        slots=1,
        cpus=1,
        memory_mb=1024,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
    )
    sandbox = FakeSandbox()
    sandbox.abort = Mock()
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    monkeypatch.setattr(agent, "RELEASE_IDLE_GRACE_SECONDS", 0.01)
    assert state.prepare_one(lambda: None)
    lease_id = state.reserve()["lease_id"]
    lease = state.leases[lease_id]
    errors = []

    def release():
        try:
            state.release(lease_id)
        except agent.LeaseCleanupPending as error:
            errors.append(error)

    with lease.interrupt_lock:
        thread = threading.Thread(target=release)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert len(errors) == 1 and errors[0].status == 503
    assert state.leases[lease_id] is lease
    assert state.free_slots == 0
    assert state.release(lease_id)["released"]
    assert state.free_slots == 1
    sandbox.abort.assert_not_called()


def test_disk_release_retains_lease_on_failed_stop_then_retries(monkeypatch):
    state = AgentState(
        service_token="service",
        slots=1,
        cpus=1,
        memory_mb=1024,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        max_storage_mb=1024,
        start_concurrency=1,
    )
    sandbox = FakeSandbox()
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    lease = state.reserve()
    original = sandbox.recycle

    def fail():
        raise RuntimeError("still mounted")

    monkeypatch.setattr(sandbox, "recycle", fail)
    with pytest.raises(RuntimeError, match="still mounted"):
        state.release(lease["lease_id"])
    assert lease["lease_id"] in state.leases
    assert state.free_slots == 0 and not sandbox.closed
    monkeypatch.setattr(sandbox, "recycle", original)
    assert state.release(lease["lease_id"])["released"]
    assert state.free_slots == 1
    assert not state.release(lease["lease_id"])["released"]

    # A stale request must never interrupt a new occupant of the reused slot.
    replacement = state.reserve()
    sandbox.abort = Mock()
    old_lease = LeaseRecord(lease_id=lease["lease_id"], token="unused", slot_id=0)
    with pytest.raises(RuntimeError, match="lost"):
        state.abort(old_lease)
    sandbox.abort.assert_not_called()
    current = state.leases[replacement["lease_id"]]
    with current.lock:
        # A different thread must not block on the active execution lock.
        interrupted = threading.Thread(target=state.abort, args=(current,))
        interrupted.start()
        interrupted.join(timeout=1)
        assert not interrupted.is_alive()
    sandbox.abort.assert_called_once()
    state.release(replacement["lease_id"])


def test_agent_lease_lifecycle(monkeypatch):
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
    )
    created: list[FakeSandbox] = []

    def new_sandbox():
        sandbox = FakeSandbox()
        created.append(sandbox)
        return sandbox

    monkeypatch.setattr(state, "new_sandbox", new_sandbox)
    assert state.prepare_one(lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        reservation = request_json(
            url,
            "/v1/control/leases",
            method="POST",
            token="service-secret",
            payload={"internet": False},
        )
        lease_id = reservation["lease_id"]
        lease_token = reservation["lease_token"]
        assert reservation["free_slots"] == 0

        started = request_json(
            url,
            f"/v1/leases/{lease_id}/start",
            method="POST",
            token=lease_token,
            payload={"image": "/shared/image.sif"},
        )
        assert started["instance_name"] == "fake-1"

        result = request_json(
            url,
            f"/v1/leases/{lease_id}/exec",
            method="POST",
            token=lease_token,
            payload={"command": "echo hello"},
        )
        assert result == {"stdout": "echo hello", "stderr": "", "exit_code": 0}

        restarted = request_json(
            url,
            f"/v1/leases/{lease_id}/restart",
            method="POST",
            token=lease_token,
            payload={"image": "/shared/image.sif"},
        )
        assert restarted["instance_name"] == "fake-2"

        released = request_json(
            url,
            f"/v1/control/leases/{lease_id}",
            method="DELETE",
            token="service-secret",
        )
        assert released == {"released": True, "free_slots": 1}
        assert not created[0].closed
        assert created[0].instance == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_prepare_failure_is_reported(monkeypatch):
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
    )
    sandbox = FakeSandbox()

    def fail_prepare():
        raise RuntimeError("bad cgroup")

    monkeypatch.setattr(sandbox, "prepare", fail_prepare)
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)

    assert not state.prepare_one(lambda: None)
    assert state.capacity()["slot_errors"] == {"0": "bad cgroup"}


def test_prepare_accepts_hyperthreads_within_the_worker_allocation(monkeypatch):
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
        allocation_cpus=frozenset({7, 103}),
        allocation_cgroup="/job/step_batch/task",
    )
    sandbox = FakeSandbox(frozenset({7, 103}), "/job/step_0/task")
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)

    assert state.prepare_one(lambda: None)
    assert state.capacity()["ready_slots"] == 1


@pytest.mark.parametrize(
    ("cpuset", "cgroup_path", "message"),
    [
        (frozenset({0, 4}), "/job/step_0/task", "escapes the worker allocation"),
        (frozenset({0}), "/job/step_batch/task", "distinct cgroup"),
    ],
)
def test_prepare_rejects_a_step_outside_its_cpu_boundary(
    monkeypatch, cpuset, cgroup_path, message
):
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
        allocation_cpus=frozenset({0, 1}),
        allocation_cgroup="/job/step_batch/task",
    )
    sandbox = FakeSandbox(cpuset, cgroup_path)
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)

    assert not state.prepare_one(lambda: None)
    assert message in state.capacity()["slot_errors"]["0"]
    assert sandbox.closed


def test_prepare_rejects_overlapping_step_cpus(monkeypatch):
    state = AgentState(
        service_token="service-secret",
        slots=2,
        cpus=1,
        memory_mb=4096,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
        allocation_cpus=frozenset({0, 1, 2}),
        allocation_cgroup="/job/step_batch/task",
    )
    sandboxes = iter(
        [
            FakeSandbox(frozenset({0, 1}), "/job/step_0/task"),
            FakeSandbox(frozenset({1, 2}), "/job/step_1/task"),
        ]
    )
    monkeypatch.setattr(state, "new_sandbox", lambda: next(sandboxes))

    assert state.prepare_one(lambda: None)
    assert not state.prepare_one(lambda: None)
    assert "overlaps slot 0" in state.capacity()["slot_errors"]["1"]


def test_prepare_rejects_overlapping_step_gpu_devices(monkeypatch):
    state = AgentState(
        service_token="service-secret",
        slots=2,
        cpus=1,
        memory_mb=4096,
        gpus=1,
        gpu_type="h100-80gb",
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
        allocation_cpus=frozenset({0, 1}),
        allocation_cgroup="/job/step_batch/task",
    )
    sandboxes = iter(
        [
            FakeSandbox(
                frozenset({0}), "/job/step_0/task", frozenset({"/dev/nvidia0"})
            ),
            FakeSandbox(
                frozenset({1}), "/job/step_1/task", frozenset({"/dev/nvidia0"})
            ),
        ]
    )
    monkeypatch.setattr(state, "new_sandbox", lambda: next(sandboxes))

    assert state.prepare_one(lambda: None)
    assert not state.prepare_one(lambda: None)
    assert "GPU devices overlap slot 0" in state.capacity()["slot_errors"]["1"]


def test_agent_advertise_modes(monkeypatch):
    monkeypatch.setattr("sandfleet.agent.socket.getfqdn", lambda: "node.example")
    monkeypatch.setattr("sandfleet.agent.socket.gethostname", lambda: "node")
    assert (
        advertise_host("explicit", "route-ip", "http://controller:8765") == "explicit"
    )
    assert advertise_host(None, "fqdn", "http://controller:8765") == "node.example"
    assert advertise_host(None, "hostname", "http://controller:8765") == "node"

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def connect(self, _target):
            pass

        def getsockname(self):
            return ("10.0.0.8", 40000)

    connection = Connection()
    monkeypatch.setattr("sandfleet.agent.socket.socket", lambda *_args: connection)
    assert advertise_host(None, "route-ip", "http://controller:8765") == "10.0.0.8"


def test_current_cgroup_path_is_absolute(monkeypatch):
    monkeypatch.setattr(
        "pathlib.Path.read_text", lambda *_args, **_kwargs: "0::/slurm/job/step_batch\n"
    )
    assert current_cgroup_path() == "/sys/fs/cgroup/slurm/job/step_batch"


def test_fatal_prewarmer_error_stops_the_agent(monkeypatch):
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
    )
    reported = []

    def fail_prepare(_acquire_permit):
        raise ValueError("invalid cgroup metadata")

    monkeypatch.setattr(state, "prepare_one", fail_prepare)
    with pytest.raises(ValueError, match="invalid cgroup metadata"):
        state._prepare_loop(None, reported.append)
    assert state.stop_event.is_set()
    assert str(reported[0]) == "invalid cgroup metadata"


def test_dead_step_is_discarded_outside_agent_lock():
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
    )
    state.lock = threading.Lock()

    class DeadSandbox(FakeSandbox):
        def discard(self):
            assert state.lock.acquire(blocking=False)
            state.lock.release()
            self.closed = True

    sandbox = DeadSandbox()
    sandbox.prepared = True
    sandbox.closed = True
    slot = state.prepared_slots[0]
    slot.status = "ready"
    slot.sandbox = sandbox

    state._discard_broken_steps()
    assert slot.status == "failed"
    assert slot.sandbox is None
    assert slot.last_error == "Slurm step exited unexpectedly"


def test_lost_step_storage_stops_the_agent_accepting_leases(monkeypatch):
    # Grace-time preemption tears down the job's /tmp under live steps without
    # signalling the agent; every disk start then fails with a storage error.
    state = AgentState(
        service_token="service-secret",
        slots=2,
        cpus=1,
        memory_mb=4096,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        max_storage_mb=1024,
        start_concurrency=1,
    )
    sandboxes = [FakeSandbox(), FakeSandbox()]
    monkeypatch.setattr(state, "new_sandbox", iter(sandboxes).__next__)
    assert state.prepare_one(lambda: None) and state.prepare_one(lambda: None)
    lease = state.leases[state.reserve()["lease_id"]]
    for sandbox in sandboxes:
        sandbox.storage_lost = True

    state.mark_broken(lease, RuntimeError("Storage command failed (1): "))
    assert state.release(lease.lease_id)["released"]
    failed = state.prepared_slots[lease.slot_id]
    assert failed.status == "failed" and failed.sandbox is None
    assert sandboxes[lease.slot_id].closed  # discarded, never recycled

    reported = []
    with pytest.raises(RuntimeError, match="storage vanished"):
        state._prepare_loop(lambda: None, reported.append)
    assert state.stop_event.is_set() and len(reported) == 1
    with pytest.raises(NoCapacityError):
        state.reserve()


def test_recycle_failure_uses_the_shared_slot_failure_transition(monkeypatch):
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
    )
    sandbox = FakeSandbox()
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    lease_id = state.reserve()["lease_id"]

    def fail_recycle():
        raise RuntimeError("recycle failed")

    monkeypatch.setattr(sandbox, "recycle", fail_recycle)

    assert state.release(lease_id) == {"released": True, "free_slots": 0}
    slot = state.prepared_slots[0]
    assert slot.status == "failed"
    assert slot.sandbox is None
    assert slot.failures == 1
    assert slot.last_error == "recycle failed"
    assert state.metrics["step_recycle_failures"] == 1


def test_cancel_command_targets_current_lease_only_without_blocking(monkeypatch):
    state = AgentState(
        service_token="service",
        slots=1,
        cpus=1,
        memory_mb=1024,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
    )
    sandbox = FakeSandbox()
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    lease = state.reserve()

    # A stale lease (same id, different record) must not cancel on whatever now
    # occupies the slot.
    sandbox.cancel_command = Mock(return_value={"cancelled": True, "state": "pending"})
    stale = LeaseRecord(lease_id="does-not-own", token="unused", slot_id=0)
    with pytest.raises(RuntimeError, match="lost"):
        state.cancel_command(stale, "job-1")
    sandbox.cancel_command.assert_not_called()

    current = state.leases[lease["lease_id"]]
    with current.lock:
        # Cancellation must not wait on the exec lock the current command holds.
        outcome: dict = {}

        def cancel():
            outcome["result"] = state.cancel_command(current, "job-1")

        thread = threading.Thread(target=cancel)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
    sandbox.cancel_command.assert_called_once_with("job-1")
    assert outcome["result"] == {"cancelled": True, "state": "pending"}
    state.release(lease["lease_id"])


class FakeEgress:
    def __init__(self):
        self.revoked: list[str] = []

    def grant(self, lease_id):
        return f"http://{lease_id}:proxy-secret@10.0.0.5:3128"

    def revoke(self, lease_id):
        self.revoked.append(lease_id)

    def metrics_snapshot(self):
        return {"egress_connections_allowed": 0}


def internet_agent(monkeypatch, egress):
    state = AgentState(
        service_token="service-secret",
        slots=2,
        cpus=1,
        memory_mb=4096,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
        egress=egress,
    )
    sandboxes = [
        Mock(cpuset=frozenset({cpu}), gpu_devices=frozenset()) for cpu in (0, 1)
    ]
    monkeypatch.setattr(state, "new_sandbox", sandboxes.pop)
    assert state.prepare_one(lambda: None)
    assert state.prepare_one(lambda: None)
    return state


def test_internet_requires_an_enabled_worker(monkeypatch):
    state = internet_agent(monkeypatch, None)
    with pytest.raises(ValueError, match="not enabled"):
        state.reserve(internet=True)
    with pytest.raises(ValueError, match="boolean"):
        state.reserve(internet="yes")
    assert state.free_slots == 2


def test_only_opted_in_leases_receive_proxy_credentials(monkeypatch):
    egress = FakeEgress()
    state = internet_agent(monkeypatch, egress)
    online = state.reserve(internet=True)
    offline = state.reserve(internet=False)

    for reservation, expected in (
        (online, f"http://{online['lease_id']}:proxy-secret@10.0.0.5:3128"),
        (offline, None),
    ):
        lease = state.leases[reservation["lease_id"]]
        assert lease.proxy_url == expected
        assert expected is None or expected not in reservation.values()
        sandbox = state.sandbox_for(lease)
        AgentHandler._operate(
            sandbox, "start", {"image": "/image.sif"}, lease.proxy_url
        )
        assert sandbox.start.call_args.kwargs["proxy_url"] == expected

    state.release(online["lease_id"])
    state.release(offline["lease_id"])
    assert egress.revoked == [online["lease_id"]]
    assert "egress_connections_allowed" in state.capacity()["metrics"]


def test_removed_fakeroot_network_mode_is_rejected():
    with pytest.raises(ValueError, match="fakeroot"):
        AgentState(
            service_token="service-secret",
            slots=1,
            cpus=1,
            memory_mb=4096,
            apptainer_binary="apptainer",
            worker_python="python",
            cache_dir=None,
            tmp_dir=None,
            start_concurrency=1,
            network_mode="fakeroot",
        )
    with pytest.raises(SystemExit):
        agent.build_parser().parse_args(
            [
                *("--controller-url", "http://c:1", "--pool", "p", "--slots", "1"),
                *("--cpus", "1", "--memory-mb", "64", "--network-mode", "fakeroot"),
            ]
        )


@pytest.mark.parametrize("value", [None, "0"])
def test_internet_agent_requires_dropped_network_caps(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SANDFLEET_SANDBOX_DROP_NET_CAPS", raising=False)
    else:
        monkeypatch.setenv("SANDFLEET_SANDBOX_DROP_NET_CAPS", value)
    parser = agent.build_parser()
    args = parser.parse_args(
        [
            *("--controller-url", "http://c:1", "--pool", "p", "--slots", "1"),
            *("--cpus", "1", "--memory-mb", "64", "--internet", "allowlist"),
        ]
    )
    if value is None:
        agent.validate_args(parser, args)
    else:
        with pytest.raises(SystemExit):
            agent.validate_args(parser, args)


def test_reservation_without_internet_field_is_offline(monkeypatch):
    state = internet_agent(monkeypatch, FakeEgress())
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reservation = request_json(
            f"http://127.0.0.1:{server.server_address[1]}",
            "/v1/control/leases",
            method="POST",
            token="service-secret",
            payload={},
        )
        assert state.leases[reservation["lease_id"]].proxy_url is None
    finally:
        server.shutdown()
        server.server_close()


class FailingSandbox(FakeSandbox):
    def run_command(self, command, timeout=None):
        raise SandfleetError("step lost")


@pytest.mark.parametrize("sandbox_type", [FakeSandbox, FailingSandbox])
def test_agent_audits_lease_operations(monkeypatch, tmp_path, sandbox_type):
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
    )
    monkeypatch.setattr(state, "new_sandbox", sandbox_type)
    assert state.prepare_one(lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        lease = request_json(
            url, "/v1/control/leases", method="POST", token="service-secret", payload={}
        )

        def call(operation, payload):
            return request_json(
                url,
                f"/v1/leases/{lease['lease_id']}/{operation}",
                method="POST",
                token=lease["lease_token"],
                payload=payload,
            )

        call("start", {"image": "/shared/image.sif"})
        call("write-file", {"path": "/tmp/f", "content_b64": encode_bytes(b"abc")})
        if sandbox_type is FailingSandbox:
            with pytest.raises(SandfleetError):
                call("exec", {"command": "curl api.ngrok.io"})
        else:
            call("exec", {"command": "curl api.ngrok.io"})
        call("read-file", {"path": "/tmp/f"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    records = [
        json.loads(line) for line in (tmp_path / "w.jsonl").read_text().splitlines()
    ]
    assert [record["op"] for record in records] == ["start", "write-file", "exec"]
    command = records[-1]
    assert command["lease_id"] == lease["lease_id"]
    assert command["slot"] == 0 and command["internet"] is False
    assert command["command"] == "curl api.ngrok.io"
    assert command["duration_s"] >= 0
    if sandbox_type is FailingSandbox:
        assert "step lost" in command["error"] and "exit_code" not in command
    else:
        assert command["exit_code"] == 0 and "error" not in command
    assert state.capacity()["metrics"]["audit_write_failures"] == 0


def test_egress_forwarder_binds_only_the_host_loopback(monkeypatch):
    started = []

    class Forwarder:
        port = 4567

        def __init__(self, settings, *, host, log):
            self.host = host

        def start(self):
            started.append(self.host)

    monkeypatch.setattr(agent, "EgressForwarder", Forwarder)
    monkeypatch.setattr(
        agent, "advertise_host", Mock(side_effect=AssertionError("not a cluster IP"))
    )
    args = agent.build_parser().parse_args(
        [
            *("--controller-url", "http://c:1", "--pool", "p", "--slots", "1"),
            *("--cpus", "1", "--memory-mb", "64", "--internet", "allowlist"),
        ]
    )
    assert agent.build_egress(args).host == "127.0.0.1"
    assert started == ["127.0.0.1"]


@pytest.mark.parametrize("fails", [False, True])
def test_kill_processes_skips_the_exec_lock_and_is_audited(
    monkeypatch, tmp_path, fails
):
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
    )
    sandbox = FakeSandbox()
    if fails:
        sandbox.kill_processes = Mock(side_effect=SandfleetError("SURVIVORS 7(sh)"))
    else:
        sandbox.kill_processes = Mock(
            return_value={"killed": 2, "cancelled_commands": ["job-1"]}
        )
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        lease = request_json(
            url, "/v1/control/leases", method="POST", token="service-secret", payload={}
        )

        def kill():
            return request_json(
                url,
                f"/v1/leases/{lease['lease_id']}/kill-processes",
                method="POST",
                token=lease["lease_token"],
                payload={},
                timeout=5,
            )

        # An active exec holds the lease lock; the kill must not wait for it.
        with state.leases[lease["lease_id"]].lock:
            if fails:
                with pytest.raises(SandfleetError, match="SURVIVORS"):
                    kill()
            else:
                assert kill() == {"killed": 2, "cancelled_commands": ["job-1"]}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    sandbox.kill_processes.assert_called_once_with()
    [record] = [
        json.loads(line) for line in (tmp_path / "w.jsonl").read_text().splitlines()
    ]
    assert record["op"] == "kill-processes"
    assert record["lease_id"] == lease["lease_id"]
    if fails:
        assert "SURVIVORS" in record["error"] and record["killed"] is None
    else:
        assert record["killed"] == 2 and record["cancelled_commands"] == ["job-1"]
        assert "error" not in record


def test_kill_processes_refuses_a_stale_lease(monkeypatch):
    state = AgentState(
        service_token="service",
        slots=1,
        cpus=1,
        memory_mb=1024,
        apptainer_binary="apptainer",
        worker_python="python",
        cache_dir=None,
        tmp_dir=None,
        start_concurrency=1,
    )
    sandbox = FakeSandbox()
    sandbox.kill_processes = Mock()
    monkeypatch.setattr(state, "new_sandbox", lambda: sandbox)
    assert state.prepare_one(lambda: None)
    state.reserve()
    stale = LeaseRecord(lease_id="does-not-own", token="unused", slot_id=0)
    with pytest.raises(RuntimeError, match="lost"):
        state.kill_processes(stale)
    sandbox.kill_processes.assert_not_called()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1800000000", 1800000000.0),
        (" 1800000000\n", 1800000000.0),
        ("0", None),
        ("", None),
        ("Unknown", None),
    ],
)
def test_job_end_time_reads_only_slurms_environment(raw, expected):
    assert agent.job_end_time({"SLURM_JOB_END_TIME": raw}) == expected
    assert agent.job_end_time({}) is None


@pytest.mark.parametrize("end_time", [None, 1800000000.0])
def test_start_restart_and_status_report_the_worker_end_time(monkeypatch, end_time):
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
        job_end_time=end_time,
    )
    monkeypatch.setattr(state, "new_sandbox", FakeSandbox)
    assert state.prepare_one(lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    server.state = state
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        lease = request_json(
            url, "/v1/control/leases", method="POST", token="service-secret", payload={}
        )
        path = f"/v1/leases/{lease['lease_id']}"
        token = lease["lease_token"]
        image = {"image": "/shared/image.sif"}
        replies = [
            request_json(
                url, f"{path}/start", method="POST", token=token, payload=image
            ),
            request_json(
                url, f"{path}/restart", method="POST", token=token, payload=image
            ),
            request_json(url, f"{path}/status", token=token),
        ]
        for reply in replies:
            # Optional: a worker that does not know its end time omits the key.
            assert reply.get("worker_end_time", "absent") == (
                "absent" if end_time is None else end_time
            )
        written = request_json(
            url,
            f"{path}/write-file",
            method="POST",
            token=token,
            payload={"path": "/tmp/f", "content_b64": encode_bytes(b"x")},
        )
        assert written == {"written": True}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
