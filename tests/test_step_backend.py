import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest

from sandfleet import step_worker
from sandfleet.step_backend import SlurmStepSandbox, parse_cpuset
from sandfleet.step_worker import (
    ApptainerInstance,
    accessible_nvidia_devices,
    current_cgroup,
    normalize_image,
    resolve_gpu_container_mode,
)


def test_normalize_image():
    assert normalize_image("ubuntu:22.04") == "docker://ubuntu:22.04"
    assert normalize_image("docker://ubuntu:22.04") == "docker://ubuntu:22.04"
    assert normalize_image("/shared/image.sif") == "/shared/image.sif"


def test_debug_command_contains_resource_limits():
    backend = SlurmStepSandbox(
        cpus=2,
        memory_mb=8192,
        node="cpu-1",
        worker_python="/usr/bin/python3",
        srun_extra_args=("--qos=debug",),
    )
    command = shlex.join(backend._srun_command("--socket", "<socket>"))
    assert "--cpus-per-task=2" in command
    assert "--mem=8192M" in command
    assert "--nodelist=cpu-1" in command
    assert "--gres=none" in command
    assert "--kill-on-bad-exit=1" in command
    assert "--qos=debug" in command
    assert "--socket" in command


def test_gpu_debug_command_requests_an_exact_step_allocation():
    backend = SlurmStepSandbox(
        cpus=2,
        memory_mb=8192,
        gpus=2,
        gpu_type="h100-80gb",
        worker_python="/usr/bin/python3",
    )
    command = shlex.join(backend._srun_command("--socket", "<socket>"))
    assert "--gres=gpu:h100-80gb:2" in command
    assert "--gpus 2" in command


def test_prepare_and_recycle_keep_the_same_step(monkeypatch):
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    launches = []
    operations = []

    class Process:
        pid = 123

        def poll(self):
            return None

    def launch():
        launches.append(True)
        backend.process = Process()
        backend.connection = object()

    def rpc(operation, **_kwargs):
        operations.append(operation)
        return {
            "instance_name": None,
            "network": {},
            "readonly_mounts": [],
            "readonly_mount_roots": [],
            "agent_readonly_mount_roots": [],
            "cgroup": {
                "memory.max.effective": "536870912",
                "cpuset.cpus.effective": "7",
            },
        }

    monkeypatch.setattr(backend, "_launch_worker", launch)
    monkeypatch.setattr(backend, "_rpc", rpc)
    prepared = backend.prepare()
    backend.prepare()
    recycled = backend.recycle()
    assert len(launches) == 1
    assert operations == ["status", "status", "stop"]
    assert prepared["healthy"] and recycled["healthy"]


def test_start_and_restart_never_create_unpaced_steps(monkeypatch):
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    launched = []
    monkeypatch.setattr(backend, "_launch_worker", lambda: launched.append(True))

    with pytest.raises(RuntimeError, match="not running"):
        backend.start(image="/shared/image.sif")
    with pytest.raises(RuntimeError, match="not running"):
        backend.restart(image="/shared/image.sif")
    assert launched == []


def test_step_start_timeout_explains_an_empty_srun_log(monkeypatch):
    backend = SlurmStepSandbox(cpus=1, memory_mb=512, start_timeout=0)

    class Process:
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout):
            pass

        def kill(self):
            pass

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr("sandfleet.step_backend.shutil.which", lambda _name: "/srun")
    monkeypatch.setattr(
        "sandfleet.step_backend.subprocess.Popen", lambda *_args, **_kwargs: Process()
    )
    with pytest.raises(TimeoutError, match="allocation may lack free step resources"):
        backend._launch_worker()


def test_step_status_rejects_weaker_memory_limit():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    with pytest.raises(RuntimeError, match="expected at most 512 MiB"):
        backend._record_status(
            {
                "instance_name": None,
                "cgroup": {
                    "memory.max.effective": str(513 * 1024 * 1024),
                    "cpuset.cpus.effective": "7",
                },
            }
        )


def test_step_status_accepts_slurm_hyperthread_expansion():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    backend._record_status(
        {
            "instance_name": None,
            "network": {},
            "readonly_mounts": [],
            "readonly_mount_roots": [],
            "agent_readonly_mount_roots": [],
            "cgroup": {
                "path": "/slurm/job/step/task",
                "memory.max.effective": str(512 * 1024 * 1024),
                "cpuset.cpus.effective": "7,103",
            },
        }
    )
    assert backend.cpuset == frozenset({7, 103})
    assert backend.metadata["cpuset.cpu_count"] == 2


def test_gpu_step_status_requires_exact_device_and_visibility_counts():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512, gpus=1, gpu_type="h100-80gb")
    backend._record_status(
        {
            "instance_name": None,
            "network": {},
            "readonly_mounts": [],
            "readonly_mount_roots": [],
            "agent_readonly_mount_roots": [],
            "cgroup": {
                "path": "/slurm/job/step/task",
                "memory.max.effective": str(512 * 1024 * 1024),
                "cpuset.cpus.effective": "7",
                "gpu": {
                    "accessible": [{"path": "/dev/nvidia3", "major": 195, "minor": 3}],
                    "identities": [
                        {
                            "index": 3,
                            "uuid": "GPU-123",
                            "name": "NVIDIA H100 80GB HBM3",
                            "memory_mb": 81559,
                        }
                    ],
                    "denied": ["/dev/nvidia0"],
                    "errors": [],
                    "cuda_visible_devices": "3",
                },
            },
        }
    )
    assert backend.gpu_devices == frozenset({"/dev/nvidia3"})
    assert backend.metadata["gpu"]["identities"][0]["uuid"] == "GPU-123"


def test_gpu_step_status_rejects_an_unconstrained_device_cgroup():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512, gpus=1, gpu_type="h100-80gb")
    with pytest.raises(RuntimeError, match="access 2 GPU devices; expected exactly 1"):
        backend._record_status(
            {
                "instance_name": None,
                "cgroup": {
                    "memory.max.effective": str(512 * 1024 * 1024),
                    "cpuset.cpus.effective": "7",
                    "gpu": {
                        "accessible": [
                            {"path": "/dev/nvidia0"},
                            {"path": "/dev/nvidia1"},
                        ],
                        "errors": [],
                        "cuda_visible_devices": "0",
                    },
                },
            }
        )


def test_cpu_step_status_rejects_access_to_a_gpu_device():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    with pytest.raises(RuntimeError, match="expected exactly 0"):
        backend._record_status(
            {
                "instance_name": None,
                "cgroup": {
                    "memory.max.effective": str(512 * 1024 * 1024),
                    "cpuset.cpus.effective": "7",
                    "gpu": {
                        "accessible": [{"path": "/dev/nvidia0"}],
                        "errors": [],
                    },
                },
            }
        )


def test_gpu_container_mode_is_worker_controlled(monkeypatch):
    instance = ApptainerInstance(
        image="ubuntu:22.04",
        timeout=30,
        pwd="/workspace",
        cache_dir=None,
        tmp_dir=None,
        extra_start_flags=[],
        apptainer_binary="apptainer",
        gpu_enabled=True,
        gpu_container_mode="nv",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-123")
    assert "--nv" in instance.start_flags
    assert instance.env()["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "GPU-123"
    with pytest.raises(ValueError, match="controlled by the Sandfleet worker"):
        ApptainerInstance(
            image="ubuntu:22.04",
            timeout=30,
            pwd="/workspace",
            cache_dir=None,
            tmp_dir=None,
            extra_start_flags=["--nvccli"],
            apptainer_binary="apptainer",
            gpu_enabled=True,
        )


def test_operator_can_select_nvccli_gpu_integration(monkeypatch):
    def instance(**kwargs):
        return ApptainerInstance(
            image="ubuntu:22.04",
            timeout=30,
            pwd="/workspace",
            cache_dir=None,
            tmp_dir=None,
            extra_start_flags=[],
            apptainer_binary="apptainer",
            **kwargs,
        )

    gpu = instance(gpu_enabled=True, gpu_container_mode="nvccli")
    assert "--nvccli" in gpu.start_flags
    assert "--nv" not in gpu.start_flags
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu.env()["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "0"
    assert gpu.env()["NVIDIA_VISIBLE_DEVICES"] == "0"
    cpu = instance(gpu_enabled=False, gpu_container_mode="nvccli")
    assert not {"--nv", "--nvccli"} & set(cpu.start_flags)
    with pytest.raises(ValueError, match="gpu_container_mode"):
        instance(gpu_enabled=True, gpu_container_mode="rocm")


@pytest.mark.parametrize(
    ("imex", "nvccli", "expected"),
    [(False, True, "nv"), (True, True, "nvccli"), (True, False, None)],
)
def test_auto_gpu_container_mode_uses_nvccli_only_on_imex_hosts(
    monkeypatch, imex, nvccli, expected
):
    monkeypatch.setattr(
        step_worker.os.path,
        "isdir",
        lambda path: imex and path == "/dev/nvidia-caps-imex-channels",
    )
    monkeypatch.setattr(
        step_worker.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if nvccli else None,
    )
    if expected is None:
        with pytest.raises(RuntimeError, match="nvidia-container-cli"):
            resolve_gpu_container_mode("auto")
    else:
        assert resolve_gpu_container_mode("auto") == expected
    assert resolve_gpu_container_mode("nv") == "nv"


def test_gpu_probe_reports_device_node_and_nvidia_identity(monkeypatch, tmp_path):
    device = tmp_path / "nvidia3"
    device.touch()
    real_glob = Path.glob
    monkeypatch.setattr(
        Path,
        "glob",
        lambda path, pattern: [device]
        if path == Path("/dev")
        else real_glob(path, pattern),
    )
    monkeypatch.setattr(
        "sandfleet.step_worker.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="3, GPU-123, NVIDIA H100 80GB HBM3, 81559\n",
            stderr="",
        ),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")

    gpu = accessible_nvidia_devices()
    assert gpu["accessible"] == [{"path": str(device), "major": 0, "minor": 0}]
    assert gpu["identities"] == [
        {
            "index": 3,
            "uuid": "GPU-123",
            "name": "NVIDIA H100 80GB HBM3",
            "memory_mb": 81559,
        }
    ]


def test_step_status_rejects_an_empty_cpuset():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    with pytest.raises(RuntimeError, match="empty CPU set"):
        backend._record_status(
            {
                "instance_name": None,
                "cgroup": {
                    "memory.max.effective": str(512 * 1024 * 1024),
                    "cpuset.cpus.effective": "",
                },
            }
        )


def test_parse_cpuset():
    assert parse_cpuset("0-2,7,10-11") == frozenset({0, 1, 2, 7, 10, 11})


def test_cgroup_metadata_requires_a_finite_memory_ancestor(monkeypatch):
    def read_text(path, **_kwargs):
        name = str(path)
        if name == "/proc/self/cgroup":
            return "0::/slurm/job/step/task\n"
        if name.endswith("memory.current"):
            return "1024\n"
        if name.endswith("cpuset.cpus.effective"):
            return "7\n"
        if name == "/sys/fs/cgroup/slurm/job/step/memory.max":
            return "536870912\n"
        if name.endswith("memory.max"):
            return "max\n"
        raise FileNotFoundError(name)

    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    metadata = current_cgroup()
    assert metadata["memory.max"] == "max"
    assert metadata["memory.max.effective"] == "536870912"
    assert metadata["memory.max.effective_path"] == "/sys/fs/cgroup/slurm/job/step"


def test_cgroup_metadata_rejects_an_unlimited_step(monkeypatch):
    def read_text(path, **_kwargs):
        name = str(path)
        if name == "/proc/self/cgroup":
            return "0::/slurm/job/step/task\n"
        if name.endswith("memory.current"):
            return "1024\n"
        if name.endswith("cpuset.cpus.effective"):
            return "7\n"
        if name.endswith("memory.max"):
            return "max\n"
        raise FileNotFoundError(name)

    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    with pytest.raises(RuntimeError, match="no finite cgroup memory limit"):
        current_cgroup()


def test_cgroup_metadata_accepts_a_finite_limit_below_a_controllerless_root(
    monkeypatch,
):
    def exists(path):
        return str(path) != "/sys/fs/cgroup/memory.max"

    def read_text(path, **_kwargs):
        name = str(path)
        if name == "/proc/self/cgroup":
            return "0::/slurm/job/step/task\n"
        if name.endswith("memory.current"):
            return "1024\n"
        if name.endswith("cpuset.cpus.effective"):
            return "7\n"
        if name == "/sys/fs/cgroup/slurm/job/step/memory.max":
            return "536870912\n"
        if name.endswith("memory.max"):
            return "max\n"
        raise FileNotFoundError(name)

    monkeypatch.setattr("pathlib.Path.exists", exists)
    monkeypatch.setattr("pathlib.Path.read_text", read_text)
    metadata = current_cgroup()
    assert metadata["memory.max.effective"] == "536870912"


def test_cancel_command_uses_control_channel_and_returns_worker_state():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)

    class FakeControl:
        def __init__(self):
            self.sent = []

        def send(self, message):
            self.sent.append(message)

        def poll(self, _timeout):
            return True

        def recv(self):
            return {
                "ok": True,
                "result": {"cancelled": True, "interrupted": True, "state": "running"},
            }

    control = FakeControl()
    backend.control_connection = control
    result = backend.cancel_command("job-1")
    assert control.sent == [{"operation": "cancel", "command_id": "job-1"}]
    assert result == {"cancelled": True, "interrupted": True, "state": "running"}


def test_kill_processes_uses_control_channel_with_a_longer_deadline():
    backend = SlurmStepSandbox(cpus=1, memory_mb=512)

    class FakeControl:
        def __init__(self):
            self.sent = []
            self.timeouts = []

        def send(self, message):
            self.sent.append(message)

        def poll(self, timeout):
            self.timeouts.append(timeout)
            return True

        def recv(self):
            return {"ok": True, "result": {"killed": 2, "cancelled_commands": []}}

    control = FakeControl()
    backend.control_connection = control
    assert backend.kill_processes() == {"killed": 2, "cancelled_commands": []}
    assert control.sent == [{"operation": "kill_processes"}]
    assert control.timeouts[0] > 40


def test_cancel_command_without_control_channel_raises():
    from sandfleet.errors import SandfleetError

    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    backend.control_connection = None
    with pytest.raises(SandfleetError, match="control channel is unavailable"):
        backend.cancel_command("job-1")


def test_cancel_command_drops_channel_and_raises_on_timeout():
    from sandfleet.errors import SandfleetError

    backend = SlurmStepSandbox(cpus=1, memory_mb=512)

    class StuckControl:
        def send(self, message):
            pass

        def poll(self, _timeout):
            return False  # worker never answers

        def recv(self):  # pragma: no cover - never reached
            raise AssertionError("recv must not be called after a poll timeout")

    backend.control_connection = StuckControl()
    with pytest.raises(SandfleetError, match="Lost sandbox control channel"):
        backend.cancel_command("job-1")
    assert backend.control_connection is None


def test_establish_control_failure_is_fatal_not_best_effort(monkeypatch):
    import sandfleet.step_backend as sb

    backend = SlurmStepSandbox(cpus=1, memory_mb=512)
    # A required same-version channel: a control socket that never appears must
    # fail the worker launch loudly, not leave a silently-unavailable channel.
    monkeypatch.setattr(sb, "CONTROL_CONNECT_TIMEOUT_SECONDS", 0.2)
    terminated = []
    monkeypatch.setattr(backend, "_terminate_worker", lambda: terminated.append(True))
    with pytest.raises(RuntimeError, match="control channel did not come up"):
        backend._establish_control("/nonexistent/never/control.sock", b"k" * 32)
    assert terminated == [True]  # the worker was torn down on the fatal failure
