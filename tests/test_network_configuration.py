"""Keep operator network settings intact from controller CLI to sandbox config."""

import shlex
from types import SimpleNamespace

import pytest

from sandfleet import agent, controller
from sandfleet.step_backend import SlurmStepSandbox


def controller_arguments(tmp_path):
    return [
        "--partition",
        "site-cpu",
        "--account",
        "team-account",
        "--source-root",
        str(tmp_path),
        "--log-dir",
        str(tmp_path / "logs"),
    ]


def test_operator_network_config_reaches_step_worker(monkeypatch, tmp_path):
    arguments = controller_arguments(tmp_path) + ["--network-mode", "loopback"]
    args = controller.build_parser().parse_args(arguments)
    state = controller.build_state(
        args,
        url="http://controller.invalid:8765",
        token="test-service",
        client_token="test-client",
        metrics_token="test-metrics",
        service_token_file=str(tmp_path / "service-token"),
    )
    commands = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            commands.append(command)
            or SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        ),
    )
    state.launcher.submit_worker_batch(
        pool="network-test", worker_count=1, slots=1, cpus=1, memory_mb=512
    )
    wrapped = commands[0][-1]
    argv = shlex.split(wrapped.rpartition("; exec ")[2])
    agent_args = agent.build_parser().parse_args(
        argv[argv.index("sandfleet.agent") + 1 :]
    )
    monkeypatch.setattr(agent.os, "sched_getaffinity", lambda _pid: {3}, raising=False)
    monkeypatch.setattr(agent, "current_cgroup_path", lambda: "/slurm/test-allocation")
    agent_state = agent.build_state(agent_args, "test-service", "local-test")
    sandbox = agent_state.new_sandbox()
    config = sandbox._config(
        image="/shared/test.sif",
        command_timeout=30,
        pwd="/workspace",
        extra_start_flags=None,
    )

    assert config["network_mode"] == "loopback"
    assert config["relay_max_pending"] == 32  # one slot gets every forwarder slot
    assert "network_dns" not in config
    assert "--partition=site-cpu" in commands[0]
    assert "--account=team-account" in commands[0]
    assert sandbox.process is None
    assert sandbox.connection is None


@pytest.mark.parametrize("mode", ["none", "host", "fakeroot", "slirp4netns"])
def test_controller_rejects_network_modes_without_isolation_and_egress(tmp_path, mode):
    with pytest.raises(SystemExit):
        controller.build_parser().parse_args(
            controller_arguments(tmp_path) + ["--network-mode", mode]
        )


def test_removed_sandbox_dns_option_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        controller.build_parser().parse_args(
            controller_arguments(tmp_path) + ["--network-dns", "192.0.2.53"]
        )


@pytest.mark.parametrize("mode", ["none", "host", "fakeroot"])
def test_backend_rejects_unsupported_network_before_step_launch(mode):
    with pytest.raises(ValueError, match="network_mode"):
        SlurmStepSandbox(cpus=1, memory_mb=512, network_mode=mode)


def test_backend_rejects_old_worker_status_without_network_contract():
    sandbox = SlurmStepSandbox(cpus=1, memory_mb=512)
    with pytest.raises(KeyError, match="network"):
        sandbox._record_status(
            {
                "instance_name": None,
                "cgroup": {
                    "memory.max.effective": str(512 * 1024 * 1024),
                    "cpuset.cpus.effective": "3",
                },
            }
        )
