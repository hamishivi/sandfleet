"""Literal command environment transport and argv-leak regressions."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from sandfleet.environment import validate_command_env
from sandfleet.step_worker import ApptainerInstance, WorkerSession
from sandfleet.step_backend import SlurmStepSandbox
from sandfleet.client import SandboxLease, SandfleetClient


@pytest.mark.parametrize(
    "env", [[], {"bad=name": "x"}, {"A": 3}, {"A": "x\0y"}, {1: "x"}]
)
def test_invalid_environment_rejected(env):
    with pytest.raises(ValueError):
        validate_command_env(env)


def test_client_uses_distinct_endpoint_without_embedding_values():
    lease = SandboxLease(
        controller=SandfleetClient("http://controller", "c"),
        lease_id="id",
        lease_token="t",
        agent_url="http://agent",
        pool="p",
    )
    lease._request = Mock(return_value={"stdout": "", "stderr": "", "exit_code": 0})
    lease.run_command("echo ok", env={"JAVA_HOME": "/java"})
    args, kwargs = lease._request.call_args
    assert args == ("/exec-env",)
    assert kwargs["payload"]["command"] == "echo ok"
    assert kwargs["payload"]["env"] == {"JAVA_HOME": "/java"}


def test_backend_uses_distinct_worker_operation():
    backend = object.__new__(SlurmStepSandbox)
    backend._rpc = Mock(return_value={})
    backend.run_command("echo ok", env={"A": "literal"})
    assert backend._rpc.call_args.args == ("run_command_env",)
    assert backend._rpc.call_args.kwargs["env"] == {"A": "literal"}


def test_worker_environment_is_literal_and_absent_from_argv():
    instance = object.__new__(ApptainerInstance)
    instance.apptainer = "apptainer"
    instance.name = "test"
    instance.env = Mock(return_value={"PATH": "/usr/bin"})
    instance._drain_threads = []
    value = '/java/$(touch /not-executed); "quoted"\nnext'
    with patch("sandfleet.step_worker.subprocess.Popen") as popen, patch(
        "sandfleet.step_worker.capture_process",
        return_value=subprocess.CompletedProcess([], 0, b"", b""),
    ):
        instance._exec_internal(
            ["bash", "-c", "echo ok"], command_env={"JAVA_HOME": value}
        )
    args, kwargs = popen.call_args
    assert value not in " ".join(args[0])
    assert "--no-eval" in args[0]
    assert kwargs["env"]["APPTAINERENV_JAVA_HOME"] == value
    assert instance.env.return_value == {"PATH": "/usr/bin"}


def test_worker_requires_environment_field_for_new_operation():
    session = object.__new__(WorkerSession)
    session._require_instance = Mock()
    with pytest.raises(KeyError, match="env"):
        session._run_command_env({"command": "echo ok"})
    session._require_instance.return_value.run_command.assert_not_called()
