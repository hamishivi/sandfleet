from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from sandfleet.egress import EgressSettings
from sandfleet.slurm import SlurmLauncher, SlurmSettings, slurm_time_seconds


def settings(tmp_path):
    return SlurmSettings(
        partition="compute",
        account="research",
        time_limit="01:00:00",
        source_root="/shared/sandfleet",
        agent_python="/usr/bin/python3",
        log_dir=str(tmp_path),
        apptainer_binary="/path/to/apptainer",
        worker_python="/usr/bin/python3",
    )


def test_submit_worker_batch_uses_one_bounded_job_array(monkeypatch, tmp_path):
    seen = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="12345;cluster-a\n", stderr="")

    monkeypatch.setattr("sandfleet.slurm.subprocess.run", fake_run)
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    submission = launcher.submit_worker_batch(
        pool="rollouts", worker_count=32, slots=16, cpus=1, memory_mb=4096
    )
    assert submission.submission_id == "12345"
    assert submission.worker_count == 32
    assert len(seen) == 1
    command = seen[0]
    assert "--array=0-31" in command
    assert "--cpus-per-task=16" in command
    assert "--mem=66048M" in command
    assert "--output=" + str(tmp_path) + "/agent-%A_%a.out" in command
    assert not any(argument.startswith("--qos=") for argument in command)
    assert "--advertise-mode" in command[-1]
    assert "--controller-loss-timeout-seconds 3600" in command[-1]
    assert "fqdn" in command[-1]
    assert "sandfleet.agent" in command[-1]


def test_worker_command_forwards_gpu_container_mode(monkeypatch, tmp_path):
    seen = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="12345;cluster-a\n", stderr="")

    monkeypatch.setattr("sandfleet.slurm.subprocess.run", fake_run)
    configured = replace(settings(tmp_path), gpu_container_mode="nvccli")
    SlurmLauncher(
        configured,
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    ).submit_worker_batch(pool="gpu", worker_count=1, slots=1, cpus=1, memory_mb=4096)
    assert "--gpu-container-mode nvccli" in seen[0][-1]


@pytest.mark.parametrize(
    ("internet", "expected"),
    [
        ("off", None),
        ("any", "--internet any --internet-allowed-hosts pypi.org,github.com"),
        (
            "allowlist",
            "--internet allowlist --internet-allowed-hosts pypi.org,github.com "
            "--internet-max-connections 256 --internet-max-tunnels-per-lease 32 "
            "--internet-idle-timeout-seconds 300",
        ),
    ],
)
def test_worker_command_forwards_internet_policy(
    monkeypatch, tmp_path, internet, expected
):
    seen = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="12345\n", stderr="")

    monkeypatch.setattr("sandfleet.slurm.subprocess.run", fake_run)
    configured = replace(
        settings(tmp_path),
        internet=EgressSettings(
            mode=internet, allowed_hosts=("pypi.org", "github.com")
        ),
    )
    SlurmLauncher(
        configured,
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    ).submit_worker_batch(pool="p", worker_count=1, slots=1, cpus=1, memory_mb=4096)
    if expected is None:
        assert "--internet" not in seen[0][-1]
    else:
        assert expected in seen[0][-1]


def test_removed_fakeroot_network_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="fakeroot"):
        replace(settings(tmp_path), network_mode="fakeroot")


def test_worker_submission_quotes_environment_in_the_wrapped_script(
    monkeypatch, tmp_path
):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service token;still-a-path",
        controller_url="http://controller:8765",
    )
    monkeypatch.setenv("PYTHONPATH", "/shared/existing path;still-a-path")
    launcher.submit_worker_batch(
        pool="secure", worker_count=1, slots=1, cpus=1, memory_mb=512
    )
    command = seen[0]
    wrapped = command[-1]
    assert not any(argument.startswith("--export") for argument in command)
    assert (
        "SANDFLEET_TOKEN_FILE='/shared/secrets/service token;still-a-path'" in wrapped
    )
    assert (
        "PYTHONPATH='/shared/sandfleet:/shared/existing path;still-a-path'" in wrapped
    )
    assert "SANDFLEET_TOKEN=" not in wrapped


def test_worker_submission_includes_optional_qos(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        ),
    )
    launcher = SlurmLauncher(
        replace(
            settings(tmp_path),
            qos="high-priority",
            agent_advertise_mode="route-ip",
        ),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    launcher.submit_worker_batch(
        pool="qos", worker_count=1, slots=1, cpus=1, memory_mb=512
    )
    assert "--qos=high-priority" in seen[0]
    assert "route-ip" in seen[0][-1]


def test_gpu_worker_submission_allocates_parent_and_exact_child_gres(
    monkeypatch, tmp_path
):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    launcher.submit_worker_batch(
        pool="gpu",
        worker_count=1,
        slots=4,
        cpus=2,
        memory_mb=4096,
        gpus=2,
        gpu_type="h100-80gb",
    )
    command = seen[0]
    assert "--gres=gpu:h100-80gb:8" in command
    assert "--gpus 2 --gpu-type h100-80gb" in command[-1]


@pytest.mark.parametrize(
    ("cpus", "expected_total"),
    [(1, 32), (2, 32), (3, 64)],
)
def test_worker_cpu_request_respects_allocation_granularity(
    monkeypatch, tmp_path, cpus, expected_total
):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(returncode=0, stdout="12345\n", stderr="")
        ),
    )
    launcher = SlurmLauncher(
        replace(settings(tmp_path), cpu_allocation_granularity=2),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    launcher.submit_worker_batch(
        pool="granularity", worker_count=1, slots=16, cpus=cpus, memory_mb=512
    )
    assert f"--cpus-per-task={expected_total}" in seen[0]


def test_cpu_allocation_granularity_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        replace(settings(tmp_path), cpu_allocation_granularity=0)


def test_snapshot_expands_and_groups_array_tasks_in_one_squeue(monkeypatch, tmp_path):
    seen = []

    def fake_run(command, **_kwargs):
        seen.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="12345|0|12345_0|RUNNING\n12345|1|12345_1|PENDING\n77|N/A|77|RUNNING\n",
            stderr="",
        )

    monkeypatch.setattr("sandfleet.slurm.subprocess.run", fake_run)
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    snapshot = launcher.snapshot(["12345", "77"])
    assert snapshot.ok
    assert snapshot.states == {
        "12345": {"12345_0": "RUNNING", "12345_1": "PENDING"},
        "77": {"77": "RUNNING"},
    }
    assert seen == [
        [
            "squeue",
            "--array",
            "--noheader",
            "--jobs",
            "12345,77",
            "--format=%F|%K|%i|%T",
        ]
    ]


def test_gpu_inventory_is_one_partition_scoped_advisory_snapshot(monkeypatch, tmp_path):
    seen = []
    payload = {
        "nodes": [
            {
                "partitions": ["compute"],
                "state": ["MIXED"],
                "tres": "cpu=64,gres/gpu:h100-80gb=8",
                "tres_used": "cpu=10,gres/gpu:h100-80gb=3",
            },
            {
                "partitions": ["compute"],
                "state": ["DRAINED"],
                "tres": "gres/gpu:h100-80gb=4",
                "tres_used": "gres/gpu:h100-80gb=0",
            },
            {
                "partitions": ["other"],
                "state": ["IDLE"],
                "tres": "gres/gpu:h100-80gb=99",
                "tres_used": "",
            },
        ]
    }
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    inventory = launcher.discover_gpu_inventory({"h100-80gb"})
    assert seen == [["scontrol", "show", "nodes", "--json"]]
    assert inventory["h100-80gb"] == {
        "configured": 12,
        "allocated": 3,
        "free_estimate": 5,
        "largest_node_free_estimate": 5,
        "schedulable_nodes": 1,
        "unavailable_nodes": 1,
    }


def test_gpu_inventory_tolerates_string_and_missing_node_fields(monkeypatch, tmp_path):
    payload = {
        "nodes": [
            "not-a-node",
            {
                "partitions": "compute,debug",
                "state": "DOWN+DRAIN",
                "tres": "gres/gpu:t=4",
            },
            {"partitions": "compute-large", "state": "IDLE", "tres": "gres/gpu:t=8"},
            {"partitions": ["compute"], "state": None, "tres": "gres/gpu:t=2"},
        ]
    }
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    inventory = launcher.discover_gpu_inventory({"t"})["t"]
    assert inventory["configured"] == 6
    assert inventory["unavailable_nodes"] == 1
    assert inventory["schedulable_nodes"] == 1


def test_gpu_inventory_uses_typed_gres_with_untyped_tres(monkeypatch, tmp_path):
    payload = {
        "nodes": [
            {
                "partitions": ["compute"],
                "state": ["MIXED"],
                "gres": "gpu:model-a:8(S:0-1)",
                "gres_used": "gpu:model-a:3(IDX:1,3-4)",
                "tres": "cpu=64,gres/gpu=8",
                "tres_used": "cpu=10,gres/gpu=3",
            }
        ]
    }
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )

    assert launcher.discover_gpu_inventory({"model-a"})["model-a"] == {
        "configured": 8,
        "allocated": 3,
        "free_estimate": 5,
        "largest_node_free_estimate": 5,
        "schedulable_nodes": 1,
        "unavailable_nodes": 0,
    }


def test_gpu_inventory_conservatively_attributes_untyped_usage(monkeypatch, tmp_path):
    payload = {
        "nodes": [
            {
                "partitions": ["compute"],
                "state": ["MIXED"],
                "gres": "gpu:model-a:4(S:0),gpu:model-b:4(S:1)",
                "tres": "cpu=64,gres/gpu=8",
                "tres_used": "cpu=10,gres/gpu=3",
            }
        ]
    }
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )

    inventory = launcher.discover_gpu_inventory({"model-a", "model-b"})
    assert inventory["model-a"]["configured"] == 4
    assert inventory["model-a"]["allocated"] == 3
    assert inventory["model-a"]["free_estimate"] == 1
    assert inventory["model-b"]["configured"] == 4
    assert inventory["model-b"]["allocated"] == 3
    assert inventory["model-b"]["free_estimate"] == 1


def test_gpu_access_probe_validates_without_submitting(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(
                returncode=0, stdout="Job 123 to start at 2030-01-01", stderr=""
            )
        ),
    )
    launcher = SlurmLauncher(
        replace(settings(tmp_path), qos="gpu-qos"),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    result = launcher.probe_gpu_access("h100-80gb")
    assert result["ok"]
    assert "--test-only" in seen[0]
    assert "--gres=gpu:h100-80gb:1" in seen[0]
    assert "--qos=gpu-qos" in seen[0]
    assert "--parsable" not in seen[0]


def test_gpu_qos_quota_parses_group_tres_headroom(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: (
            seen.append(command)
            or SimpleNamespace(
                returncode=0,
                stdout="QOS=example_gpu_qos GrpTRES=cpu=128,gres/gpu=16(3)\n",
                stderr="",
            )
        ),
    )
    launcher = SlurmLauncher(
        replace(settings(tmp_path), qos="example_gpu_qos"),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    assert launcher.discover_gpu_quota() == {
        "limit": 16,
        "allocated": 3,
        "free_estimate": 13,
    }
    assert seen == [
        [
            "scontrol",
            "show",
            "assoc_mgr",
            "qos=example_gpu_qos",
            "flags=qos",
        ]
    ]


def test_gpu_qos_quota_ignores_per_user_limits(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "QOS=example_gpu_qos\n"
                "    GrpTRES=cpu=128,gres/gpu=N(3)\n"
                "    MaxTRESPU=gres/gpu=8(0)\n"
            ),
            stderr="",
        ),
    )
    launcher = SlurmLauncher(
        replace(settings(tmp_path), qos="example_gpu_qos"),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    assert launcher.discover_gpu_quota() is None


def test_gpu_qos_quota_does_not_misreport_an_account_subrecord(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                "QOS=normal GrpTRES=gres/gpu=N(0)\n"
                "Account Limits\n  other GrpTRES=gres/gpu=8(2)\n"
            ),
            stderr="",
        ),
    )
    launcher = SlurmLauncher(
        replace(settings(tmp_path), qos="normal"),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    assert launcher.discover_gpu_quota() is None


def test_scheduler_failure_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="controller unavailable"
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    snapshot = launcher.snapshot(["12"])
    assert not snapshot.ok
    assert snapshot.error == "controller unavailable"


def test_malformed_scheduler_output_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="not-a-record\n", stderr=""
        ),
    )
    launcher = SlurmLauncher(
        settings(tmp_path),
        service_token_file="/shared/secrets/service.token",
        controller_url="http://controller:8765",
    )
    snapshot = launcher.snapshot(["12"])
    assert not snapshot.ok
    assert snapshot.error == "Malformed squeue output: 'not-a-record'"


def test_cancel_failure_is_not_ignored(monkeypatch):
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="permission denied"
        ),
    )
    with pytest.raises(RuntimeError, match="scancel failed: permission denied"):
        SlurmLauncher.cancel(["12"])


def test_worker_command_enables_audit_log(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        "sandfleet.slurm.subprocess.run",
        lambda command, **_: seen.append(command)
        or SimpleNamespace(returncode=0, stdout="1\n", stderr=""),
    )
    configured = replace(settings(tmp_path), audit_log_dir="/logs/audit")
    for value in (settings(tmp_path), configured):
        SlurmLauncher(
            value, service_token_file="/t", controller_url="http://c:1"
        ).submit_worker_batch(pool="p", worker_count=1, slots=1, cpus=1, memory_mb=64)
    assert "--audit-log" not in seen[0][-1]
    assert "--audit-log-dir /logs/audit --audit-log-max-mb 256" in seen[1][-1]


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("90", 90 * 60),
        ("90:30", 90 * 60 + 30),
        ("04:00:00", 4 * 3600),
        ("2-0", 48 * 3600),
        ("1-02:30", 26 * 3600 + 30 * 60),
        ("1-02:30:15", 26 * 3600 + 30 * 60 + 15),
        (" 12:00:00 ", 12 * 3600),
        ("0", None),  # Slurm: no time limit
        ("00:00:00", None),
        ("UNLIMITED", None),
        ("infinite", None),
        ("", None),
        ("1:2:3:4", None),
    ],
)
def test_slurm_time_seconds_parses_sbatch_time_forms(value, seconds):
    assert slurm_time_seconds(value) == seconds
