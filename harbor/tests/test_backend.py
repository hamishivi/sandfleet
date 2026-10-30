import asyncio
import base64
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[2]))
from image_manifest import context_digest, resolve_image  # noqa: E402
from sandfleet_harbor import (  # noqa: E402
    AGENT_MOUNTS_FEATURE,
    CHUNK_BYTES,
    REQUIRED_FEATURES,
    VERIFIER_MOUNTS_FEATURE,
    SandfleetEnvironment,
    resolve_gpu_types,
)
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy, TpuSpec  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402
from harbor.models.trial.config import ResourceMode  # noqa: E402


class FakeLease:
    def __init__(self):
        self.files = {}
        self.closed = False
        self.started = None
        self.commands = []
        self.entered = threading.Event()
        self.interrupted = threading.Event()
        self.cancelled_commands = []
        self.command_ids = []
        self.mount_roots = None  # a pre-0.13 worker reports none
        self.mount_requests = []
        self.reported_mounts = None
        self.gpu_uuids = []  # the step's GPUs; [] for a CPU lease
        self.gpu_type = None
        self.visible_uuids = None  # what nvidia-smi lists inside (default: the step's)
        self.smi_exit = 0
        self.imex_channels = []  # what /dev/nvidia-caps-imex-channels lists inside
        self.network = {"mode": "none", "internet": False, "helper_pid": None}

    def status(self):
        status = {
            "cpus": 1,
            "memory_mb": 1024,
            "worker_pid": 123,
            "network": self.network,
            "cgroup": {"memory.max.effective": "1073741824"},
            "storage": {
                "mode": "disk-overlay",
                "storage_mb": 1024,
                "policy": "nominal-image-mib",
                "image_sha256": "a" * 64,
                "writable_paths_verified": True,
            },
        }
        if self.mount_roots is not None:
            status["readonly_mount_roots"] = self.mount_roots
        if self.gpu_uuids:
            status["gpus"] = len(self.gpu_uuids)
            status["gpu_type"] = self.gpu_type
            status["cgroup"]["gpu"] = {
                "accessible": [
                    {"path": f"/dev/nvidia{index}"}
                    for index in range(len(self.gpu_uuids))
                ],
                "identities": [
                    {"index": index, "uuid": uuid, "name": "NVIDIA GB300"}
                    for index, uuid in enumerate(self.gpu_uuids)
                ],
                "cuda_visible_devices": ",".join(
                    str(index) for index in range(len(self.gpu_uuids))
                ),
            }
        return status

    def start(self, image, **kwargs):
        self.started = kwargs
        return self.status()

    def add_readonly_mounts(self, mounts):
        self.mount_requests.append(mounts)
        described = [{**item, "read_only": True} for item in mounts]
        return {
            **self.status(),
            "readonly_mounts": (
                described if self.reported_mounts is None else self.reported_mounts
            ),
            "added_mounts": described,
            "stopped_processes": 3,
        }

    def run_command(self, command, timeout, command_id=None, *, env=None):
        self.last_env = env
        self.commands.append((command, timeout))
        self.command_ids.append(command_id)
        if "mkdir -p /logs/agent" in command:
            return SimpleNamespace(
                stdout="", stderr="", exit_code=0, output_truncated=False
            )
        if "nvidia-smi --query-gpu=uuid" in command:
            visible = (
                self.gpu_uuids if self.visible_uuids is None else self.visible_uuids
            )
            return SimpleNamespace(
                stdout="".join(f"{uuid}\n" for uuid in visible)
                + "".join(f"imex: {name}\n" for name in self.imex_channels),
                stderr="" if not self.smi_exit else "NVIDIA-SMI has failed",
                exit_code=self.smi_exit,
                output_truncated=False,
            )
        if "WAIT_FOR_ABORT" in command:
            self.entered.set()
            if not self.interrupted.wait(3):
                raise RuntimeError("abort did not wake exec")
            raise ConnectionError("aborted")
        if "WAIT_FOR_CANCEL" in command:
            self.entered.set()
            if not self.interrupted.wait(3):
                raise RuntimeError("cancel did not wake exec")
            return SimpleNamespace(
                stdout="", stderr="cancelled", exit_code=130, output_truncated=False
            )
        return SimpleNamespace(
            stdout="hello", stderr="warning", exit_code=7, output_truncated=False
        )

    def write_file_chunk(self, path, content, *, offset, truncate):
        before = b"" if truncate else self.files[path]
        self.files[path] = before[:offset] + content

    def read_file_chunk(self, path, *, offset, size):
        return self.files[path][offset : offset + size]

    def abort(self):
        self.interrupted.set()

    def cancel_command(self, command_id):
        self.cancelled_commands.append(command_id)
        self.interrupted.set()
        return {"state": "running", "interrupted": True, "cancelled": True}

    def kill_processes(self):
        self.commands.append(("KILL_PROCESSES", None))
        return {"killed": 2, "cancelled_commands": []}

    def close(self):
        self.closed = True


class BackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.context = self.root / "task" / "environment"
        self.context.mkdir(parents=True)
        (self.context / "Dockerfile").write_text("FROM python:3.12-slim\n")
        (self.context.parent / "task.toml").write_text(
            "[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\n"
        )
        self.manifest = self.root / "images.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "images": {
                        context_digest(self.context): {
                            "sif_path": "/images/task.sif",
                            "sif_sha256": "a" * 64,
                            "user": "root",
                            "workdir": "/workspace",
                            "env": {"IMAGE_VAR": "value"},
                            "build_provenance": "built whole fixture context",
                        }
                    },
                }
            )
        )
        self.lease = FakeLease()
        self.client = unittest.mock.Mock()
        self.client.health.return_value = {
            "ok": True,
            "features": sorted(REQUIRED_FEATURES),
            "max_storage_mb": 1024,
        }
        self.client.acquire.return_value = self.lease
        patcher = patch(
            "sandfleet_harbor.configuration",
            return_value=("http://controller", "private", self.manifest),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("sandfleet_harbor.SandfleetClient", return_value=self.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def environment(self, **kwargs):
        return SandfleetEnvironment(
            environment_dir=self.context,
            environment_name="fixture",
            session_id="unique",
            trial_paths=TrialPaths(self.root / "trial"),
            task_env_config=EnvironmentConfig(cpus=1, memory_mb=1024, storage_mb=1024),
            **kwargs,
        )

    async def test_exact_resources_lifecycle_evidence_and_exit_code(self):
        env = self.environment()
        await env.start()
        self.assertEqual(self.client.acquire.call_args.kwargs["memory_mb"], 1024)
        self.assertFalse(self.client.acquire.call_args.kwargs["internet"])
        self.assertEqual(self.lease.started["storage_mb"], 1024)
        self.assertEqual(self.lease.started["command_timeout"], 3600)
        result = await env.exec(
            "echo hello", cwd="/some path", env={"CUSTOM": "a;b"}, timeout_sec=4.2
        )
        self.assertEqual(
            (result.stdout, result.stderr, result.return_code), ("hello", "warning", 7)
        )
        command, timeout = self.lease.commands[-1]
        self.assertEqual(timeout, 5)
        self.assertNotIn("CUSTOM", command)
        self.assertEqual(self.lease.last_env, {"IMAGE_VAR": "value", "CUSTOM": "a;b"})
        self.assertIn("cd -- '/some path'", command)
        with env.scoped_exec_env({"AGENT_VAR": "x"}):
            await env.exec("true")
        self.assertEqual(self.lease.last_env["AGENT_VAR"], "x")
        evidence = (self.root / "trial/sandfleet-environment.json").read_text()
        self.assertNotIn("private", evidence)
        await env.stop()
        self.assertTrue(self.lease.closed)

    async def test_worker_end_time_is_exposed_only_when_reported(self):
        cases = [
            (1800000000, 1800000000.0),
            (None, None),
            ("soon", None),
            (True, None),
            (10**400, None),
        ]
        for reported, expected in cases:
            with self.subTest(reported=reported):
                status = FakeLease.status(self.lease)
                if reported is not None:
                    status["worker_end_time"] = reported
                self.lease.status = lambda status=status: status
                env = self.environment()
                self.assertIsNone(env.worker_end_time)
                await env.start()
                self.assertEqual(env.worker_end_time, expected)
                await env.stop()
                self.assertIsNone(env.worker_end_time)
                (self.root / "trial/sandfleet-environment.json").unlink()

    def offer_internet(self):
        self.client.health.return_value["features"] = sorted(
            {*REQUIRED_FEATURES, "lease-internet-v1"}
        )
        self.client.health.return_value["internet"] = "any"

    async def test_internet_follows_public_task_when_offered(self):
        evidence = self.root / "trial/sandfleet-environment.json"
        cases = [
            (False, {}, False),
            (True, {}, True),
            (True, {"internet": False}, False),
            (True, {"internet": True}, True),
        ]
        for offered, kwargs, expected in cases:
            with self.subTest(offered=offered, **kwargs):
                if offered:
                    self.offer_internet()
                env = self.environment(**kwargs)
                await env.start()
                self.assertIs(
                    self.client.acquire.call_args.kwargs["internet"], expected
                )
                await env.stop()
                evidence.unlink()

    async def test_internet_mode_off_is_not_an_offer(self):
        self.offer_internet()
        self.client.health.return_value["internet"] = "off"
        env = self.environment()
        await env.start()
        self.assertIs(self.client.acquire.call_args.kwargs["internet"], False)
        await env.stop()
        with self.assertRaisesRegex(RuntimeError, "internet"):
            await self.environment(internet=True).start()

    async def test_internet_feature_without_mode_is_rejected(self):
        self.offer_internet()
        del self.client.health.return_value["internet"]
        with self.assertRaises(KeyError):
            await self.environment().start()
        self.client.acquire.assert_not_called()

    async def test_internet_true_needs_an_offering_controller(self):
        with self.assertRaisesRegex(RuntimeError, "internet"):
            await self.environment(internet=True).start()
        self.client.acquire.assert_not_called()

    def test_internet_and_timeout_kwargs_are_validated(self):
        for kwargs in (
            {"internet": "false"},
            {"command_timeout_sec": 0},
            {"command_timeout_sec": math.inf},
            {"command_timeout_sec": math.nan},
            {"command_timeout_sec": -5},
            {"command_timeout_sec": True},
            {"command_timeout_sec": False},
            {"command_timeout_sec": "1800"},
            {"command_timeout_sec": [1800]},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self.environment(**kwargs)

    async def test_command_timeout_override(self):
        env = self.environment(command_timeout_sec=1800)
        await env.start()
        self.assertEqual(self.lease.started["command_timeout"], 1800)
        await env.stop()

    async def test_fractional_command_timeout_rounds_up(self):
        env = self.environment(command_timeout_sec=90.5)
        await env.start()
        self.assertEqual(self.lease.started["command_timeout"], 91)
        await env.stop()

    async def test_missing_protocol_rejected_before_queue(self):
        self.client.health.return_value["features"] = []
        with self.assertRaisesRegex(RuntimeError, "protocol"):
            await self.environment().start()
        self.client.acquire.assert_not_called()

    async def test_missing_health_fields_are_protocol_errors(self):
        for key in ("ok", "features", "max_storage_mb"):
            with self.subTest(key=key), patch.dict(self.client.health.return_value):
                del self.client.health.return_value[key]
                with self.assertRaisesRegex(KeyError, key):
                    await self.environment().start()
                self.client.acquire.assert_not_called()

    async def test_storage_too_large_rejected_before_queue(self):
        self.client.health.return_value["max_storage_mb"] = 0
        with self.assertRaisesRegex(ValueError, "storage"):
            await self.environment().start()
        self.client.acquire.assert_not_called()

    async def test_mismatched_storage_closes_lease(self):
        original = self.lease.start

        def incorrect(*args, **kwargs):
            status = original(*args, **kwargs)
            status["storage"]["storage_mb"] = 2048
            return status

        self.lease.start = incorrect
        with self.assertRaisesRegex(RuntimeError, "disk volume"):
            await self.environment().start()
        self.assertTrue(self.lease.closed)

    async def test_chunked_binary_and_empty_file_roundtrip(self):
        env = self.environment()
        await env.start()
        env._checked = lambda command: ""
        for content in (b"", os.urandom(CHUNK_BYTES * 5 + 17)):
            source = self.root / "source"
            source.write_bytes(content)
            await env.upload_file(source, "/binary")
            output = self.root / "download"
            await env.download_file("/binary", output)
            self.assertEqual(output.read_bytes(), content)
        await env.stop()

    async def test_cancellation_interrupts_and_drains_before_release(self):
        env = self.environment()
        await env.start()
        task = asyncio.create_task(env.exec("WAIT_FOR_ABORT", timeout_sec=3600))
        self.assertTrue(await asyncio.to_thread(self.lease.entered.wait, 2))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertTrue(self.lease.interrupted.is_set())
        self.assertTrue(self.lease.closed)
        self.assertIsNone(env._lease)

    async def test_command_cancellation_preserves_verifier_files_and_lease(self):
        env = self.environment()
        await env.start()
        self.lease.files["/answer"] = b"original task work"
        task = asyncio.create_task(env.exec("WAIT_FOR_CANCEL", timeout_sec=3600))
        self.assertTrue(await asyncio.to_thread(self.lease.entered.wait, 2))
        command_id = self.lease.command_ids[-1]
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        self.assertEqual(self.lease.cancelled_commands, [command_id])
        self.assertIs(env._lease, self.lease)
        self.assertFalse(self.lease.closed)
        env._check_cancelled()
        output = self.root / "answer"
        await env.download_file("/answer", output)
        self.assertEqual(output.read_bytes(), b"original task work")
        await env.exec("next command")
        self.assertNotEqual(command_id, self.lease.command_ids[-1])
        await env.stop()

    async def test_failed_command_interrupt_is_visible_and_aborts_lease(self):
        env = self.environment()
        await env.start()
        task = asyncio.create_task(env.exec("WAIT_FOR_ABORT", timeout_sec=3600))
        self.assertTrue(await asyncio.to_thread(self.lease.entered.wait, 2))
        with patch.object(
            self.lease, "cancel_command", side_effect=RuntimeError("kill failed")
        ):
            task.cancel()
            with self.assertRaises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(task, 2)
        self.assertTrue(
            any("kill failed" in note for note in caught.exception.__notes__)
        )
        self.assertTrue(self.lease.closed)
        self.assertIsNone(env._lease)
        with self.assertRaises(InterruptedError):
            env._check_cancelled()

    async def test_failed_cancel_reply_aborts_instead_of_preserving_lease(self):
        env = self.environment()
        await env.start()
        task = asyncio.create_task(env.exec("WAIT_FOR_ABORT", timeout_sec=3600))
        self.assertTrue(await asyncio.to_thread(self.lease.entered.wait, 2))
        with patch.object(
            self.lease,
            "cancel_command",
            return_value={"state": "running", "interrupted": False, "cancelled": True},
        ):
            task.cancel()
            with self.assertRaises(asyncio.CancelledError) as caught:
                await asyncio.wait_for(task, 2)
        self.assertTrue(
            any("not interrupted" in note for note in caught.exception.__notes__)
        )
        self.assertTrue(self.lease.closed)
        self.assertIsNone(env._lease)

    async def test_repeated_cancel_drains_one_interrupt_and_preserves_lease(self):
        env = self.environment()
        await env.start()
        entered = threading.Event()
        release = threading.Event()
        original = self.lease.cancel_command

        def slow_cancel(command_id):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test did not release cancellation")
            return original(command_id)

        with patch.object(self.lease, "cancel_command", side_effect=slow_cancel):
            task = asyncio.create_task(env.exec("WAIT_FOR_CANCEL", timeout_sec=3600))
            try:
                self.assertTrue(await asyncio.to_thread(self.lease.entered.wait, 2))
                task.cancel()
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task.cancel()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        self.assertEqual(len(self.lease.cancelled_commands), 1)
        self.assertIs(env._lease, self.lease)
        self.assertFalse(self.lease.closed)
        await env.exec("verifier command")
        await env.stop()

    async def test_cancelled_before_start_never_acquires(self):
        env = self.environment()
        env._cancel.set()
        with self.assertRaises(InterruptedError):
            await env.start()
        self.client.acquire.assert_not_called()

    async def test_cancelled_start_is_not_restarted(self):
        env = self.environment()
        entered = threading.Event()

        def acquire(**kwargs):
            entered.set()
            if not kwargs["cancel_event"].wait(2):
                raise RuntimeError("test acquisition was not cancelled")
            raise InterruptedError("Lease acquisition cancelled")

        self.client.acquire.side_effect = acquire
        task = asyncio.create_task(env.start())
        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertIsNone(env._lease)
        with self.assertRaises(InterruptedError):
            await env.start()
        self.assertEqual(self.client.acquire.call_count, 1)

    async def test_failed_start_cleanup_retains_lease(self):
        env = self.environment()
        entered = threading.Event()

        def start(*args, **kwargs):
            entered.set()
            if not env._cancel.wait(2):
                raise RuntimeError("test start was not cancelled")
            raise InterruptedError("start cancelled")

        with (
            patch.object(self.lease, "start", side_effect=start),
            patch.object(self.lease, "close", side_effect=RuntimeError("busy")),
        ):
            task = asyncio.create_task(env.start())
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertIs(env._lease, self.lease)
            with self.assertRaisesRegex(RuntimeError, "already holds"):
                await env.start()
        await env.stop()

    async def test_failed_cleanup_retains_lease_for_retry(self):
        env = self.environment()
        await env.start()
        with patch.object(self.lease, "close", side_effect=RuntimeError("busy")):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                await env.stop()
        self.assertIs(env._lease, self.lease)
        await env.stop()
        self.assertIsNone(env._lease)

    async def test_no_resource_override(self):
        for name in ("cpus", "memory_mb", "storage_mb", "gpus"):
            with self.subTest(resource=name):
                with self.assertRaisesRegex(ValueError, "overrides"):
                    self.environment(**{f"override_{name}": 2})
        with self.assertRaisesRegex(ValueError, "overrides"):
            self.environment(override_tpu=TpuSpec(type="v6e", topology="2x2"))

    async def test_resource_modes_must_guarantee_exact_resources(self):
        self.environment(cpu_enforcement_policy=ResourceMode.GUARANTEE)
        for mode in ("limit", "request", "ignore"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "exact"):
                    self.environment(memory_enforcement_policy=ResourceMode(mode))

    async def test_omitted_resources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Positive integer"):
            SandfleetEnvironment(
                environment_dir=self.context,
                environment_name="fixture",
                session_id="unique",
                trial_paths=TrialPaths(self.root / "trial"),
                task_env_config=EnvironmentConfig(cpus=1, memory_mb=1024),
            )

    async def test_command_timeout_covers_steps_and_unset_agent_timeout(self):
        (self.context.parent / "task.toml").write_text(
            "[verifier]\ntimeout_sec=60\n"
            '[[steps]]\nname="a"\n[steps.agent]\ntimeout_sec=7200\n'
        )
        await self.environment().start()
        self.assertEqual(self.lease.started["command_timeout"], 7200)

    async def test_no_network_policy_runs_on_an_offline_lease(self):
        # Offline leases have no network at all (Sandfleet >= 0.9), so Harbor's
        # no-network tasks (e.g. allow_internet = false) run, never with internet.
        self.offer_internet()
        offline = NetworkPolicy(network_mode="no-network")
        env = self.environment(network_policy=offline)
        self.assertTrue(env.capabilities.disable_internet)
        await env.start()
        self.assertIs(self.client.acquire.call_args.kwargs["internet"], False)
        await env.stop()
        self.client.acquire.reset_mock()
        with self.assertRaisesRegex(ValueError, "no-network"):
            await self.environment(network_policy=offline, internet=True).start()
        self.client.acquire.assert_not_called()
        # The policy cannot differ between phases (one lease, fixed network).
        with self.assertRaisesRegex(ValueError, "between phases"):
            await self.environment(phase_network_policies=[offline]).start()
        self.client.acquire.assert_not_called()

    async def test_no_network_task_needs_a_worker_reporting_an_offline_sandbox(self):
        offline = NetworkPolicy(network_mode="no-network")
        for network in (
            {"mode": "loopback", "internet": True, "helper_pid": None},
            {"mode": "slirp4netns", "internet": False, "helper_pid": 7},
            {},
        ):
            with self.subTest(network=network):
                self.lease.closed, self.lease.network = False, network
                with self.assertRaisesRegex(
                    RuntimeError, "not report the sandbox as offline"
                ):
                    await self.environment(network_policy=offline).start()
                self.assertTrue(self.lease.closed)

    async def test_allowlist_network_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "allowlist"):
            self.environment(
                network_policy=NetworkPolicy(
                    network_mode="allowlist", allowed_hosts=["isolation-test.invalid"]
                )
            )

    async def test_positional_resource_override_cannot_bypass_validation(self):
        config = EnvironmentConfig(cpus=1, memory_mb=1024, storage_mb=1024)
        with self.assertRaisesRegex(ValueError, "overrides"):
            SandfleetEnvironment(
                self.context,
                "fixture",
                "unique",
                TrialPaths(self.root / "trial"),
                config,
                None,
                2,
            )
        self.assertEqual(config.cpus, 1)
        self.client.acquire.assert_not_called()

    async def test_kill_sandbox_processes_keeps_the_lease(self):
        env = self.environment()
        await env.start()
        self.assertEqual(await env.kill_sandbox_processes(), 2)
        self.assertEqual(self.lease.commands[-1], ("KILL_PROCESSES", None))
        self.assertFalse(self.lease.closed)
        result = await env.exec("echo hello")
        self.assertEqual(result.stdout, "hello")
        await env.stop()

    async def test_failed_process_kill_is_raised(self):
        env = self.environment()
        await env.start()
        failure = RuntimeError("Could not kill the sandbox's processes: SURVIVORS")
        with patch.object(self.lease, "kill_processes", side_effect=failure):
            with self.assertRaisesRegex(RuntimeError, "SURVIVORS"):
                await env.kill_sandbox_processes()
        await env.stop()

    def declare_verifier_mounts(self, mounts_toml):
        (self.context.parent / "task.toml").write_text(
            "[metadata]\nverifier_mounts = " + mounts_toml + "\n"
            "[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\n"
        )

    def offer_verifier_mounts(self, roots=("/models",)):
        self.client.health.return_value["features"] = sorted(
            {*REQUIRED_FEATURES, VERIFIER_MOUNTS_FEATURE}
        )
        self.client.health.return_value["readonly_mount_roots"] = list(roots)
        self.lease.mount_roots = list(roots)

    MODEL_MOUNT = '[{ source = "/models/example", target = "/verifier/models/example" }]'

    async def test_verifier_mounts_are_added_only_on_request(self):
        self.declare_verifier_mounts(self.MODEL_MOUNT)
        self.offer_verifier_mounts()
        env = self.environment()
        declared = [{"source": "/models/example", "target": "/verifier/models/example"}]
        self.assertEqual(env.verifier_mounts, declared)
        await env.start()
        self.assertEqual(self.lease.mount_requests, [])  # never during the agent phase
        result = await env.add_verifier_mounts()
        self.assertEqual(self.lease.mount_requests, [declared])
        self.assertEqual(result["stopped_processes"], 3)
        self.assertEqual(await env.add_verifier_mounts(), result)
        self.assertEqual(len(self.lease.mount_requests), 1)
        self.assertFalse(self.lease.closed)
        await env.stop()

    async def test_tasks_without_verifier_mounts_never_restart(self):
        env = self.environment()
        await env.start()
        self.assertEqual(env.verifier_mounts, [])
        self.assertIsNone(await env.add_verifier_mounts())
        self.assertEqual(self.lease.mount_requests, [])
        await env.stop()

    async def test_malformed_verifier_mount_declarations_are_rejected(self):
        for declaration, message in (
            ('[{ source = "/models/q", target = "/etc/q" }]', "reserved"),
            ('[{ source = "/models/q", target = "/logs" }]', "cover the writable"),
            ('[{ source = "/models/q", target = "/tests/m" }]', "Harbor writes"),
            ('[{ source = "models/q", target = "/v/q" }]', "absolute"),
            ('[{ source = "/models/q" }]', "source and target"),
            ('"/models/q:/v/q"', "non-empty list"),
        ):
            with self.subTest(declaration=declaration):
                self.declare_verifier_mounts(declaration)
                with self.assertRaisesRegex(ValueError, message):
                    self.environment()

    async def test_verifier_mounts_fail_before_acquire_without_controller_support(self):
        self.declare_verifier_mounts(self.MODEL_MOUNT)
        with self.assertRaisesRegex(RuntimeError, VERIFIER_MOUNTS_FEATURE):
            await self.environment().start()
        self.offer_verifier_mounts(roots=("/other",))
        with self.assertRaisesRegex(ValueError, "not under one of"):
            await self.environment().start()
        self.client.acquire.assert_not_called()

    async def test_verifier_mounts_need_a_capable_worker(self):
        self.declare_verifier_mounts(self.MODEL_MOUNT)
        self.offer_verifier_mounts()
        self.lease.mount_roots = None
        with self.assertRaisesRegex(RuntimeError, "predates"):
            await self.environment().start()
        self.assertTrue(self.lease.closed)

    def declare_mounts(self, agent_toml, verifier_toml):
        (self.context.parent / "task.toml").write_text(
            f"[metadata]\nagent_mounts = {agent_toml}\nverifier_mounts = {verifier_toml}\n"
            "[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\n"
        )

    async def test_agent_mounts_start_with_the_sandbox_from_agent_roots_only(self):
        data = '[{ source = "/assets/data", target = "/assets/data" }]'
        self.declare_mounts(data, self.MODEL_MOUNT)
        self.offer_verifier_mounts()
        health = self.client.health.return_value
        health["features"] = sorted({*health["features"], AGENT_MOUNTS_FEATURE})
        # A verifier-only root does not allow an agent mount.
        health["agent_readonly_mount_roots"] = ["/models"]
        with self.assertRaisesRegex(ValueError, "agent_readonly_mount_roots"):
            await self.environment().start()
        self.client.acquire.assert_not_called()
        health["agent_readonly_mount_roots"] = ["/assets"]
        env = self.environment()
        await env.start()
        declared = [{"source": "/assets/data", "target": "/assets/data"}]
        self.assertEqual(self.lease.started["readonly_mounts"], declared)
        self.assertEqual(self.lease.mount_requests, [])
        await env.add_verifier_mounts()
        self.assertEqual(len(self.lease.mount_requests), 1)
        await env.stop()

    async def test_agent_mounts_need_controller_support_and_distinct_targets(self):
        data = '[{ source = "/assets/data", target = "/assets/data" }]'
        self.declare_mounts(data, self.MODEL_MOUNT)
        self.offer_verifier_mounts()
        with self.assertRaisesRegex(RuntimeError, AGENT_MOUNTS_FEATURE):
            await self.environment().start()
        nested = '[{ source = "/models/example", target = "/assets/data/m" }]'
        self.declare_mounts(data, nested)
        with self.assertRaisesRegex(ValueError, "equal or nested"):
            self.environment()

    async def test_unreported_verifier_mount_is_an_error(self):
        self.declare_verifier_mounts(self.MODEL_MOUNT)
        self.offer_verifier_mounts()
        self.lease.reported_mounts = []
        env = self.environment()
        await env.start()
        with self.assertRaisesRegex(RuntimeError, "did not report"):
            await env.add_verifier_mounts()
        await env.stop()

    def declare_separate(self, tests_source):
        (self.context.parent / "task.toml").write_text(
            "[metadata]\n"
            'agent_mounts = [{ source = "/assets/data", target = "/assets/data" }]\n'
            'produce_mounts = [{ source = "/models/inputs", target = "/produce/inputs" }]\n'
            f'verifier_mounts = [{{ source = "{tests_source}", target = "/tests" }}, '
            '{ source = "/models/example", target = "/verifier/models/example" }]\n'
            "[agent]\ntimeout_sec=1800\n"
            '[verifier]\ntimeout_sec=3600\nenvironment_mode = "separate"\n'
        )
        manifest = json.loads(self.manifest.read_text())
        manifest["named"] = {
            "shared-verifier": {
                **next(iter(manifest["images"].values())),
                "sif_path": "/images/shared.sif",
            }
        }
        self.manifest.write_text(json.dumps(manifest))
        self.offer_verifier_mounts(roots=("/models", str(self.root)))
        health = self.client.health.return_value
        health["features"] = sorted({*health["features"], AGENT_MOUNTS_FEATURE})
        health["agent_readonly_mount_roots"] = ["/assets"]

    async def test_separate_verifier_env_uses_the_shared_image_and_its_own_mounts(self):
        tests = self.context.parent / "tests"
        tests.mkdir()
        (tests / "test.sh").write_text("exit 0\n")
        (tests / "test.sh").chmod(0o755)
        self.declare_separate(tests)
        env = SandfleetEnvironment(
            environment_dir=tests,
            environment_name="fixture",
            session_id="trial__verifier__trial",
            trial_paths=TrialPaths(self.root / "trial"),
            task_env_config=EnvironmentConfig(
                cpus=1, memory_mb=1024, storage_mb=1024, docker_image="shared-verifier"
            ),
        )
        self.assertTrue(env.is_verifier_env)
        self.assertEqual(
            env._image["sif_path"], "/images/shared.sif"
        )  # not a per-task image
        agent_evidence = self.root / "trial/sandfleet-environment.json"
        agent_evidence.parent.mkdir(parents=True)
        agent_evidence.write_text("{}")  # the agent sandbox's, in the same trial dir
        with patch.object(
            SandfleetEnvironment, "_upload_environment_dir_after_start"
        ) as upload:
            await env.start()
        upload.assert_not_called()  # /tests is the read-only tests mount
        evidence = self.root / "trial/sandfleet-verifier-environment.json"
        self.assertIn("/images/shared.sif", evidence.read_text())
        self.assertEqual(self.lease.started["readonly_mounts"], [])
        self.assertEqual(
            [[m["target"] for m in r] for r in self.lease.mount_requests],
            [["/tests", "/verifier/models/example"]],
        )
        await env.stop()

    async def test_separate_mode_agent_sandbox_gets_agent_then_producer_mounts(self):
        self.declare_separate(self.root / "tests")
        env = self.environment()
        self.assertFalse(env.is_verifier_env)
        self.assertEqual(env.verifier_mounts, [])  # the separate verifier env adds them
        await env.start()
        self.assertEqual(
            self.lease.started["readonly_mounts"],
            [{"source": "/assets/data", "target": "/assets/data"}],
        )
        self.assertEqual(self.lease.mount_requests, [])
        result = await env.add_produce_mounts()
        self.assertEqual(
            self.lease.mount_requests,
            [[{"source": "/models/inputs", "target": "/produce/inputs"}]],
        )
        self.assertEqual(await env.add_produce_mounts(), result)
        self.assertIsNone(await env.add_verifier_mounts())
        await env.stop()

    async def test_produce_mounts_and_a_tests_mount_need_a_separate_verifier(self):
        for declaration, message in (
            (
                'produce_mounts = [{ source = "/models/i", target = "/produce/inputs" }]',
                "separate",
            ),
            (
                'verifier_mounts = [{ source = "/models/t", target = "/tests" }]',
                "Harbor writes",
            ),
        ):
            with self.subTest(declaration=declaration):
                (self.context.parent / "task.toml").write_text(
                    f"[metadata]\n{declaration}\n[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\n"
                )
                with self.assertRaisesRegex(ValueError, message):
                    self.environment()

    def separate_env(self, directory, session_id):
        return SandfleetEnvironment(
            environment_dir=directory,
            environment_name="fixture",
            session_id=session_id,
            trial_paths=TrialPaths(self.root / "trial"),
            task_env_config=EnvironmentConfig(
                cpus=1, memory_mb=1024, storage_mb=1024, docker_image="shared-verifier"
            ),
        )

    async def test_the_role_comes_from_the_build_context_not_the_session_id(self):
        tests = self.context.parent / "tests"
        tests.mkdir()
        self.declare_separate(tests)
        # Harbor truncates a long verifier session id, dropping the marker: still a verifier.
        verifier = self.separate_env(tests, "a-long-trial-name-truncated__0123abcd")
        self.assertTrue(verifier.is_verifier_env)
        self.assertEqual(
            [m["target"] for m in verifier.verifier_mounts],
            ["/tests", "/verifier/models/example"],
        )
        # Harbor cannot chmod the read-only /tests mount, so a non-executable test.sh is refused.
        script = tests / "test.sh"
        script.write_text("exit 0\n")
        script.chmod(0o640)
        with self.assertRaisesRegex(ValueError, "must be executable"):
            self.separate_env(tests, "t__verifier__trial")
        script.chmod(0o750)
        # A trial name that contains the marker never makes an agent sandbox get the answers.
        with self.assertRaisesRegex(ValueError, "tests/"):
            self.separate_env(self.context, "x__verifier__y__env")
        # A shared-mode task has no verifier environment.
        (self.context.parent / "task.toml").write_text(
            "[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\n"
        )
        with self.assertRaisesRegex(ValueError, "separate verifier task"):
            self.separate_env(tests, "t__verifier__trial")

    async def test_agent_and_producer_sources_never_overlap_verifier_sources(self):
        link = self.root / "inputs-link"
        link.symlink_to("/models/example")
        for declaration in (
            'produce_mounts = [{ source = "/models/example/split", target = "/produce/inputs" }]',
            'agent_mounts = [{ source = "/models", target = "/assets/models" }]',
            f'produce_mounts = [{{ source = "{link}", target = "/produce/inputs" }}]',
        ):
            with self.subTest(declaration=declaration):
                (self.context.parent / "task.toml").write_text(
                    f"[metadata]\n{declaration}\n"
                    'verifier_mounts = [{ source = "/models/example", target = "/verifier/models/example" }]\n'
                    '[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\nenvironment_mode = "separate"\n'
                )
                with self.assertRaisesRegex(ValueError, "overlaps the verifier"):
                    self.environment()

    async def test_infrastructure_download_failures_are_recorded(self):
        from sandfleet.errors import SandboxLostError, SandfleetError

        env = self.environment()
        await env.start()
        for error, infra in (
            (SandboxLostError("worker gone"), True),
            (SandfleetError("agent error", status=500), True),
            (SandfleetError("no lease", status=404), True),
            (OSError("host disk full"), True),
            (SandfleetError("FileNotFoundError: /app/x", status=400), False),
            (ValueError("Unsafe file type/path in remote artifacts"), False),
            (RuntimeError("File transfer command failed (1)"), False),
        ):
            with self.subTest(error=error):
                env.transfer_infra_failures.clear()

                def fail(*_args, error=error):
                    raise error

                env._download_file = fail
                with self.assertRaises(type(error)):
                    await env.download_file("/app/prediction.json", self.root / "p")
                self.assertEqual(bool(env.transfer_infra_failures), infra)
        await env.stop()

    async def test_operations_require_a_started_environment(self):
        env = self.environment()
        for operation in (
            lambda: env.kill_sandbox_processes(),
            lambda: env.exec("true"),
            lambda: env.upload_file(self.root / "source", "/target"),
            lambda: env.download_file("/source", self.root / "target"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not started"):
                await operation()

    async def test_malformed_manifest_is_not_reported_as_a_missing_image(self):
        self.manifest.write_text(json.dumps({"schema_version": 1}))
        with self.assertRaisesRegex(KeyError, "images"):
            resolve_image(self.manifest, self.context)

    async def test_tampered_context_fails_closed(self):
        (self.context / "Dockerfile").write_text("FROM different\n")
        with self.assertRaisesRegex(ValueError, "No exact"):
            self.environment()

    async def test_non_root_image_fails_before_acquire(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["images"][context_digest(self.context)]["user"] = "taskuser"
        self.manifest.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "root-user"):
            self.environment()
        self.client.acquire.assert_not_called()

    async def test_exec_truncation_preserves_output_and_exit_status(self):
        env = self.environment()
        await env.start()
        self.lease.run_command = lambda *_a, **_kw: SimpleNamespace(
            output_truncated=True,
            stdout="retained output",
            stderr="warning",
            exit_code=7,
        )
        result = await env.exec("large output")
        self.assertEqual(result.stdout, "retained output")
        self.assertEqual(result.return_code, 7)
        self.assertIn("warning", result.stderr)
        self.assertIn("command output truncated", result.stderr)

    async def test_worker_deadline_reports_timeout_exit_status(self):
        env = self.environment()
        await env.start()
        self.lease.run_command = lambda *_a, **_kw: SimpleNamespace(
            output_truncated=False,
            stdout="",
            stderr="Command timed out after 5s.",
            exit_code=143,
            timed_out=True,
        )
        result = await env.exec("sleep 60", timeout_sec=5)
        self.assertEqual(result.return_code, 124)

    async def test_file_transfer_truncation_is_an_error(self):
        env = self.environment()
        await env.start()
        self.lease.run_command = lambda *_a, **_kw: SimpleNamespace(
            output_truncated=True
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            env._checked("file inventory")

    async def test_directory_download_preserves_binary_and_rejects_links(self):
        env = self.environment()
        await env.start()
        source = self.root / "remote"
        source.mkdir()
        (source / "binary").write_bytes(b"\x00\xff")
        self.lease.files[str(source / "binary")] = b"\x00\xff"
        # macOS BSD find lacks -printf; inject the same GNU inventory encoding.
        env._checked = lambda _command: base64.b64encode(
            b"d\0\000755\0f\0binary\0" + b"644\0"
        ).decode()
        await env.download_dir(str(source), self.root / "result")
        self.assertEqual((self.root / "result/binary").read_bytes(), b"\x00\xff")
        env._checked = lambda _command: base64.b64encode(
            b"l\0escape\0" + b"777\0"
        ).decode()
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            await env.download_dir(str(source), self.root / "result")

    # ---------- GPU sandboxes ----------

    GPU_A = "GPU-00000000-0000-0000-0000-00000000000a"
    GPU_B = "GPU-00000000-0000-0000-0000-00000000000b"

    def offer_gpus(self, enabled=True, types=None):
        self.client.gpu_capabilities.return_value = {
            "enabled": enabled,
            "types": (
                types
                if types is not None
                else {"gb300_288gb": {"memory_mb": 284208, "max_per_sandbox": 4}}
                if enabled
                else {}
            ),
        }
        self.lease.gpu_uuids = [self.GPU_A]
        self.lease.gpu_type = "gb300"  # the worker reports the Slurm GRES type
        self.lease.pool = "sfres-c1-m1024-gpu1-gb300_288gb"

    def gpu_environment(self, gpus=1, gpu_types=None, **kwargs):
        return SandfleetEnvironment(
            environment_dir=self.context,
            environment_name="fixture",
            session_id="unique",
            trial_paths=TrialPaths(self.root / "trial"),
            task_env_config=EnvironmentConfig(
                cpus=1, memory_mb=1024, storage_mb=1024, gpus=gpus, gpu_types=gpu_types
            ),
            **kwargs,
        )

    def declare_metadata(self, metadata_toml):
        (self.context.parent / "task.toml").write_text(
            "[metadata]\n" + metadata_toml + "\n"
            "[agent]\ntimeout_sec=1800\n[verifier]\ntimeout_sec=3600\n"
        )

    async def test_gpu_task_leases_a_dedicated_gpu_and_records_it(self):
        self.offer_gpus()
        env = self.gpu_environment(gpu_types=["GB300"])
        self.assertTrue(env.capabilities.gpus)
        await env.start()
        kwargs = self.client.acquire.call_args.kwargs
        self.assertEqual((kwargs["gpu"], kwargs["gpu_type"]), (1, ["gb300_288gb"]))
        self.assertEqual((kwargs["cpus"], kwargs["memory_mb"]), (1, 1024))
        smi = [c for c, _t in self.lease.commands if "nvidia-smi" in c]
        self.assertEqual(len(smi), 1)  # checked inside the sandbox before the agent
        evidence = json.loads(
            (self.root / "trial/sandfleet-environment.json").read_text()
        )
        self.assertEqual(evidence["gpus"]["uuids"], [self.GPU_A])
        self.assertEqual(evidence["gpus"]["type"], "gb300_288gb")
        self.assertEqual(evidence["gpus"]["slurm_type"], "gb300")
        await env.stop()

    async def test_cpu_task_never_asks_for_gpus(self):
        env = self.environment()
        await env.start()
        self.client.gpu_capabilities.assert_not_called()
        self.assertNotIn("gpu", self.client.acquire.call_args.kwargs)
        self.assertFalse(any("nvidia-smi" in c for c, _t in self.lease.commands))
        evidence = json.loads(
            (self.root / "trial/sandfleet-environment.json").read_text()
        )
        self.assertNotIn("gpus", evidence)
        await env.stop()

    async def test_controller_without_gpu_pools_fails_before_acquire(self):
        self.offer_gpus(enabled=False)
        with self.assertRaisesRegex(RuntimeError, "no GPU sandboxes configured"):
            await self.gpu_environment().start()
        self.client.acquire.assert_not_called()

    async def test_gpu_count_above_the_controller_limit_fails_before_acquire(self):
        self.offer_gpus(
            types={"gb300_288gb": {"memory_mb": 284208, "max_per_sandbox": 1}}
        )
        with self.assertRaisesRegex(RuntimeError, "at most"):
            await self.gpu_environment(gpus=2).start()
        self.client.acquire.assert_not_called()

    async def test_unavailable_gpu_type_fails_before_acquire(self):
        self.offer_gpus()
        with self.assertRaisesRegex(RuntimeError, "None of the task's gpu_types"):
            await self.gpu_environment(gpu_types=["H100"]).start()
        self.client.acquire.assert_not_called()

    def test_gpu_type_resolution(self):
        enabled = ["gb300_288gb", "h100_80gb", "h100_94gb"]
        self.assertEqual(resolve_gpu_types(None, enabled), sorted(enabled))
        self.assertEqual(resolve_gpu_types(["GB300"], enabled), ["gb300_288gb"])
        self.assertEqual(
            resolve_gpu_types(["h100", "gb300_288gb", "H100_80GB"], enabled),
            ["h100_80gb", "h100_94gb", "gb300_288gb"],
        )
        self.assertEqual(
            resolve_gpu_types(["a100_80gb", "gb300"], enabled), ["gb300_288gb"]
        )
        for requested in (["a100_80gb"], ["gb3"], ["h100_8"]):
            with self.subTest(requested=requested), self.assertRaises(RuntimeError):
                resolve_gpu_types(requested, enabled)
        with self.assertRaises(RuntimeError):
            resolve_gpu_types(None, [])
        with self.assertRaises(ValueError):
            resolve_gpu_types([""], enabled)

    async def test_lease_with_the_wrong_gpus_is_released(self):
        cases = {
            "two devices": lambda: setattr(
                self.lease, "gpu_uuids", [self.GPU_A, self.GPU_B]
            ),
            "other type": lambda: setattr(
                self.lease, "pool", "sfres-c1-m1024-gpu1-h100_80gb"
            ),
            "CPU pool": lambda: setattr(self.lease, "pool", "sfres-c1-m1024-gpu-none"),
        }
        for name, mutate in cases.items():
            with self.subTest(name):
                self.lease.closed = False
                self.offer_gpus()
                mutate()
                with self.assertRaisesRegex(RuntimeError, "GPU"):
                    await self.gpu_environment().start()
                self.assertTrue(self.lease.closed)

    async def test_sandbox_must_see_exactly_its_gpu(self):
        cases = {
            "another GPU": dict(visible_uuids=[self.GPU_B]),
            "an extra GPU": dict(visible_uuids=[self.GPU_A, self.GPU_B]),
            "no GPU": dict(visible_uuids=[]),
            "nvidia-smi fails": dict(smi_exit=9),
        }
        for name, attrs in cases.items():
            with self.subTest(name):
                self.lease.closed = False
                self.lease.visible_uuids, self.lease.smi_exit = None, 0
                self.offer_gpus()
                for key, value in attrs.items():
                    setattr(self.lease, key, value)
                with self.assertRaisesRegex(RuntimeError, "exactly its leased GPU"):
                    await self.gpu_environment().start()
                self.assertTrue(self.lease.closed)
                self.assertFalse(
                    (self.root / "trial/sandfleet-environment.json").exists()
                )

    async def test_sandbox_must_not_see_imex_channels(self):
        self.offer_gpus()
        self.lease.imex_channels = ["channel0"]
        with self.assertRaisesRegex(RuntimeError, "IMEX channels"):
            await self.gpu_environment().start()
        self.assertTrue(self.lease.closed)

    async def test_verifier_mount_restart_keeps_the_same_gpu(self):
        self.declare_verifier_mounts(self.MODEL_MOUNT)
        self.offer_verifier_mounts()
        self.offer_gpus()
        env = self.gpu_environment()
        await env.start()
        result = await env.add_verifier_mounts()
        self.assertEqual(result["stopped_processes"], 3)
        self.assertEqual(await env.kill_sandbox_processes(), 2)
        await env.stop()

        self.lease.closed = False
        (self.root / "trial/sandfleet-environment.json").unlink()
        env = self.gpu_environment()
        await env.start()
        self.lease.gpu_uuids = [self.GPU_B]  # the restart came back on another device
        with self.assertRaisesRegex(RuntimeError, "changed its GPUs"):
            await env.add_verifier_mounts()
        await env.stop()

    async def test_exclusive_gpu_declaration(self):
        self.offer_gpus()
        self.declare_metadata("exclusive_gpu = true")
        env = self.gpu_environment()
        self.assertTrue(env.supports_exclusive_gpus)
        await env.start()
        await env.stop()
        with self.assertRaisesRegex(ValueError, "exclusive_gpu needs"):
            self.environment()
        self.declare_metadata('exclusive_gpu = "yes"')
        with self.assertRaisesRegex(ValueError, "true or false"):
            self.gpu_environment()

    def test_gpu_fields_are_validated(self):
        with self.assertRaisesRegex(ValueError, "gpu_types needs"):
            self.gpu_environment(gpus=0, gpu_types=["gb300_288gb"])
        self.gpu_environment(gpus=0)  # an explicit zero is a CPU task

    async def test_manifest_symlink_rejected(self):
        (self.context / "link").symlink_to(self.context / "Dockerfile")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            resolve_image(self.manifest, self.context)


if __name__ == "__main__":
    unittest.main()
