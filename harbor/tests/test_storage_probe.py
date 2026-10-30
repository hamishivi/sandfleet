import importlib.util
import errno
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "storage_probe", Path(__file__).parents[1] / "storage_probe.py"
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class StorageProbeTests(unittest.TestCase):
    def test_mount_lookup_uses_longest_component_prefix(self):
        mounts = (
            "1 0 0:1 / / rw - overlay overlay rw\n"
            "2 1 0:2 / /tmp rw - tmpfs tmpfs rw\n"
        )
        self.assertEqual(probe.mount_for("/tmp/file", mounts)["id"], "2")
        self.assertEqual(probe.mount_for("/tmpother", mounts)["id"], "1")

    def test_tmpfs_hidden_under_writable_root_rejected(self):
        paths = {name: {"id": "1", "type": "overlay"} for name in probe.PATHS}
        paths["/tmp"] = {"id": "2", "type": "tmpfs"}
        with self.assertRaisesRegex(RuntimeError, "escape"):
            probe.check_paths({"paths": paths})

    def test_all_critical_paths_share_disk_overlay(self):
        paths = {name: {"id": "1", "type": "fuse-overlayfs"} for name in probe.PATHS}
        probe.check_paths({"paths": paths})

    def test_ram_backed_root_rejected(self):
        paths = {name: {"id": "1", "type": "tmpfs"} for name in probe.PATHS}
        with self.assertRaisesRegex(RuntimeError, "not an overlay"):
            probe.check_paths({"paths": paths})

    def test_strongest_finite_memory_ancestor(self):
        self.assertEqual(
            probe.finite_memory(
                {
                    "levels": [
                        {},
                        {"memory.max": "max"},
                        {"memory.max": "1024"},
                        {"memory.max": "2048"},
                    ]
                }
            ),
            1024,
        )
        with self.assertRaises(ValueError):
            probe.finite_memory({"levels": [{"memory.max": "max"}]})

    def test_oom_counters_not_reclaim_counter(self):
        before = {
            "levels": [
                {
                    "path": "job",
                    "memory.max": "1024",
                    "memory.events": "max 10\noom 0\noom_kill 0",
                }
            ]
        }
        after = {
            "levels": [
                {
                    "path": "job",
                    "memory.max": "1024",
                    "memory.events": "max 100\noom 0\noom_kill 0",
                }
            ]
        }
        self.assertEqual(probe.oom_counts(before), probe.oom_counts(after))
        after["levels"][0]["memory.events"] += "\noom_group_kill 1"
        self.assertNotEqual(probe.oom_counts(before), probe.oom_counts(after))

    def test_unrelated_host_oom_excluded(self):
        snapshot = {
            "levels": [
                {"path": "job", "memory.max": "1024", "memory.events": "oom 0"},
                {"path": "host", "memory.max": "max", "memory.events": "oom 17"},
            ]
        }
        self.assertEqual(probe.oom_counts(snapshot), {"job": {"oom": 0}})

    def test_failed_stop_retains_overlay(self):
        overlay = unittest.mock.Mock()
        with patch.object(
            probe, "instance_list", return_value={"instances": [{"instance": "mine"}]}
        ):
            with patch.object(probe, "run", side_effect=RuntimeError("stop failed")):
                with self.assertRaisesRegex(RuntimeError, "stop failed"):
                    probe.cleanup_instance("mine", overlay, {}, {})
        overlay.unlink.assert_not_called()

    def test_no_other_instance_is_stopped(self):
        overlay = unittest.mock.Mock()
        overlay.exists.return_value = False
        with patch.object(
            probe, "instance_list", return_value={"instances": [{"instance": "other"}]}
        ):
            with patch.object(probe, "run") as run:
                probe.cleanup_instance("mine", overlay, {}, {})
        run.assert_not_called()
        overlay.unlink.assert_called_once()

    def test_still_listed_instance_retains_overlay(self):
        overlay = unittest.mock.Mock()
        with patch.object(
            probe, "instance_list", return_value={"instances": [{"instance": "mine"}]}
        ):
            with patch.object(probe, "run"):
                with self.assertRaisesRegex(RuntimeError, "retaining"):
                    probe.cleanup_instance("mine", overlay, {}, {})
        overlay.unlink.assert_not_called()

    def test_delayed_disk_full_counts_only_synced_bytes_as_durable(self):
        with patch.object(probe, "Path") as path, patch.object(
            probe, "inside_snapshot", return_value={}
        ):
            writer = path.return_value.open.return_value.__enter__.return_value
            writer.write.return_value = probe.MIB
            path.return_value.stat.return_value.st_size = 16 * probe.MIB
            with patch.object(
                probe.os, "fsync", side_effect=[None, OSError(errno.ENOSPC, "full")]
            ):
                result = probe.inside_fill()
        self.assertTrue(result["enospc"])
        self.assertEqual(result["acknowledged_bytes"], 16 * probe.MIB)
        self.assertEqual(result["durable_before_full_bytes"], 8 * probe.MIB)

    def test_unexpected_io_error_not_relabelled_as_disk_full(self):
        with patch.object(probe, "Path") as path:
            writer = path.return_value.open.return_value.__enter__.return_value
            writer.write.side_effect = OSError(errno.EIO, "io failure")
            with self.assertRaises(OSError) as caught:
                probe.inside_fill()
        self.assertEqual(caught.exception.errno, errno.EIO)

    def test_readback_rejects_wrong_data_and_truncation(self):
        for chunks, message in (([b"bad"], "mismatch"), ([b""], "wrong durable")):
            with patch.object(probe, "Path") as path:
                reader = path.return_value.open.return_value.__enter__.return_value
                reader.read.side_effect = chunks
                with self.assertRaisesRegex(RuntimeError, message):
                    probe.inside_verify()


if __name__ == "__main__":
    unittest.main()
