"""Bounded, standalone Apptainer storage experiment; not a Sandfleet backend.

Run inside an isolated Slurm step with a Python-capable SIF. No scheduler calls,
controller access, tokens, or changes to existing workers are made here.
"""

import argparse
import errno
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

MIB = 1024 * 1024
PATHS = ("/", "/tmp", "/var/tmp", "/root")
MARKER = "sandfleet-storage-probe-marker"


def run(argv, **kwargs):
    result = subprocess.run(argv, text=True, capture_output=True, timeout=180, **kwargs)
    if result.returncode:
        raise RuntimeError(f"{argv!r}: {result.returncode}\n{result.stderr}")
    return result.stdout


def mount_for(path, mountinfo):
    candidates = []
    for line in mountinfo.splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        mount = fields[4]
        if path == mount or path.startswith(mount.rstrip("/") + "/"):
            candidates.append((len(mount), fields[0], mount, right.split()[0]))
    _, identity, mount, fs_type = max(candidates)
    return {"id": identity, "mount": mount, "type": fs_type}


def inside_snapshot():
    mountinfo = Path("/proc/self/mountinfo").read_text()
    result = {}
    for name in PATHS:
        stat = os.statvfs(name)
        result[name] = {
            **mount_for(name, mountinfo),
            "fsid": stat.f_fsid,
            "total_bytes": stat.f_blocks * stat.f_frsize,
            "available_bytes": stat.f_bavail * stat.f_frsize,
            "marker": (Path(name) / MARKER).exists(),
        }
    return {"paths": result, "mountinfo": mountinfo, "home": os.getenv("HOME")}


def inside_write(count_mib):
    block = b"s" * MIB
    with Path("/sandfleet-probe.bin").open("xb", buffering=0) as target:
        for _ in range(count_mib):
            if target.write(block) != len(block):
                raise RuntimeError("short write before the deliberate disk-full test")
        os.fsync(target.fileno())
    for name in PATHS:
        (Path(name) / MARKER).write_text("trial-a\n")
    return {"written_bytes": count_mib * MIB, **inside_snapshot()}


def inside_fill():
    count = 0
    durable = 0
    block = b"f" * MIB
    try:
        with Path("/tmp/sandfleet-fill.bin").open("xb", buffering=0) as target:
            for index in range(4096):
                count += target.write(block)
                if (index + 1) % 8 == 0:
                    os.fsync(target.fileno())
                    durable = count
            os.fsync(target.fileno())
    except OSError as error:
        if error.errno != errno.ENOSPC:
            raise
        return {
            "enospc": True,
            "acknowledged_bytes": count,
            "durable_before_full_bytes": durable,
            "logical_file_bytes": Path("/tmp/sandfleet-fill.bin").stat().st_size,
            **inside_snapshot(),
        }
    raise RuntimeError("bounded fill did not reach ENOSPC")


def inside_verify():
    count = 0
    block = b"s" * MIB
    with Path("/sandfleet-probe.bin").open("rb", buffering=0) as source:
        while chunk := source.read(MIB):
            if chunk != block:
                raise RuntimeError(f"readback mismatch after {count} bytes")
            count += len(chunk)
    if count != 1200 * MIB:
        raise RuntimeError(f"wrong durable file size: {count}")
    return {"verified_bytes": count}


def cgroups():
    relative = next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    root = Path("/sys/fs/cgroup")
    cursor = root / relative.lstrip("/")
    levels = []
    while cursor.is_relative_to(root):
        entry = {"path": str(cursor)}
        for key in (
            "memory.max",
            "memory.current",
            "memory.peak",
            "memory.events",
            "memory.swap.max",
            "memory.swap.current",
            "cpuset.cpus.effective",
        ):
            source = cursor / key
            if source.exists():
                entry[key] = source.read_text().strip()
        levels.append(entry)
        if cursor == root:
            break
        cursor = cursor.parent
    return {"affinity": sorted(os.sched_getaffinity(0)), "levels": levels}


def finite_memory(snapshot):
    return min(
        int(level["memory.max"])
        for level in snapshot["levels"]
        if level.get("memory.max", "max") != "max"
    )


def oom_counts(snapshot):
    # Do not attribute unrelated OOMs in another user's job to this probe.
    last_limited = max(
        index
        for index, level in enumerate(snapshot["levels"])
        if level.get("memory.max", "max") != "max"
    )
    return {
        level["path"]: {
            key: int(value)
            for key, value in (
                line.split() for line in level["memory.events"].splitlines()
            )
            if key in ("oom", "oom_kill", "oom_group_kill")
        }
        for level in snapshot["levels"][: last_limited + 1]
        if "memory.events" in level
    }


def check_paths(snapshot):
    paths = snapshot["paths"]
    root = paths["/"]
    if "overlay" not in root["type"]:
        raise RuntimeError(f"root is not an overlay: {root}")
    if any(value["id"] != root["id"] for value in paths.values()):
        raise RuntimeError(f"writable paths escape the overlay: {paths}")


def instance_list(env):
    return json.loads(run(["apptainer", "instance", "list", "--json"], env=env))


def cleanup_instance(name, overlay, env, evidence):
    listed = instance_list(env)
    if any(item["instance"] == name for item in listed["instances"]):
        run(["apptainer", "instance", "stop", name], env=env)
    remaining = instance_list(env)
    evidence["instances_after_stop"] = remaining
    if any(item["instance"] == name for item in remaining["instances"]):
        raise RuntimeError("instance still listed; retaining overlay")
    overlay.unlink()
    evidence["overlay_removed"] = not overlay.exists()


def trial(image, directory, label, env, report):
    overlay = directory / f"{label}.img"
    name = f"sf-disk-{os.getpid()}-{label}"
    run(
        [
            "apptainer",
            "overlay",
            "create",
            "--fakeroot",
            "--size",
            "2048",
            str(overlay),
        ],
        env=env,
    )
    stat = overlay.stat()
    evidence = {
        "instance": name,
        "overlay_bytes": stat.st_size,
        "allocated_bytes": stat.st_blocks * 512,
    }
    report[label] = evidence
    argv = [
        "apptainer",
        "instance",
        "start",
        "--fakeroot",
        "--containall",
        "--cleanenv",
        "--no-mount",
        "tmp,home,cwd,hostfs,bind-paths",
        "--overlay",
        str(overlay),
        "--bind",
        f"{Path(__file__).resolve()}:/opt/storage_probe.py:ro",
        str(image),
        name,
    ]
    evidence["start_argv"] = argv
    try:
        run(argv, env=env, cwd="/")
        evidence["instance_state"] = instance_list(env)
        cmd = [
            "apptainer",
            "exec",
            f"instance://{name}",
            "python",
            "/opt/storage_probe.py",
            "--inside",
        ]
        snapshot = json.loads(run(cmd + ["snapshot"], env=env))
        evidence["initial"] = snapshot
        check_paths(snapshot)
        if any(value["marker"] for value in snapshot["paths"].values()):
            raise RuntimeError("new trial sees another trial's marker")
        if label == "a":
            evidence["write"] = json.loads(run(cmd + ["write"], env=env))
            evidence["readback_before_full"] = json.loads(
                run(cmd + ["verify"], env=env)
            )
            evidence["fill"] = json.loads(run(cmd + ["fill"], env=env))
            evidence["readback_after_full"] = json.loads(run(cmd + ["verify"], env=env))
            evidence["after_full"] = json.loads(run(cmd + ["snapshot"], env=env))
            if not all(
                value["marker"] for value in evidence["after_full"]["paths"].values()
            ):
                raise RuntimeError("markers did not persist across commands")
        evidence["success"] = True
    except BaseException as error:
        try:
            cleanup_instance(name, overlay, env, evidence)
        except Exception as cleanup_error:
            error.add_note(f"cleanup failed; overlay retained: {cleanup_error!r}")
        raise
    else:
        cleanup_instance(name, overlay, env, evidence)


def host_probe(args, report):
    report["cgroup_before"] = cgroups()
    if finite_memory(report["cgroup_before"]) > 1024 * MIB:
        raise RuntimeError("probe requires a finite RAM limit of at most 1024 MiB")
    parent = Path(args.scratch).resolve()
    report["scratch_mount"] = run(["findmnt", "-T", str(parent), "-J"])
    if "tmpfs" in report["scratch_mount"]:
        raise RuntimeError("scratch is RAM-backed")
    if os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize < 6 * 1024 * MIB:
        raise RuntimeError("less than 6 GiB scratch headroom")
    directory = Path(tempfile.mkdtemp(prefix="sf-disk-", dir=parent))
    report["scratch_directory"] = str(directory)
    # Explicit allowlist: inherited binds/options could invalidate this experiment.
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "USER", "LOGNAME")
        if key in os.environ
    }
    env["APPTAINER_TMPDIR"] = str(directory)
    env["APPTAINER_CACHEDIR"] = str(directory / "cache")
    env["APPTAINER_CONFIGDIR"] = str(directory / "config")
    env["APPTAINER_PWD"] = "/"
    report["apptainer_version"] = run(["apptainer", "--version"], env=env).strip()
    report["worker_pid"] = os.getpid()
    for label in ("a", "b"):
        trial(Path(args.image).resolve(), directory, label, env, report)
        report[f"cgroup_after_{label}"] = cgroups()
    report["cgroup_after"] = cgroups()
    if oom_counts(report["cgroup_before"]) != oom_counts(report["cgroup_after"]):
        raise RuntimeError("cgroup OOM counters increased")
    report["success"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", choices=("snapshot", "write", "fill", "verify"))
    parser.add_argument("--image")
    parser.add_argument("--scratch", default="/tmp")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.inside:
        operations = {
            "snapshot": inside_snapshot,
            "write": lambda: inside_write(1200),
            "fill": inside_fill,
            "verify": inside_verify,
        }
        print(json.dumps(operations[args.inside]()))
        return
    if not args.image or not args.output:
        parser.error("--image and --output are required on the host")
    report = {
        "success": False,
        "started_unix": time.time(),
        "job": os.getenv("SLURM_JOB_ID"),
    }
    with Path(args.output).open("x") as output:
        try:
            host_probe(args, report)
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            report["finished_unix"] = time.time()
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
