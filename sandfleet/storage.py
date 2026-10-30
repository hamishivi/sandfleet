"""Opt-in, bounded writable volumes owned by one prepared Slurm step.

storage_mb is the nominal ext3 image size, including filesystem metadata. The
immutable SIF is shared and not charged to this writable-volume allowance.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

MIB = 1024 * 1024
DISK_START_FLAGS = (
    "--fakeroot",
    "--containall",
    "--cleanenv",
    "--no-mount",
    "tmp,home,cwd,hostfs,bind-paths",
)


def validate_storage(storage_mb: int | None, maximum_mb: int) -> None:
    if storage_mb is None:
        return
    if type(storage_mb) is not int or not 64 <= storage_mb <= maximum_mb:
        raise ValueError(
            f"storage_mb must be an integer in [64, {maximum_mb}]; disk mode is opt-in"
        )


def image_digest(image: str, expected: str | None) -> str:
    path = Path(image)
    if not path.is_absolute() or path.suffix != ".sif" or not path.is_file():
        raise ValueError("Disk mode requires a prebuilt absolute .sif path")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("Disk mode requires the SIF image_sha256")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(MIB), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("SIF image_sha256 mismatch")
    return expected


def checked(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=300)
    if result.returncode:
        raise RuntimeError(
            f"Storage command failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def overlay_mounts(mountinfo: str, paths: list[str]) -> None:
    mounts = []
    for line in mountinfo.splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        point = re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), fields[4])
        mounts.append((point, fields[0], right.split()[0]))
    root = next(item for item in mounts if item[0] == "/")
    if "overlay" not in root[2]:
        raise RuntimeError("Sandbox root is not an overlay filesystem")
    for path in paths:
        selected = max(
            (
                item
                for item in mounts
                if path == item[0] or path.startswith(item[0].rstrip("/") + "/")
            ),
            key=lambda item: len(item[0]),
        )
        if selected[1] != root[1]:
            raise RuntimeError(f"Writable path {path!r} escapes the disk overlay")


class DiskOverlay:
    def __init__(self, root: str, size_mb: int, apptainer: str):
        self.root = Path(root)
        self.size_mb = size_mb
        self.apptainer = apptainer
        self.directory: Path | None = None
        self.image: Path | None = None

    def env(self) -> dict[str, str]:
        value = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "USER", "LOGNAME")
            if key in os.environ
        }
        if self.directory is not None:
            value.update(
                {
                    "APPTAINER_TMPDIR": str(self.directory),
                    "APPTAINER_CACHEDIR": str(self.directory / "cache"),
                    "APPTAINER_CONFIGDIR": str(self.directory / "config"),
                }
            )
        return value

    def create(self, create_dirs: list[str] | tuple[str, ...] = ()) -> Path:
        """Create the image; ``create_dirs`` are absolute directories made in it.

        Start-time read-only mounts need them: Apptainer refuses a bind whose
        destination is missing from a persistent overlay.
        """
        mount = json.loads(checked(["findmnt", "-T", str(self.root), "-J"]))[
            "filesystems"
        ]
        if not mount or any(
            item["fstype"] not in {"xfs", "ext4", "ext3"} for item in mount
        ):
            raise ValueError(
                "Disk overlay root must be local XFS/ext3/ext4, not RAM or network storage"
            )
        if shutil.disk_usage(self.root).free < (self.size_mb + 256) * MIB:
            raise OSError(
                "Insufficient local disk space for requested overlay and headroom"
            )
        self.directory = Path(tempfile.mkdtemp(prefix="disk-overlay-", dir=self.root))
        self.image = self.directory / "writable.img"
        try:
            checked(
                [
                    self.apptainer,
                    "overlay",
                    "create",
                    "--fakeroot",
                    "--sparse",
                    "--size",
                    str(self.size_mb),
                    *(
                        argument
                        for directory in create_dirs
                        for argument in ("--create-dir", directory)
                    ),
                    str(self.image),
                ],
                env=self.env(),
            )
            nominal_bytes = self.size_mb * MIB
            if self.image.stat().st_size != nominal_bytes:
                raise RuntimeError("Overlay has an incorrect nominal size")
            # Sparse formatting skips Apptainer's full-image zero fill. Reserve
            # every extent before use; no sparse image is accepted at startup.
            checked(
                [
                    "fallocate",
                    "--keep-size",
                    "--length",
                    str(nominal_bytes),
                    "--",
                    str(self.image),
                ],
                env=self.env(),
            )
            with self.image.open("r+b") as image:
                os.fsync(image.fileno())
            stat = self.image.stat()
            if stat.st_size != nominal_bytes or stat.st_blocks * 512 < stat.st_size:
                raise RuntimeError(
                    "Overlay was not fully allocated at the requested nominal size"
                )
        except BaseException as error:
            try:
                self.remove()
            except Exception as cleanup_error:
                error.add_note(f"Unstarted overlay cleanup failed: {cleanup_error}")
            raise
        return self.image

    def remove(self) -> None:
        # Caller must have proved instance stop before invoking this method.
        # The target is solely a mkdtemp-created child, never an operator root.
        if self.directory is not None:
            shutil.rmtree(self.directory)
            self.directory = None
            self.image = None
