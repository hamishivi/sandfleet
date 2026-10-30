"""Strict mapping from complete environment contexts to operator-built SIFs."""

import hashlib
import json
from pathlib import Path
import re


def context_digest(directory: Path) -> str:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Environment context must be a real directory")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Image context symlinks require explicit build review")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Image context contains a non-regular file")
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        files.append(
            [
                path.relative_to(directory).as_posix(),
                path.stat().st_mode & 0o777,
                digest,
            ]
        )
    return hashlib.sha256(
        json.dumps(files, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


def resolve_image(
    manifest_path: Path, directory: Path, name: str | None = None
) -> dict:
    """The SIF for an environment context, or, given ``name`` (a shared prebuilt
    image such as a separate verifier's ``docker_image``), the manifest's
    ``named`` entry (same fields; its context is not hashed)."""
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported SIF manifest schema")
    if name is not None:
        digest = f"named:{name}"
        images = {
            digest: image for image in [manifest.get("named", {}).get(name)] if image
        }
    else:
        digest = context_digest(directory)
        images = manifest["images"]
    try:
        image = dict(images[digest])
    except KeyError as error:
        raise ValueError(
            f"No exact built SIF for environment context {digest}"
        ) from error
    if (
        not isinstance(image["sif_path"], str)
        or not Path(image["sif_path"]).is_absolute()
        or not image["sif_path"].endswith(".sif")
    ):
        raise ValueError("Manifest requires an absolute worker-readable SIF path")
    if re.fullmatch(r"[0-9a-f]{64}", image["sif_sha256"]) is None:
        raise ValueError("Manifest requires a pinned SIF sha256")
    if not image["build_provenance"]:
        raise ValueError("Manifest must record the complete-context build provenance")
    if image["user"] not in ("", "root", "0"):
        raise ValueError("This backend currently supports root-user images only")
    if not isinstance(image["workdir"], str) or not image["workdir"].startswith("/"):
        raise ValueError("Manifest must record the image's absolute WORKDIR")
    if not isinstance(image["env"], dict) or any(
        not isinstance(value, str) for value in image["env"].values()
    ):
        raise ValueError("Manifest ENV must be a string mapping")
    image["context_sha256"] = digest
    return image
