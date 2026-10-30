"""Archive destinations must not accidentally depend on the episode cwd.

These host-tar tests cover cwd selection, not Apptainer environment plumbing.
Real Apptainer missing-cwd behavior also requires the isolated integration probe.
"""

import io
import subprocess
import tarfile

import pytest

from sandfleet.step_worker import ApptainerInstance


@pytest.mark.parametrize("relative", [False, True])
def test_archive_cwd(tmp_path, monkeypatch, relative):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    destination = workspace / "tests"
    destination.mkdir()
    instance = ApptainerInstance(
        image="/image.sif",
        timeout=30,
        pwd=str(workspace),
        cache_dir=None,
        tmp_dir=None,
        extra_start_flags=[],
        apptainer_binary="apptainer",
    )
    instance.name = "test"
    instance.network = None

    def run(argv, *, stdin=None, pwd=None, **kwargs):
        return subprocess.run(
            argv,
            input=stdin,
            capture_output=True,
            cwd=instance.pwd if pwd is None else pwd,
        )

    monkeypatch.setattr(instance, "_exec_internal", run)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        member = tarfile.TarInfo("marker")
        member.size = 2
        tar.addfile(member, io.BytesIO(b"ok"))
    if not relative:
        instance.pwd = str(tmp_path / "missing-cwd")
    instance.put_archive("tests" if relative else str(destination), archive.getvalue())
    assert (destination / "marker").read_bytes() == b"ok"
    with pytest.raises(RuntimeError):
        instance.put_archive(str(tmp_path / "absent-destination"), archive.getvalue())
    with pytest.raises(RuntimeError):
        instance.put_archive(str(destination), b"not a tar archive")
    if not relative:
        with pytest.raises(FileNotFoundError):
            instance.exec(["pwd"])
    with pytest.raises((RuntimeError, FileNotFoundError)):
        instance.put_archive("missing-destination", archive.getvalue())
