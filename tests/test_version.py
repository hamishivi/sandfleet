"""Source/PYTHONPATH launches must enforce the package's Python requirement."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
ENTRY_POINTS = [
    ("module", name)
    for name in (
        "sandfleet.cli",
        "sandfleet.controller",
        "sandfleet.agent",
        "sandfleet.step_worker",
    )
] + [("console", target) for target in SCRIPTS.values()]

STARTUP = """
import importlib
import json
import runpy
import sys

kind, target = sys.argv[1:]
sys.argv = [target, "--help"]
sys.version_info = (3, 10, 0, "final", 0)
try:
    if kind == "module":
        runpy.run_module(target, run_name="__main__")
    else:
        module, function = target.split(":")
        getattr(importlib.import_module(module), function)()
except RuntimeError:
    print(json.dumps(sorted(name for name in sys.modules if name.startswith("sandfleet."))))
    raise
"""


@pytest.mark.parametrize(("kind", "target"), ENTRY_POINTS)
def test_unsupported_python_fails_before_runtime_imports(kind, target):
    result = subprocess.run(
        [sys.executable, "-c", STARTUP, kind, target],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode != 0
    assert result.stderr.strip().splitlines()[-1] == (
        "RuntimeError: Sandfleet requires Python 3.11 or newer; found Python 3.10"
    )
    assert json.loads(result.stdout) == []
