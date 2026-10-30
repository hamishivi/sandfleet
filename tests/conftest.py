"""Test-suite safety net for the sandbox process-kill paths.

No unit test should ever send a real ``kill(-1)``, join a namespace, or spawn a
real process that would (the namespace-kill helper, nsenter, unshare, or a shell
``kill ... -1``). Tests mock those paths; this autouse guard makes the real
operations fail loudly if a mock is ever missing, across every test file.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

# CLONE_NEWPID marks the namespace-kill helper program; nsenter/unshare enter or
# create namespaces; `kill [opts] -1` signals every process the caller can reach.
_FORBIDDEN_ARGV = re.compile(
    r"CLONE_NEWPID|\bnsenter\b|\bunshare\b|\bkill\b[^;&|\n]*\s-1(?![\w])"
)


def _forbidden_argv(args, kwargs):
    candidate = kwargs.get("args", args[0] if args else None)
    parts = candidate if isinstance(candidate, (list, tuple)) else [candidate]
    return bool(_FORBIDDEN_ARGV.search(" ".join(str(part) for part in parts)))


@pytest.fixture(autouse=True)
def forbid_real_namespace_kill(monkeypatch):
    real_run = subprocess.run
    real_popen = subprocess.Popen
    real_kill = os.kill

    def guarded_run(*args, **kwargs):
        if _forbidden_argv(args, kwargs):
            raise AssertionError(f"test attempted a forbidden command: {args!r}")
        return real_run(*args, **kwargs)

    def guarded_popen(*args, **kwargs):
        if _forbidden_argv(args, kwargs):
            raise AssertionError(f"test attempted a forbidden command: {args!r}")
        return real_popen(*args, **kwargs)

    def guarded_kill(pid, sig):
        if pid == -1:
            raise AssertionError("test attempted a real os.kill(-1, ...)")
        return real_kill(pid, sig)

    def forbidden_setns(*_args, **_kwargs):
        raise AssertionError("test attempted a real os.setns")

    monkeypatch.setattr(subprocess, "run", guarded_run)
    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    monkeypatch.setattr(os, "kill", guarded_kill)
    monkeypatch.setattr(os, "setns", forbidden_setns)
    yield
