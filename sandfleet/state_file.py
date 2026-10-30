"""Crash-durable JSON state with one validated fallback generation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar


DecodedState = TypeVar("DecodedState")


class StateValidationError(RuntimeError):
    """A state snapshot is readable JSON but violates the persisted schema."""


@dataclass(frozen=True)
class StateRecovery:
    source: Path
    quarantined: Path | None
    failures: tuple[str, ...]


class DurableStateFile(Generic[DecodedState]):
    """Persist one current and one previous fsynced JSON generation."""

    def __init__(self, path: Path):
        self.path = path
        self.backup_path = path.with_suffix(path.suffix + ".bak")
        self.temporary_path = path.with_suffix(path.suffix + ".tmp")

    def load(
        self, decode: Callable[[dict[str, Any]], DecodedState]
    ) -> tuple[DecodedState | None, StateRecovery | None]:
        candidates = (self.path, self.backup_path, self.temporary_path)
        if not any(candidate.exists() for candidate in candidates):
            return None, None

        failures: list[str] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                raw = candidate.read_bytes()
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise StateValidationError("State root must be a JSON object")
                decoded = decode(payload)
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                StateValidationError,
            ) as error:
                failures.append(f"{candidate}: {type(error).__name__}: {error}")
                continue
            if candidate == self.path:
                self._fsync_file(candidate)
                self._fsync_parent()
                return decoded, None
            quarantined = self._restore_primary(candidate, raw)
            return decoded, StateRecovery(
                source=candidate,
                quarantined=quarantined,
                failures=tuple(failures),
            )

        details = "; ".join(failures)
        raise RuntimeError(f"No valid Sandfleet state generation found: {details}")

    def save(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._write_temporary(raw)
        if self.path.exists():
            os.replace(self.path, self.backup_path)
            self._fsync_parent()
        os.replace(self.temporary_path, self.path)
        self._fsync_parent()

    def _restore_primary(self, source: Path, raw: bytes) -> Path | None:
        quarantined = None
        if self.path.exists():
            quarantined = self.path.with_name(
                f"{self.path.name}.corrupt-{time.time_ns()}"
            )
            os.replace(self.path, quarantined)
            self._fsync_parent()
        if source == self.temporary_path:
            self._fsync_file(source)
        else:
            self._write_temporary(raw)
        os.replace(self.temporary_path, self.path)
        self._fsync_parent()
        return quarantined

    def _write_temporary(self, raw: bytes) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    def _fsync_parent(self) -> None:
        descriptor = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
