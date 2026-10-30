"""Per-worker JSONL audit trail of sandbox commands and file writes."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Commands longer than this are truncated in the audit record (not in execution).
AUDIT_COMMAND_BYTES = 4096
DEFAULT_AUDIT_MAX_MB = 256


def truncate_utf8(text: str, limit: int = AUDIT_COMMAND_BYTES) -> tuple[str, bool]:
    """Return at most `limit` UTF-8 bytes of `text` and whether it was cut."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


class AuditLog:
    """Append one JSON line per sandbox operation to a size-bounded file.

    The file is rotated to `<path>.1` (replacing an older one) before a write
    would exceed `max_bytes`, so a worker keeps at most twice that on disk.
    Records are written after the operation finishes. A failed write never
    fails the operation, which has already run and may not be idempotent; it
    is counted in `failures` and reported on stderr (the agent log).
    """

    def __init__(
        self, path: str | Path, *, max_bytes: int, context: dict[str, Any]
    ) -> None:
        if max_bytes < 1:
            raise ValueError("audit max_bytes must be positive")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.context = dict(context)
        self.failures = 0
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, fields: dict[str, Any]) -> None:
        now = time.time()
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
            + f".{int(now % 1 * 1000):03d}Z",
            **self.context,
            **fields,
        }
        line = (json.dumps(entry, sort_keys=True, default=str) + "\n").encode()
        with self._lock:
            try:
                self._append(line)
            except OSError as error:
                self.failures += 1
                print(
                    f"Sandfleet audit write to {self.path} failed "
                    f"({self.failures} so far): {error}",
                    file=sys.stderr,
                    flush=True,
                )

    def _append(self, line: bytes) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size and size + len(line) > self.max_bytes:
            os.replace(self.path, self.path.with_name(self.path.name + ".1"))
        descriptor = os.open(
            self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600
        )
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)
