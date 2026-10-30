"""Controller discovery and bounded retries; never replay sandbox commands."""

from __future__ import annotations

import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from sandfleet.errors import SandfleetError

RETRY_SECONDS = 300.0
RETRY_STATUSES = {408, 429, 502, 503, 504}


def controller_url(url: str, registry: str | None = None) -> str:
    """Read the operator-owned shared registry on every controller request."""
    registry = registry or os.environ.get("SANDFLEET_REGISTRY")
    if not registry:
        return url
    try:
        payload = json.loads(Path(registry).read_text(encoding="utf-8"))
    except FileNotFoundError:
        # During shutdown the registry can temporarily disappear. Keep trying
        # the last supplied address until the replacement publishes readiness.
        return url
    if not isinstance(payload, dict) or not isinstance(payload.get("url"), str):
        raise ValueError("Sandfleet registry must contain a string URL")
    discovered = payload["url"]
    parsed = urlsplit(discovered)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Invalid Sandfleet registry URL")
    return discovered.rstrip("/")


def transient(error: SandfleetError) -> bool:
    return error.status in RETRY_STATUSES or (
        error.status is None and isinstance(error.__cause__, OSError)
    )


def controller_request(
    transport: Callable[..., dict[str, Any]],
    url: str,
    path: str,
    *,
    registry: str | None = None,
    retry_seconds: float = RETRY_SECONDS,
    stop: threading.Event | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Retry only reads, release, and renewal; ambiguous acquisitions fail."""
    retry = kwargs.get("method", "GET") in {"GET", "DELETE"} or path.endswith("/renew")
    deadline = time.monotonic() + retry_seconds
    delay = 0.5
    while True:
        if stop is not None and stop.is_set():
            raise InterruptedError("Controller request stopped")
        remaining = max(0.001, deadline - time.monotonic())
        attempt = {**kwargs, "timeout": min(kwargs.get("timeout", 60), remaining)}
        try:
            return transport(controller_url(url, registry), path, **attempt)
        except SandfleetError as error:
            remaining = deadline - time.monotonic()
            cause = error.__cause__
            refused = isinstance(
                getattr(cause, "reason", cause), ConnectionRefusedError
            )
            if not (retry or refused) or not transient(error) or remaining <= 0:
                raise
            wait = min(delay * random.uniform(0.9, 1.1), remaining)
            if stop is None:
                time.sleep(wait)
            elif stop.wait(wait):
                raise InterruptedError("Controller request stopped") from error
            delay = min(5.0, delay * 2)
