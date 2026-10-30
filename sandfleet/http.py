"""Small authenticated JSON-over-HTTP helpers using only the standard library."""

from __future__ import annotations

import base64
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sandfleet.errors import (
    NoCapacityError,
    SandfleetError,
    SandboxLostError,
    SandboxOOMError,
)
from sandfleet.version import VERSION

API_VERSION = "v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024


def secret_from_env(name: str) -> str:
    """Load a secret from NAME or the permission-restricted NAME_FILE path."""
    value = os.environ.get(name)
    path = os.environ.get(f"{name}_FILE")
    if value and path:
        raise ValueError(f"Set only one of {name} and {name}_FILE")
    if path:
        value = Path(path).read_text(encoding="utf-8").strip()
    return value or ""


def encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def decode_bytes(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    body = (
        None
        if payload is None
        else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    )
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        base_url.rstrip("/") + path, data=body, headers=headers, method=method
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller controls a private service URL
            raw = response.read()
            return json.loads(raw) if raw else {}
    except HTTPError as error:
        raw = error.read()
        try:
            detail = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            detail = {}
        error_info = detail.get("error", {})
        error_type = error_info.get("type", "SandfleetError")
        message = error_info.get(
            "message", f"Sandfleet request failed with HTTP {error.code}"
        )
        error_class = {
            "SandboxOOMError": SandboxOOMError,
            "SandboxLostError": SandboxLostError,
            "NoCapacityError": NoCapacityError,
        }.get(error_type, SandfleetError)
        raise error_class(message, error_type=error_type, status=error.code) from error
    except URLError as error:
        raise SandfleetError(
            f"Could not reach Sandfleet endpoint {base_url!r}: {error.reason}"
        ) from error
    except OSError as error:
        raise SandfleetError(
            f"Connection to Sandfleet endpoint {base_url!r} failed: {error}"
        ) from error


class JsonHandler(BaseHTTPRequestHandler):
    """Base request handler with consistent JSON errors and bearer auth."""

    server_version = f"Sandfleet/{VERSION}"

    def log_message(self, fmt: str, *args: Any) -> None:
        # Use the server's logger-like callback when present.
        callback = getattr(self.server, "log_callback", None)
        if callback:
            callback(f"{self.client_address[0]} - {fmt % args}")

    def read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body exceeds {MAX_REQUEST_BYTES} bytes")
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON body: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError("JSON request body must be an object")
        return value

    def bearer_token(self) -> str:
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        return header[len(prefix) :] if header.startswith(prefix) else ""

    def path_parts(self) -> list[str]:
        return [part for part in self.path.partition("?")[0].split("/") if part]

    def require_token(self, expected: str) -> None:
        if not expected or not hmac.compare_digest(self.bearer_token(), expected):
            raise PermissionError("Invalid or missing bearer token")

    def send_json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(
        self,
        status: int,
        value: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
    ) -> None:
        body = value.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, error: Exception) -> None:
        if isinstance(error, PermissionError):
            status = 401
        elif isinstance(error, KeyError):
            status = 404
        elif isinstance(error, NoCapacityError):
            status = 503
        elif isinstance(error, (ValueError, FileNotFoundError, IsADirectoryError)):
            status = 400
        elif isinstance(error, SandfleetError) and error.status:
            status = error.status
        else:
            status = 500
        self.send_json(
            status,
            {"error": {"type": type(error).__name__, "message": str(error)}},
        )
