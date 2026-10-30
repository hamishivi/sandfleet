from __future__ import annotations

import pytest

from sandfleet.errors import SandfleetError
from sandfleet.http import request_json, secret_from_env


class ResettingResponse:
    def __enter__(self) -> ResettingResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        raise ConnectionResetError(104, "Connection reset by peer")


def test_request_wraps_mid_response_connection_reset(monkeypatch):
    monkeypatch.setattr(
        "sandfleet.http.urlopen", lambda *_args, **_kwargs: ResettingResponse()
    )

    with pytest.raises(SandfleetError, match="Connection reset by peer") as caught:
        request_json("http://agent:9000", "/v1/control/leases/lease-1")

    assert caught.value.status is None
    assert isinstance(caught.value.__cause__, ConnectionResetError)


def test_secret_can_be_loaded_from_a_file(monkeypatch, tmp_path):
    secret = tmp_path / "service.token"
    secret.write_text("secret-value\n", encoding="utf-8")
    monkeypatch.delenv("SANDFLEET_TOKEN", raising=False)
    monkeypatch.setenv("SANDFLEET_TOKEN_FILE", str(secret))
    assert secret_from_env("SANDFLEET_TOKEN") == "secret-value"


def test_secret_rejects_ambiguous_sources(monkeypatch, tmp_path):
    secret = tmp_path / "service.token"
    secret.write_text("file-value", encoding="utf-8")
    monkeypatch.setenv("SANDFLEET_TOKEN", "environment-value")
    monkeypatch.setenv("SANDFLEET_TOKEN_FILE", str(secret))
    with pytest.raises(ValueError, match="only one"):
        secret_from_env("SANDFLEET_TOKEN")
