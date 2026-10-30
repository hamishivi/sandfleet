"""Validation of literal, per-command sandbox environment overrides."""

import re


def validate_command_env(env: dict[str, str]) -> dict[str, str]:
    if not isinstance(env, dict) or any(
        not isinstance(key, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None
        or not isinstance(value, str)
        or "\x00" in value
        for key, value in env.items()
    ):
        raise ValueError(
            "Command env must map shell variable names to NUL-free strings"
        )
    return env
