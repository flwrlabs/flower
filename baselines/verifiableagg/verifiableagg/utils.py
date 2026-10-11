"""Shared utility helpers for verifiable aggregation baseline."""

from __future__ import annotations


def as_bool(value: object) -> bool:
    """Convert bool-like run-config values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")
