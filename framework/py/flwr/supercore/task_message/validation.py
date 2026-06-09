# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Validation helpers for task message payloads."""

from __future__ import annotations

from collections.abc import Sequence

from flwr.supercore.typing import JSONObject, JSONValue


def set_optional(payload: JSONObject, field: str, value: JSONValue | None) -> None:
    """Add an optional payload field only when the caller provided a value.

    This keeps absent optional fields out of the JSON payload instead of encoding
    them as ``null``.
    """
    if value is not None:
        payload[field] = value


def validate_present(payload: JSONObject, field: str, *, owner: str) -> None:
    """Validate that a payload field exists, without validating its value.

    Use this when ``None`` is a valid value and type-specific validation happens
    separately.
    """
    if field not in payload:
        raise ValueError(f"{owner} payload requires field '{field}'.")


def validate_non_empty_string(
    payload: JSONObject, field: str, *, owner: str
) -> None:
    """Validate that a payload field exists and is a non-empty string."""
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner} payload requires a non-empty string field '{field}'."
        )


def validate_optional_string(
    payload: JSONObject, field: str, *, owner: str
) -> None:
    """Validate that an optional payload field is a string when present."""
    if field in payload and not isinstance(payload[field], str):
        raise ValueError(f"{owner} payload field '{field}' must be a string.")


def validate_optional_bool(payload: JSONObject, field: str, *, owner: str) -> None:
    """Validate that an optional payload field is a bool when present."""
    if field in payload and not isinstance(payload[field], bool):
        raise ValueError(f"{owner} payload field '{field}' must be a bool.")


def validate_optional_int(payload: JSONObject, field: str, *, owner: str) -> None:
    """Validate that an optional payload field is an integer when present."""
    if field in payload and not isinstance(payload[field], int):
        raise ValueError(f"{owner} payload field '{field}' must be an integer.")


def validate_json_object(
    payload: JSONObject,
    field: str,
    *,
    owner: str,
    required: bool = True,
    allow_none: bool = False,
) -> None:
    """Validate that a payload field is a JSON object.

    ``required`` controls whether the field must exist. ``allow_none`` allows
    explicit ``null`` values for fields such as response errors.
    """
    if field not in payload:
        if required:
            raise ValueError(f"{owner} payload requires a JSON object field '{field}'.")
        return

    value = payload[field]
    if allow_none and value is None:
        return
    if not isinstance(value, dict):
        if required:
            raise ValueError(f"{owner} payload requires a JSON object field '{field}'.")
        raise ValueError(f"{owner} payload field '{field}' must be a JSON object.")


def validate_json_object_sequence(
    payload: JSONObject,
    field: str,
    *,
    owner: str,
    required: bool = False,
) -> None:
    """Validate that a payload field is a sequence of JSON objects.

    ``required`` controls whether the field must exist. When present, each item
    must be a JSON object.
    """
    if field not in payload:
        if required:
            raise ValueError(f"{owner} payload requires field '{field}'.")
        return

    value = payload[field]
    # Strings are sequences in Python, but they are not valid object sequences.
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not all(isinstance(item, dict) for item in value)
    ):
        raise ValueError(
            f"{owner} payload field '{field}' must be a sequence of JSON objects."
        )
