# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for the infrastructure."""


import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import requests

from flwr.common.constant import FLWR_DIR, FLWR_HOME
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.supercore.version import package_version as flwr_version

from .constant import APP_ID_PATTERN, APP_VERSION_PATTERN, MAX_NAME_LENGTH

T = TypeVar("T", str, bytes)


def mask_string(value: str, head: int = 4, tail: int = 4) -> str:
    """Mask a string by preserving only the head and tail characters.

    Mask a string for safe display by preserving the head and tail characters,
    and replacing the middle with '...'. Useful for logging tokens, secrets,
    or IDs without exposing sensitive data.

    Notes
    -----
    If the string is shorter than the combined length of `head` and `tail`,
    the original string is returned unchanged.
    """
    if len(value) <= head + tail:
        return value
    return f"{value[:head]}...{value[-tail:]}"


def uint64_to_int64(unsigned: int) -> int:
    """Convert a uint64 integer to a sint64 with the same bit pattern.

    For values >= 2^63, wraps around by subtracting 2^64.
    """
    if unsigned >= (1 << 63):
        return unsigned - (1 << 64)
    return unsigned


def int64_to_uint64(signed: int) -> int:
    """Convert a sint64 integer to a uint64 with the same bit pattern.

    For negative values, wraps around by adding 2^64.
    """
    if signed < 0:
        return signed + (1 << 64)
    return signed


def get_flwr_home() -> Path:
    """Get the Flower home directory path.

    Returns FLWR_HOME environment variable if set, otherwise returns a default
    subdirectory in the user's home directory.
    """
    if flwr_home := os.getenv(FLWR_HOME):
        return Path(flwr_home)
    return Path.home() / FLWR_DIR


def parse_app_spec(app_spec: str) -> tuple[str, str | None]:
    """Parse app specification string into app ID and version.

    Parameters
    ----------
    app_spec : str
        The app specification string in the format '@account/app' or
        '@account/app==x.y.z' (digits only).

    Returns
    -------
    tuple[str, str | None]
        A tuple containing the app ID and optional version.

    Raises
    ------
    ValueError
        If the app specification format is invalid.
    """
    if "==" in app_spec:
        app_id, app_version = app_spec.split("==", 1)

        if not re.match(APP_VERSION_PATTERN, app_version):
            raise ValueError(
                "Invalid app version. Expected format: x.y.z (digits only)."
            )
    else:
        app_id = app_spec
        app_version = None

    if not re.match(APP_ID_PATTERN, app_id):
        raise ValueError(
            "Invalid remote app ID. Expected format: '@account_name/app_name'."
        )

    return app_id, app_version


def request_download_link(
    app_id: str, app_version: str | None, in_url: str, out_url: str
) -> tuple[str, list[dict[str, str]] | None, str | None]:
    """Request a download link for the given app from Flower Hub.

    Parameters
    ----------
    app_id : str
        The application identifier in the format '@account/app'.
    app_version : str | None
        The application version (e.g., '1.2.3'), or None to request the latest version.
    in_url : str
        The Platform API endpoint URL to query.
    out_url : str
        The key name in the response that contains the download URL.

    Returns
    -------
    tuple[str, list[dict[str, str]] | None, str | None]
        A tuple containing:
        - The download URL for the application.
        - A list of verification dictionaries if provided by the API, otherwise None.
        - A compatibility note if provided by the API, otherwise None.

    Raises
    ------
    ValueError
        If the API connection fails, the application or version is not found,
        the API returns a non-200 response, or the response format is invalid.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    body = {
        "app_id": app_id,  # send raw string of app_id
        "app_version": app_version,
        "flwr_version": flwr_version,
    }

    try:
        resp = requests.post(in_url, headers=headers, data=json.dumps(body), timeout=20)
    except requests.RequestException as e:
        raise ValueError(f"Unable to connect to Platform API: {e}") from e

    if resp.status_code == 404:
        # Expecting a JSON body with a "detail" field
        try:
            error_message = resp.json().get("detail")
        except ValueError:
            # JSON parsing failed
            raise ValueError(f"{app_id} not found in Platform API.") from None

        if isinstance(error_message, dict):
            available_app_versions = error_message.get("available_app_versions", [])
            available_versions_str = (
                ", ".join(map(str, available_app_versions))
                if available_app_versions
                else "None"
            )
            raise ValueError(
                f"{app_id}=={app_version} not found in Platform API. "
                f"Available app versions for {app_id}: {available_versions_str}"
            )

        raise ValueError(f"{app_id} not found in Platform API.")

    if not resp.ok:
        raise ValueError(
            f"Platform API request failed with status {resp.status_code}. "
            f"Details: {resp.text}"
        )

    data = resp.json()
    if out_url not in data:
        raise ValueError("Invalid response from Platform API")

    verifications = data["verifications"] if "verifications" in data else None
    note = data["note"] if "note" in data else None

    return str(data[out_url]), verifications, note


def simulation_config_to_json(config: SimulationConfig) -> dict[str, Any]:
    """Convert a simulation config protobuf to a JSON-serializable dictionary."""
    payload: dict[str, Any] = {}
    for field in config.DESCRIPTOR.fields:
        if field.has_presence and not config.HasField(field.name):
            payload[field.name] = None
            continue
        payload[field.name] = getattr(config, field.name)

    return payload


def simulation_config_from_json(payload: dict[str, Any]) -> SimulationConfig:
    """Convert a JSON payload into a simulation config protobuf."""
    config = SimulationConfig()
    valid_fields = {field.name for field in config.DESCRIPTOR.fields}
    unknown_fields = set(payload) - valid_fields
    if unknown_fields:
        field_names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown simulation config field(s): {field_names}")

    for field_name, value in payload.items():
        if value is None:
            continue
        setattr(config, field_name, value)

    return config


def humanize_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-friendly string.

    Rules:
      - < 90 seconds: show seconds
      - < 1 hour: show minutes + seconds
      - < 1 day: show hours + minutes
      - >= 1 day: show days + hours
    """
    seconds = int(seconds)

    # Under 90 seconds → Seconds only
    if seconds < 90:
        return f"{seconds}s"

    # Under 1 hour → Minutes and seconds
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"

    # Under 1 day → Hours and minutes
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes}m"

    # 1+ days → Days and hours
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h"


def humanize_bytes(num_bytes: int) -> str:
    """Convert a number of bytes to a human-friendly string.

    Uses 1024-based units and 0-1 decimal precision.
    Rules:
      - < 1 KB: bytes
      - < 1 MB: KB
      - < 1 GB: MB
      - < 1 TB: GB
    """
    value = float(num_bytes)

    for suffix in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or suffix == "TB":
            # Bytes → no decimals
            if suffix == "B":
                return f"{int(value)} B"

            # Decide precision: 1 decimal for <10, otherwise no decimal
            if value < 10:
                formatted = f"{value:.1f}"
            else:
                formatted = f"{int(value)}"

            return f"{formatted} {suffix}"

        value /= 1024

    raise RuntimeError("Unreachable code")  # Make mypy happy


def check_federation_format(federation: str) -> None:
    """Check if the federation string is valid.

    Parameters
    ----------
    federation : str
        The federation string to check.

    Raises
    ------
    ValueError
        If the federation string is not valid. The expected
        format is '@<account-name>/<federation-name>'.
    """
    if not re.match(r"^@[a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+$", federation):
        raise ValueError(
            f"Invalid federation format: {federation}. "
            f"Expected format: '@<account-name>/<federation-name>'."
        )


def is_valid_name(name: str) -> tuple[bool, str]:
    """Check if the given string is a valid name for an app or federation.

    A valid name must start with a letter and can only contain letters, digits, and
    hyphens. It must be less than or equal to MAX_NAME_LENGTH characters.
    """
    if not name:
        return False, "Cannot be empty."

    # Check if the name exceeds the maximum length
    if len(name) > MAX_NAME_LENGTH:
        return False, f"Must be no longer than {MAX_NAME_LENGTH} characters."

    # Check if the first character is a letter
    if not name[0].isalpha():
        return False, "Must start with a letter."

    # Check if the rest of the characters are valid (letter, digit, or dash)
    for char in name[1:]:
        if not (char.isalnum() or char in "-"):
            return False, "Can only contain letters, digits, and hyphens."

    return True, ""


def _get_metadata_typed(
    metadata: Sequence[tuple[str, str | bytes]] | None,
    key: str,
    value_type: type[T],
) -> T | None:
    """Return exactly one non-empty string or bytes metadata value for `key`."""
    if metadata is None:
        return None
    values: list[Any] = [
        value for metadata_key, value in metadata if metadata_key == key
    ]
    if len(values) != 1:
        return None
    value = values[0]
    if not isinstance(value, value_type):
        return None
    if value in ("", b""):
        return None
    return value


def get_metadata_str(
    metadata: Sequence[tuple[str, str | bytes]] | None,
    key: str,
) -> str | None:
    """Return exactly one non-empty string metadata value for `key`."""
    return _get_metadata_typed(metadata, key, str)


def get_metadata_bytes(
    metadata: Sequence[tuple[str, str | bytes]] | None,
    key: str,
) -> bytes | None:
    """Return exactly one non-empty bytes metadata value for `key`."""
    return _get_metadata_typed(metadata, key, bytes)
