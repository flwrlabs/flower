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
"""Flower update-check helpers."""


import json
import os
import platform
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from flwr.supercore.date import now as utcnow
from flwr.supercore.version import package_name as flwr_package_name
from flwr.supercore.version import package_version as flwr_version

from .constant import (
    FLWR_DISABLE_UPDATE_CHECK,
    FLWR_UPDATE_CHECK_CACHE_DIR,
    FLWR_UPDATE_CHECK_CACHE_FILENAME,
    FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_SHOW_INTERVAL_SECONDS,
    FLWR_UPDATE_CHECK_URL,
)
from .utils import get_flwr_home

__all__ = ["get_flwr_update_check_payload", "warn_if_flwr_update_available"]


def get_flwr_update_check_payload(process_name: str | None = None) -> dict[str, str]:
    """Return the runtime payload sent to the update-check endpoint."""
    payload = {
        "package_name": flwr_package_name,
        "flwr_version": flwr_version,
        "python_version": platform.python_version(),
        "os": platform.system().lower(),
        "os_version": platform.release(),
    }
    if process_name:
        payload["process_name"] = process_name
    return payload


def _get_flwr_update_check_cache_path() -> Path:
    """Return the local cache file path for update-check state."""
    return (
        get_flwr_home() / FLWR_UPDATE_CHECK_CACHE_DIR / FLWR_UPDATE_CHECK_CACHE_FILENAME
    )


def _parse_flwr_update_check_timestamp(value: Any) -> datetime | None:
    """Parse a timestamp from cache state."""
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _read_flwr_update_check_cache() -> dict[str, Any] | None:
    """Read cached update-check state from disk."""
    cache_path = _get_flwr_update_check_cache_path()
    try:
        body = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    return body if isinstance(body, dict) else None


def _write_flwr_update_check_cache(cache: dict[str, Any]) -> None:
    """Atomically persist cached update-check state to disk."""
    cache_path = _get_flwr_update_check_cache_path()
    tmp_path = cache_path.parent / f".{cache_path.name}.tmp"

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(cache), encoding="utf-8")
        tmp_path.replace(cache_path)
    except OSError:
        pass


def _cache_matches_current_install(cache: dict[str, Any]) -> bool:
    """Return True if the cached state belongs to the current install."""
    cached_package_name = cache.get("package_name") or "flwr"
    cached_flwr_version = cache.get("flwr_version")

    return (
        cached_package_name == flwr_package_name and cached_flwr_version == flwr_version
    )


def _should_show_cached_flwr_update_message(cache: dict[str, Any]) -> bool:
    """Return True if the cached update message should be shown now."""
    if not _cache_matches_current_install(cache):
        return False

    if not cache.get("update_available"):
        return False

    message = cache.get("message")
    if not isinstance(message, str) or not message.strip():
        return False

    last_shown_at = _parse_flwr_update_check_timestamp(cache.get("last_shown_at"))
    if last_shown_at is None:
        return True

    return utcnow() - last_shown_at >= timedelta(
        seconds=FLWR_UPDATE_CHECK_SHOW_INTERVAL_SECONDS
    )


def _should_refresh_flwr_update_check_cache(cache: dict[str, Any] | None) -> bool:
    """Return True if cached state should be refreshed from the server."""
    if cache is None:
        return True

    if not _cache_matches_current_install(cache):
        return True

    last_checked_at = _parse_flwr_update_check_timestamp(cache.get("last_checked_at"))
    if last_checked_at is None:
        return True

    return last_checked_at.date() < utcnow().date()


def _mark_cached_flwr_update_message_shown(cache: dict[str, Any]) -> None:
    """Update and persist the last time the cached message was shown."""
    cache["last_shown_at"] = utcnow().isoformat()
    _write_flwr_update_check_cache(cache)


def _request_flwr_update_check(
    process_name: str | None = None,
) -> dict[str, Any] | None:
    """Perform the update-check request and return a parsed JSON object."""
    try:
        response = requests.post(
            FLWR_UPDATE_CHECK_URL,
            json=get_flwr_update_check_payload(process_name=process_name),
            timeout=(
                FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
                FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
            ),
        )
    except requests.RequestException:
        return None

    if not response.ok:
        return None

    try:
        body: Any = response.json()
    except ValueError:
        return None

    return body if isinstance(body, dict) else None


def _refresh_flwr_update_check_cache(process_name: str | None = None) -> None:
    """Refresh cached update-check state from the platform API."""
    body = _request_flwr_update_check(process_name=process_name)
    if body is None:
        return

    previous_cache = _read_flwr_update_check_cache() or {}
    cache: dict[str, Any] = {
        "package_name": flwr_package_name,
        "flwr_version": flwr_version,
        "update_available": body.get("update_available") is True,
        "last_checked_at": utcnow().isoformat(),
    }

    if cache["update_available"]:
        message = body.get("message")
        latest_version = body.get("latest_version")
        upgrade_hint = body.get("upgrade_hint")

        if isinstance(message, str) and message.strip():
            cache["message"] = message
        if isinstance(latest_version, str) and latest_version.strip():
            cache["latest_version"] = latest_version
        if isinstance(upgrade_hint, str) and upgrade_hint.strip():
            cache["upgrade_hint"] = upgrade_hint

        if _cache_matches_current_install(previous_cache):
            last_shown_at = previous_cache.get("last_shown_at")
            if isinstance(last_shown_at, str):
                cache["last_shown_at"] = last_shown_at

    _write_flwr_update_check_cache(cache)


def _start_flwr_update_check_refresh_thread(process_name: str | None = None) -> None:
    """Refresh cached update-check state in the background."""
    thread = threading.Thread(
        target=_refresh_flwr_update_check_cache,
        args=(process_name,),
        daemon=False,
    )
    thread.start()


def warn_if_flwr_update_available(process_name: str | None = None) -> None:
    """Print the cached update message and refresh state in the background."""
    if os.getenv(FLWR_DISABLE_UPDATE_CHECK) == "1":
        return

    cache = _read_flwr_update_check_cache()
    if cache is not None and _should_show_cached_flwr_update_message(cache):
        message = cache.get("message")
        if isinstance(message, str):
            Console(file=sys.stderr).print(
                Panel.fit(
                    Text(message),
                    title="Update available",
                    border_style="yellow",
                )
            )
            _mark_cached_flwr_update_message_shown(cache)

    if _should_refresh_flwr_update_check_cache(cache):
        _start_flwr_update_check_refresh_thread(process_name)
