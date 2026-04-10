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
"""Tests for Flower update-check helpers."""


import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests

from . import update_check as update_check_module
from .constant import (
    FLWR_DISABLE_UPDATE_CHECK,
    FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
    FLWR_UPDATE_CHECK_URL,
)
from .update_check import get_flwr_update_check_payload, warn_if_flwr_update_available


class _Response:
    def __init__(self, ok: bool, body: dict[str, Any]) -> None:
        self.ok = ok
        self._body = body

    def json(self) -> dict[str, Any]:
        """Return the mocked JSON response body."""
        return self._body


def test_get_flwr_update_check_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The payload should include the expected runtime metadata."""
    monkeypatch.setattr(
        "flwr.supercore.update_check.platform.python_version", lambda: "3.11.11"
    )
    monkeypatch.setattr("flwr.supercore.update_check.platform.system", lambda: "Linux")
    monkeypatch.setattr("flwr.supercore.update_check.platform.release", lambda: "6.8.0")
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr-nightly")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")

    payload = get_flwr_update_check_payload(process_name="flower-superlink")

    assert payload == {
        "package_name": "flwr-nightly",
        "flwr_version": "1.28.0",
        "python_version": "3.11.11",
        "os": "linux",
        "os_version": "6.8.0",
        "process_name": "flower-superlink",
    }


def test_warn_if_flwr_update_available_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The update check should not run when disabled via environment variable."""
    monkeypatch.setenv(FLWR_DISABLE_UPDATE_CHECK, "1")
    start_thread = Mock()

    monkeypatch.setattr(
        update_check_module, "_start_flwr_update_check_refresh_thread", start_thread
    )

    warn_if_flwr_update_available(process_name="flower-superlink")

    start_thread.assert_not_called()


def _write_update_check_cache(tmp_path: Path, body: dict[str, Any]) -> None:
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "update-check.json").write_text(json.dumps(body), encoding="utf-8")


def _read_update_check_cache(tmp_path: Path) -> dict[str, Any] | None:
    cache_path = tmp_path / ".cache" / "update-check.json"
    if not cache_path.exists():
        return None

    body = json.loads(cache_path.read_text(encoding="utf-8"))
    return body if isinstance(body, dict) else None


def test_warn_if_flwr_update_available_prints_cached_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The cached update message should be printed at most once every 12 hours."""
    monkeypatch.delenv(FLWR_DISABLE_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    monkeypatch.setattr(
        update_check_module,
        "_start_flwr_update_check_refresh_thread",
        lambda process_name=None: None,
    )

    now = datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(update_check_module, "utcnow", lambda: now)
    _write_update_check_cache(
        tmp_path,
        {
            "package_name": "flwr-nightly",
            "flwr_version": "1.28.0",
            "update_available": True,
            "message": "A newer Flower version is available: 1.0.0 -> 1.1.0",
            "last_shown_at": (now - timedelta(days=2)).isoformat(),
        },
    )
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr-nightly")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")

    warn_if_flwr_update_available(process_name="flower-superlink")

    captured = capsys.readouterr()
    assert "Update available" in captured.err
    assert "A newer Flower version is available: 1.0.0 -> 1.1.0" in captured.err
    cache = _read_update_check_cache(tmp_path)
    assert cache is not None
    assert cache["last_shown_at"] == now.isoformat()


def test_warn_if_flwr_update_available_suppresses_recent_cached_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The cached message should not be shown again within 12 hours."""
    monkeypatch.delenv(FLWR_DISABLE_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    monkeypatch.setattr(
        update_check_module,
        "_start_flwr_update_check_refresh_thread",
        lambda process_name=None: None,
    )

    now = datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(update_check_module, "utcnow", lambda: now)
    _write_update_check_cache(
        tmp_path,
        {
            "package_name": "flwr-nightly",
            "flwr_version": "1.28.0",
            "update_available": True,
            "message": "A newer Flower version is available: 1.0.0 -> 1.1.0",
            "last_shown_at": (now - timedelta(hours=6)).isoformat(),
        },
    )
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr-nightly")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")

    warn_if_flwr_update_available(process_name="flower-superlink")

    captured = capsys.readouterr()
    assert captured.err == ""


def test_warn_if_flwr_update_available_renders_brackets_as_plain_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Server-provided Rich markup-like text should be rendered literally."""
    monkeypatch.delenv(FLWR_DISABLE_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    monkeypatch.setattr(
        update_check_module,
        "_start_flwr_update_check_refresh_thread",
        lambda process_name=None: None,
    )

    now = datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(update_check_module, "utcnow", lambda: now)
    _write_update_check_cache(
        tmp_path,
        {
            "package_name": "flwr-nightly",
            "flwr_version": "1.28.0",
            "update_available": True,
            "message": "[bold]literal[/bold] [link=https://flower.ai]text[/link]",
            "last_shown_at": (now - timedelta(days=2)).isoformat(),
        },
    )
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr-nightly")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")

    warn_if_flwr_update_available(process_name="flower-superlink")

    captured = capsys.readouterr()
    assert "[bold]literal[/bold] [link=https://flower.ai]text[/link]" in captured.err


def test_warn_if_flwr_update_available_skips_refresh_if_checked_today(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cache should not refresh again within the same UTC calendar day."""
    monkeypatch.delenv(FLWR_DISABLE_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    start_thread = Mock()

    now = datetime(2026, 3, 18, 15, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(update_check_module, "utcnow", lambda: now)
    monkeypatch.setattr(
        update_check_module, "_start_flwr_update_check_refresh_thread", start_thread
    )
    _write_update_check_cache(
        tmp_path,
        {
            "package_name": "flwr",
            "flwr_version": "1.28.0",
            "update_available": False,
            "last_checked_at": datetime(
                2026, 3, 18, 1, 0, tzinfo=timezone.utc
            ).isoformat(),
        },
    )
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")

    warn_if_flwr_update_available(process_name="flower-superlink")

    start_thread.assert_not_called()


def test_warn_if_flwr_update_available_refreshes_on_new_utc_day(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The cache should refresh again after the UTC date changes."""
    monkeypatch.delenv(FLWR_DISABLE_UPDATE_CHECK, raising=False)
    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    start_thread = Mock()

    now = datetime(2026, 3, 18, 0, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(update_check_module, "utcnow", lambda: now)
    monkeypatch.setattr(
        update_check_module, "_start_flwr_update_check_refresh_thread", start_thread
    )
    _write_update_check_cache(
        tmp_path,
        {
            "package_name": "flwr",
            "flwr_version": "1.28.0",
            "update_available": False,
            "last_checked_at": datetime(
                2026, 3, 17, 23, 30, tzinfo=timezone.utc
            ).isoformat(),
        },
    )
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")

    warn_if_flwr_update_available(process_name="flower-superlink")

    start_thread.assert_called_once_with("flower-superlink")


def test_refresh_flwr_update_check_cache_sends_expected_request_and_writes_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The refresh helper should request and persist update-check state."""
    captured: dict[str, Any] = {}

    def _post(url: str, **kwargs: Any) -> _Response:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["timeout"] = kwargs["timeout"]
        return _Response(
            ok=True,
            body={
                "update_available": True,
                "latest_version": "1.29.0",
                "upgrade_hint": "python -m pip install -U flwr",
                "message": "A newer Flower version is available: 1.28.0 -> 1.29.0",
            },
        )

    def _get_payload(process_name: str | None = None) -> dict[str, str]:
        return {
            "package_name": "flwr-nightly",
            "flwr_version": "1.28.0",
            "python_version": "3.11.11",
            "os": "linux",
            "os_version": "6.8.0",
            "process_name": process_name or "",
        }

    class _ImmediateThread:
        def __init__(self, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            self._target = target
            self._args = args
            self.daemon = daemon

        def start(self) -> None:
            """Execute the target immediately instead of spawning a thread."""
            self._target(*self._args)

    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    monkeypatch.setattr("flwr.supercore.update_check.requests.post", _post)
    monkeypatch.setattr(
        update_check_module, "get_flwr_update_check_payload", _get_payload
    )
    monkeypatch.setattr(update_check_module, "flwr_package_name", "flwr-nightly")
    monkeypatch.setattr(update_check_module, "flwr_version", "1.28.0")
    monkeypatch.setattr(
        "flwr.supercore.update_check.threading.Thread", _ImmediateThread
    )

    warn_if_flwr_update_available(process_name="flower-superlink")

    assert captured == {
        "url": FLWR_UPDATE_CHECK_URL,
        "json": {
            "package_name": "flwr-nightly",
            "flwr_version": "1.28.0",
            "python_version": "3.11.11",
            "os": "linux",
            "os_version": "6.8.0",
            "process_name": "flower-superlink",
        },
        "timeout": (
            FLWR_UPDATE_CHECK_CONNECT_TIMEOUT_SECONDS,
            FLWR_UPDATE_CHECK_READ_TIMEOUT_SECONDS,
        ),
    }
    cache = _read_update_check_cache(tmp_path)
    assert cache is not None
    assert cache["package_name"] == "flwr-nightly"
    assert cache["update_available"] is True
    assert cache["latest_version"] == "1.29.0"
    assert cache["message"] == "A newer Flower version is available: 1.28.0 -> 1.29.0"


def test_refresh_flwr_update_check_cache_swallows_request_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The refresh helper should fail open when the endpoint is unreachable."""

    def _post(*_args: Any, **_kwargs: Any) -> _Response:
        raise requests.RequestException("offline")

    class _ImmediateThread:
        def __init__(self, target: Any, args: tuple[Any, ...], daemon: bool) -> None:
            self._target = target
            self._args = args
            self.daemon = daemon

        def start(self) -> None:
            """Execute the target immediately instead of spawning a thread."""
            self._target(*self._args)

    monkeypatch.setattr(update_check_module, "get_flwr_home", lambda: tmp_path)
    monkeypatch.setattr("flwr.supercore.update_check.requests.post", _post)
    monkeypatch.setattr(
        "flwr.supercore.update_check.threading.Thread", _ImmediateThread
    )

    warn_if_flwr_update_available(process_name="flower-superlink")
    assert _read_update_check_cache(tmp_path) is None
