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
"""Tests for optional SuperLink extensions."""

import asyncio
import threading
from types import ModuleType
from unittest.mock import Mock

import pytest

from flwr.supercore.run import Run

from . import extensions


def test_notify_run_started_calls_installed_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward successful run creation to an installed extension."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    callback = Mock()
    completed = threading.Event()

    def on_run_started(*args: object) -> None:
        callback(*args)
        completed.set()

    extension.on_run_started = on_run_started  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)

    extensions.notify_run_started(run, "automation")

    assert completed.wait(timeout=1)
    callback.assert_called_once_with(run, "automation")
    assert callback.call_args.args[0] is not run


def test_notify_run_started_uses_existing_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schedule callbacks on the service loop instead of blocking the caller."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    callback = Mock()
    completed = threading.Event()

    def on_run_started(*args: object) -> None:
        callback(*args)
        completed.set()

    extension.on_run_started = on_run_started  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)
    loop = asyncio.new_event_loop()
    extensions.set_notification_loop(loop)

    try:

        async def notify_from_running_loop() -> None:
            extensions.notify_run_started(run, "unknown")
            callback.assert_not_called()
            for _ in range(100):
                if completed.is_set():
                    break
                await asyncio.sleep(0.01)

        loop.run_until_complete(notify_from_running_loop())
        assert completed.is_set()
        callback.assert_called_once_with(run, "unknown")
    finally:
        extensions.clear_notification_loop()
        loop.close()


def test_shutdown_drains_submitted_callback_before_clearing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run a ready-queue submission before clearing the notification loop."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    completed = threading.Event()

    def on_run_started(*args: object) -> None:
        del args
        completed.set()

    extension.on_run_started = on_run_started  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)
    loop = asyncio.new_event_loop()
    extensions.set_notification_loop(loop)

    try:

        async def notify_and_shutdown() -> None:
            extensions.notify_run_started(run, "unknown")
            await extensions.shutdown_notification_loop()

        loop.run_until_complete(notify_and_shutdown())
        assert completed.is_set()
    finally:
        extensions.clear_notification_loop()
        loop.close()


def test_callback_completion_does_not_require_closed_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove timed-out callbacks even when their loop closes first."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    callback_started = threading.Event()
    callback_release = threading.Event()
    callback_done = threading.Event()

    def on_run_started(*args: object) -> None:
        del args
        callback_started.set()
        callback_release.wait(timeout=1)
        callback_done.set()

    extension.on_run_started = on_run_started  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)
    monkeypatch.setattr(extensions, "_NOTIFICATION_CALLBACK_TIMEOUT_SECONDS", 0.01)
    loop = asyncio.new_event_loop()
    extensions.set_notification_loop(loop)

    try:

        async def notify_and_shutdown() -> None:
            extensions.notify_run_started(run, "unknown")
            await asyncio.to_thread(callback_started.wait, 1)
            await extensions.shutdown_notification_loop()

        loop.run_until_complete(notify_and_shutdown())
    finally:
        loop.close()

    callback_release.set()
    assert callback_done.wait(timeout=1)
    # pylint: disable=protected-access
    with extensions._NOTIFICATION_CALLBACK_EVENTS_LOCK:
        assert (
            not extensions._NOTIFICATION_CALLBACK_EVENTS
        )  # pylint: disable=protected-access


def test_notify_run_started_isolates_extension_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail an already-created run when an extension raises."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    callback = Mock()
    completed = threading.Event()

    def fail(*args: object) -> None:
        callback(*args)
        completed.set()
        raise RuntimeError

    extension.on_run_started = fail  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)

    extensions.notify_run_started(run, "unknown")

    assert completed.wait(timeout=1)
    callback.assert_called_once_with(run, "unknown")


def test_notify_run_started_isolates_extension_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail an already-created run when extension discovery raises."""
    run = Run.create_empty(42)

    def fail_import() -> ModuleType | None:
        raise ModuleNotFoundError("missing extension dependency")

    monkeypatch.setattr(extensions, "_try_import_sgxt", fail_import)

    extensions.notify_run_started(run, "unknown")
