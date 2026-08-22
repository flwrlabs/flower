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

from threading import Event
from types import ModuleType
from unittest.mock import Mock

import pytest

from flwr.supercore.run import Run

from . import extensions


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, "grpc", "grpc"),
        ("web_ui", "grpc", "web_ui"),
        (b"automation", "grpc", "automation"),
        ("unrecognized", "http", "unknown"),
        (b"\xff", "http", "unknown"),
    ],
)
def test_resolve_run_start_source(
    value: str | bytes | None,
    default: extensions.RunStartSource,
    expected: extensions.RunStartSource,
) -> None:
    """Normalize caller-provided sources and retain transport defaults."""
    assert extensions.resolve_run_start_source(value, default=default) == expected


def test_notify_run_started_calls_installed_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward successful run creation to an installed extension."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    called = Event()
    callback = Mock(side_effect=lambda *_args: called.set())
    extension.on_run_started = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)

    extensions.notify_run_started(run, "automation")

    assert called.wait(timeout=1)
    callback.assert_called_once_with(run, "automation")
    assert callback.call_args.args[0] is not run


def test_notify_run_started_isolates_extension_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail an already-created run when an extension raises."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    called = Event()

    def fail(*_args: object) -> None:
        called.set()
        raise RuntimeError

    callback = Mock(side_effect=fail)
    extension.on_run_started = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)

    extensions.notify_run_started(run, "http")

    assert called.wait(timeout=1)
    callback.assert_called_once_with(run, "http")


def test_notify_run_started_timeout_does_not_wedge_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue dispatching after an extension callback times out."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    first_called = Event()
    release_first = Event()

    def block(*_args: object) -> None:
        first_called.set()
        release_first.wait()

    extension.on_run_started = block  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)
    monkeypatch.setattr(extensions, "_RUN_STARTED_CALLBACK_TIMEOUT_SECONDS", 0.01)

    extensions.notify_run_started(run, "grpc")
    assert first_called.wait(timeout=1)

    second_called = Event()
    extension.on_run_started = lambda *_args: second_called.set()  # type: ignore[attr-defined]
    extensions.notify_run_started(run, "http")

    assert second_called.wait(timeout=1)
    release_first.set()


def test_notify_run_started_isolates_extension_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail an already-created run when extension discovery raises."""
    run = Run.create_empty(42)

    def fail_import() -> ModuleType | None:
        raise ModuleNotFoundError("missing extension dependency")

    monkeypatch.setattr(extensions, "_try_import_sgxt", fail_import)

    extensions.notify_run_started(run, "grpc")
