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
    extension.on_run_started = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)

    extensions.notify_run_started(run, "automation")

    callback.assert_called_once_with(run, "automation")
    assert callback.call_args.args[0] is not run


def test_notify_run_started_isolates_extension_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not fail an already-created run when an extension raises."""
    run = Run.create_empty(42)
    extension = ModuleType("test_extension")
    callback = Mock(side_effect=RuntimeError)
    extension.on_run_started = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: extension)

    extensions.notify_run_started(run, "unknown")

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
