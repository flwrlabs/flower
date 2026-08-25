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
"""Tests for SuperLink extension notifications."""

import builtins
import sys
from collections.abc import Mapping
from types import ModuleType
from unittest.mock import Mock

from pytest import MonkeyPatch

from flwr.supercore.run import Run

from . import extensions


def _install_sgxt_module(monkeypatch: MonkeyPatch, **callbacks: Mock) -> None:
    """Install a fake optional SGXT module for the test."""
    module = ModuleType("flwr.ee.superlink.extensions")
    for name, callback in callbacks.items():
        setattr(module, name, callback)

    monkeypatch.setitem(sys.modules, "flwr.ee", ModuleType("flwr.ee"))
    monkeypatch.setitem(
        sys.modules,
        "flwr.ee.superlink",
        ModuleType("flwr.ee.superlink"),
    )
    monkeypatch.setitem(sys.modules, "flwr.ee.superlink.extensions", module)


def test_notify_run_started_passes_a_snapshot_to_the_extension(
    monkeypatch: MonkeyPatch,
) -> None:
    """Pass a copy of the persisted run to the optional extension."""
    callback = Mock()
    _install_sgxt_module(monkeypatch, on_run_started=callback)
    run = Run.create_empty(123)

    extensions.notify_run_started(run, "unknown")

    callback.assert_called_once()
    notified_run, source = callback.call_args.args
    assert notified_run == run
    assert notified_run is not run
    assert source == "unknown"


def test_notify_run_started_skips_missing_extension(monkeypatch: MonkeyPatch) -> None:
    """Do nothing when the optional extension package is absent."""
    import_function = builtins.__import__

    def fail_sgxt_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("flwr.ee"):
            raise ModuleNotFoundError(
                "SuperGrid Extensions is unavailable", name="flwr.ee"
            )
        return import_function(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_sgxt_import)

    extensions.notify_run_started(Run.create_empty(123), "unknown")


def test_notify_run_started_isolates_extension_import_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep a persisted run successful when extension discovery fails."""
    callback = Mock(side_effect=RuntimeError("extension failed"))
    _install_sgxt_module(monkeypatch, on_run_started=callback)

    extensions.notify_run_started(Run.create_empty(123), "unknown")
    callback.assert_called_once()


def test_notify_result_delivered_passes_a_snapshot_to_the_extension(
    monkeypatch: MonkeyPatch,
) -> None:
    """Pass a copy of the accepted run to the optional extension."""
    callback = Mock()
    _install_sgxt_module(monkeypatch, on_result_delivered=callback)
    run = Run.create_empty(123)

    extensions.notify_result_delivered(run, "account-123", "logs")

    callback.assert_called_once()
    notified_run, flwr_aid, channel = callback.call_args.args
    assert notified_run == run
    assert notified_run is not run
    assert flwr_aid == "account-123"
    assert channel == "logs"


def test_notify_result_delivered_isolates_extension_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep an accepted result request successful when the extension fails."""
    callback = Mock(side_effect=RuntimeError("extension failed"))
    _install_sgxt_module(monkeypatch, on_result_delivered=callback)

    extensions.notify_result_delivered(Run.create_empty(123), "account-123", "chat")
    callback.assert_called_once()
