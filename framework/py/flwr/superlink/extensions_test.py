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

from pytest import MonkeyPatch, raises

from flwr.supercore.run import Run

from . import extensions


def _install_sgxt_module(monkeypatch: MonkeyPatch, **callbacks: Mock) -> None:
    """Install a fake optional SGXT module for the test."""
    ee_module = ModuleType("flwr.ee")
    ee_module.__dict__["__path__"] = []
    superlink_module = ModuleType("flwr.ee.superlink")
    superlink_module.__dict__["__path__"] = []
    module = ModuleType("flwr.ee.superlink.extensions")
    for name, callback in callbacks.items():
        setattr(module, name, callback)

    ee_module.__dict__["superlink"] = superlink_module
    superlink_module.__dict__["extensions"] = module
    monkeypatch.setitem(sys.modules, "flwr.ee", ee_module)
    monkeypatch.setitem(sys.modules, "flwr.ee.superlink", superlink_module)
    monkeypatch.setitem(sys.modules, "flwr.ee.superlink.extensions", module)


def _block_sgxt_import(monkeypatch: MonkeyPatch) -> None:
    """Make imports of the optional SGXT package fail as unavailable."""
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
    _block_sgxt_import(monkeypatch)

    extensions.notify_run_started(Run.create_empty(123), "unknown")


def test_notify_run_started_isolates_callback_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep a persisted run successful when the extension callback fails."""
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


def test_configure_app_uses_sgxt_callback(monkeypatch: MonkeyPatch) -> None:
    """Delegate application configuration to the optional extension."""
    callback = Mock()
    _install_sgxt_module(monkeypatch, configure_app=callback)
    app = Mock()

    extensions.configure_app(app)

    callback.assert_called_once_with(app)


def test_get_middleware_uses_sgxt_callback(monkeypatch: MonkeyPatch) -> None:
    """Return middleware supplied by the optional extension."""
    callback = Mock(return_value=())
    _install_sgxt_module(monkeypatch, get_middleware=callback)

    assert extensions.get_middleware() == ()
    callback.assert_called_once_with()


def test_get_lifespan_contexts_uses_sgxt_callback(monkeypatch: MonkeyPatch) -> None:
    """Return lifespan contexts supplied by the optional extension."""
    callback = Mock(return_value=())
    _install_sgxt_module(monkeypatch, get_lifespan_contexts=callback)

    assert extensions.get_lifespan_contexts() == ()
    callback.assert_called_once_with()


def test_optional_configuration_bridges_skip_missing_extension(
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep optional configuration bridges as no-ops without SGXT."""
    _block_sgxt_import(monkeypatch)

    extensions.configure_app(Mock())
    assert extensions.get_middleware() == ()
    assert extensions.get_lifespan_contexts() == ()


def test_import_errors_without_a_module_name_are_not_ignored(
    monkeypatch: MonkeyPatch,
) -> None:
    """Do not hide import failures whose source cannot be identified."""
    import_function = builtins.__import__

    def fail_sgxt_import(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("flwr.ee"):
            raise ImportError("Unexpected extension import failure", name=None)
        return import_function(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_sgxt_import)

    with raises(ImportError, match="Unexpected extension import failure"):
        extensions.get_middleware()
