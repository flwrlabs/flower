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

from types import ModuleType
from unittest.mock import Mock

from pytest import MonkeyPatch, mark

from flwr.supercore.run import Run

from . import extensions


@mark.parametrize(
    ("value", "expected"),
    [
        (None, "unknown"),
        ("web_ui", "web_ui"),
        (b"automation", "automation"),
        ("not-a-source", "unknown"),
        (b"\\xff", "unknown"),
    ],
)
def test_resolve_run_start_source(
    value: str | bytes | None, expected: extensions.RunStartSource
) -> None:
    """Normalize best-effort caller attribution without treating it as auth."""
    assert extensions.resolve_run_start_source(value) == expected


def test_notify_run_started_passes_a_snapshot_to_the_extension(
    monkeypatch: MonkeyPatch,
) -> None:
    """Pass a copy of the persisted run to the optional extension."""
    callback = Mock()
    module = ModuleType("flwr.ee.superlink.extensions")
    module.on_run_started = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: module)
    run = Run.create_empty(123)

    extensions.notify_run_started(run, "unknown")

    callback.assert_called_once()
    notified_run, source = callback.call_args.args
    assert notified_run == run
    assert notified_run is not run
    assert source == "unknown"


def test_notify_run_started_skips_missing_extension(monkeypatch: MonkeyPatch) -> None:
    """Do nothing when the optional extension package is absent."""
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: None)

    extensions.notify_run_started(Run.create_empty(123), "unknown")


def test_notify_run_started_isolates_extension_import_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Keep a persisted run successful when extension discovery fails."""

    def fail_import() -> ModuleType | None:
        raise RuntimeError("extension import failed")

    monkeypatch.setattr(extensions, "_try_import_sgxt", fail_import)

    extensions.notify_run_started(Run.create_empty(123), "unknown")


def test_notify_result_delivered_passes_a_snapshot_to_the_extension(
    monkeypatch: MonkeyPatch,
) -> None:
    """Pass a copy of the accepted run to the optional extension."""
    callback = Mock()
    module = ModuleType("flwr.ee.superlink.extensions")
    module.on_result_delivered = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: module)
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
    module = ModuleType("flwr.ee.superlink.extensions")
    module.on_result_delivered = callback  # type: ignore[attr-defined]
    monkeypatch.setattr(extensions, "_try_import_sgxt", lambda: module)

    extensions.notify_result_delivered(Run.create_empty(123), "account-123", "chat")
