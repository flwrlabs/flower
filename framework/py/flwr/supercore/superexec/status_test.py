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
"""Tests for optional SuperExec status observers."""

from unittest.mock import Mock

import pytest

from . import status


@pytest.mark.parametrize("missing", ["flwr.ee", "flwr.ee.supercore.superexec_status"])
def test_missing_extension_is_noop(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """An unavailable extension needs no status configuration or cleanup."""
    monkeypatch.setattr(
        status, "import_module", Mock(side_effect=ModuleNotFoundError(name=missing))
    )
    with status.start_superexec_status() as observer:
        observer.set_phase("initializing")
        observer.runtime_available(True)
        observer.close()


@pytest.mark.parametrize("stage", ["import", "start"])
@pytest.mark.parametrize(
    "error", [RuntimeError("startup failed"), ModuleNotFoundError(name="dependency")]
)
def test_installed_extension_failure_propagates(
    monkeypatch: pytest.MonkeyPatch, stage: str, error: Exception
) -> None:
    """Startup and dependency failures must remain visible to the caller."""
    backend = Mock()
    backend.start_superexec_status.side_effect = error
    monkeypatch.setattr(
        status,
        "import_module",
        Mock(side_effect=error) if stage == "import" else Mock(return_value=backend),
    )
    with pytest.raises(type(error)) as raised:
        status.start_superexec_status()
    assert raised.value is error


def test_observer_failures_preserve_application_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notification and cleanup failures cannot replace execution failures."""
    backend = Mock()
    delegate = backend.start_superexec_status.return_value
    delegate.set_phase.side_effect = RuntimeError("phase failed")
    delegate.runtime_available.side_effect = RuntimeError("connection failed")
    delegate.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(status, "import_module", Mock(return_value=backend))
    application_error = ValueError("application failed")
    with pytest.raises(ValueError) as raised:
        with status.start_superexec_status() as observer:
            observer.set_phase("capacity")
            observer.runtime_available(False)
            raise application_error
    assert raised.value is application_error
    observer.close()
    delegate.close.assert_called_once_with()


def test_invocations_have_independent_observers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing one invocation must leave another invocation active."""
    first, second = Mock(), Mock()
    backend = Mock(start_superexec_status=Mock(side_effect=[first, second, None]))
    monkeypatch.setattr(status, "import_module", Mock(return_value=backend))
    with status.start_superexec_status() as one, status.start_superexec_status() as two:
        one.close()
        two.set_phase("acquiring")
    first.close.assert_called_once_with()
    second.set_phase.assert_called_once_with("acquiring")
    second.close.assert_called_once_with()
    with status.start_superexec_status() as disabled:
        disabled.set_phase("initializing")
