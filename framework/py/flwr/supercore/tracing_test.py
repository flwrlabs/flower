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
"""Tests for optional tracing and task context propagation."""

# pylint: disable=protected-access

from collections.abc import Iterator
from unittest.mock import MagicMock, Mock

import pytest

from . import tracing

_CARRIER = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"


@pytest.fixture(autouse=True)
def reset_backend_cache() -> Iterator[None]:
    """Prevent optional backend discovery from leaking between tests."""
    loader = tracing._load_backend
    loader.cache_clear()
    yield
    loader.cache_clear()


@pytest.mark.parametrize("enabled", ["", "1"])
def test_disabled_or_unavailable_backend_is_inert(
    monkeypatch: pytest.MonkeyPatch, enabled: str
) -> None:
    """Tracing is optional and failed backend discovery is cached."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", enabled)
    discover = Mock(side_effect=RuntimeError("private import detail"))
    monkeypatch.setattr(tracing, "import_module", discover)
    with tracing.trace_span("test", traceparent=_CARRIER) as span:
        span.set_attribute("flwr.task_id", "1")
        span.add_event("provider.first_text")
        assert tracing.current_traceparent() == ""
    tracing.flush_traces()
    assert discover.call_count == int(enabled == "1")


@pytest.mark.parametrize(
    "failure", ["enter", "exit", "attribute", "event", "carrier", "flush"]
)
def test_backend_failures_preserve_application_exception(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Tracing errors cannot replace application errors or capture their text."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    backend = Mock()
    manager = MagicMock()
    raw_span = Mock()
    manager.__enter__.return_value = raw_span
    backend.trace_span.return_value = manager
    backend.current_traceparent.return_value = _CARRIER
    method = {
        "enter": manager.__enter__,
        "exit": manager.__exit__,
        "attribute": raw_span.set_attribute,
        "event": raw_span.add_event,
        "carrier": backend.current_traceparent,
        "flush": backend.flush_traces,
    }[failure]
    method.side_effect = RuntimeError("backend-secret")
    monkeypatch.setattr(tracing, "_load_backend", lambda: backend)
    error = ValueError("application-secret")
    with pytest.raises(ValueError) as caught:
        with tracing.trace_span("test", traceparent="invalid") as span:
            span.set_attribute("flwr.task_id", "1")
            span.add_event("provider.first_text")
            tracing.current_traceparent()
            raise error
    assert caught.value is error
    tracing.flush_traces()
    assert backend.trace_span.call_args.kwargs["traceparent"] == ""
    if failure != "enter":
        manager.__exit__.assert_called_once_with(None, None, None)
        assert "application-secret" not in str(raw_span.mock_calls)
        assert raw_span.set_attribute.call_args.args == ("error.type", "ValueError")


@pytest.mark.parametrize(
    "value",
    [
        None,
        3,
        "",
        _CARRIER + "\n",
        _CARRIER.replace("1", "A", 1),
        _CARRIER.replace("00-", "01-", 1),
        _CARRIER[:-2] + "0A",
        _CARRIER[:-2] + "gg",
        _CARRIER[:-2] + "001",
        "00-" + "0" * 32 + "-" + "2" * 16 + "-01",
        "00-" + "1" * 32 + "-" + "0" * 16 + "-01",
    ],
)
def test_invalid_traceparent_is_discarded(value: object) -> None:
    """Only bounded version-00 carriers with nonzero IDs are accepted."""
    assert tracing.validate_traceparent(value) == ""


@pytest.mark.parametrize("flags", ["00", "01", "02", "03", "80", "ff"])
def test_valid_traceparent(flags: str) -> None:
    """The complete version-00 trace-flags byte round trips unchanged."""
    carrier = _CARRIER[:-2] + flags
    assert tracing.validate_traceparent(carrier) == carrier


def test_context_storage_failure_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    """Optional storage cannot affect task execution or return invalid metadata."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    backend = Mock()
    backend.store_task_context.side_effect = RuntimeError("private database detail")
    backend.load_task_context.side_effect = RuntimeError("private database detail")
    monkeypatch.setattr(tracing, "_load_backend", lambda: backend)
    state = object()
    tracing.store_task_context(state, 1, _CARRIER)
    assert tracing.load_task_context(state, 1) == ""
    tracing.store_task_context(state, 1, "invalid")
    backend.store_task_context.assert_called_once_with(state, 1, _CARRIER)


def test_disabled_storage_does_not_discover_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled tracing does not import an SDK or access optional storage."""
    monkeypatch.delenv("FLWR_TRACING_ENABLED", raising=False)
    discover = Mock()
    monkeypatch.setattr(tracing, "_load_backend", discover)
    tracing.store_task_context(object(), 1, _CARRIER)
    assert tracing.load_task_context(object(), 1) == ""
    discover.assert_not_called()
