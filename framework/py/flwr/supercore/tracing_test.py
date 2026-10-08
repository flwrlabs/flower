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

from flwr.proto.runtime_pb2 import CreateTaskRequest  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.model_message import ModelRequest
from flwr.supercore.servicer.runtime import runtime_handlers
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.task_process.model import task as model_task

from . import tracing

_CARRIER = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"


@pytest.fixture(autouse=True)
def reset_backend_cache() -> Iterator[None]:
    """Prevent optional backend discovery from leaking between tests."""
    loader = tracing._load_backend
    loader.cache_clear()
    yield
    loader.cache_clear()


@pytest.mark.parametrize("enabled", ["", "0", "true"])
def test_disabled_tracing_does_not_import_backend(
    monkeypatch: pytest.MonkeyPatch, enabled: str
) -> None:
    """Disabled tracing is inert even when a carrier is supplied."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", enabled)
    discover = Mock(side_effect=AssertionError("must not import"))
    monkeypatch.setattr(tracing, "import_module", discover)
    with tracing.trace_span("test", traceparent=_CARRIER) as span:
        span.set_attribute("flwr.task_id", "1")
        span.add_event("provider.first_text")
        assert tracing.current_traceparent() == ""
    tracing.flush_traces()
    discover.assert_not_called()


def test_missing_or_broken_backend_is_cached_and_inert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend import errors do not fail or repeatedly import in task work."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    discover = Mock(side_effect=RuntimeError("private import detail"))
    monkeypatch.setattr(tracing, "import_module", discover)
    with tracing.trace_span("test"):
        assert tracing.current_traceparent() == ""
    tracing.flush_traces()
    discover.assert_called_once_with("flwr.ee.supercore.tracing")


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


def test_provider_failure_exports_only_a_fixed_event_and_error_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider body and exception text never reach the tracing backend."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    monkeypatch.setattr(TaskIdentity, "_task_id", 2)
    monkeypatch.setattr(TaskIdentity, "_run_id", 3)
    monkeypatch.setattr(TaskIdentity, "_node_id", 1)
    backend = Mock()
    manager = MagicMock()
    raw_span = Mock()
    manager.__enter__.return_value = raw_span
    backend.trace_span.return_value = manager
    monkeypatch.setattr(tracing, "_load_backend", lambda: backend)
    request = ModelRequest(
        dst_task_id=2, input_="prompt-secret", model="model", stream=True
    )
    request.metadata.src_task_id = 1
    request.metadata.__dict__["_message_id"] = "request-id"
    monkeypatch.setattr(model_task, "_pull_model_request", Mock(return_value=request))
    error = RuntimeError("provider-body-secret")
    monkeypatch.setattr(model_task, "invoke_model_provider", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        model_task.handle_task(Mock())
    assert caught.value is error
    raw_span.add_event.assert_called_once_with("provider.error")
    raw_span.set_attribute.assert_called_once_with("error.type", "RuntimeError")
    manager.__exit__.assert_called_once_with(None, None, None)
    assert "secret" not in str(backend.mock_calls)
    assert "secret" not in str(raw_span.mock_calls)


@pytest.mark.parametrize("current", ["", _CARRIER[:-2] + "03"])
def test_child_task_uses_current_context_or_authenticated_parent(
    monkeypatch: pytest.MonkeyPatch,
    current: str,
) -> None:
    """Model tasks inherit their parent when no current span is available."""
    state = Mock()
    state.create_task.return_value = 2
    parent = Task(task_id=1, run_id=3, type=TaskType.AGENT_APP, traceparent=_CARRIER)
    monkeypatch.setattr(runtime_handlers, "current_traceparent", lambda: current)
    runtime_handlers.create_task(
        CreateTaskRequest(type=TaskType.MODEL, model_ref="model"), state, parent
    )
    assert state.create_task.call_args.kwargs["traceparent"] == (current or _CARRIER)
