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
"""Tests for optional tracing and task-carried process continuity."""

# pylint: disable=protected-access

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from flwr.common.serde import message_to_proto
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    PullTaskInputRequest,
    PullTaskInputResponse,
)
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.fab import Fab
from flwr.supercore.json_message.model_message import ModelRequest
from flwr.supercore.object_store.in_memory_object_store import InMemoryObjectStore
from flwr.supercore.routers.runtime.responses import _start_exchange
from flwr.supercore.servicer.runtime.runtime_handlers import acquire_task
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.task_process.model import task as model_task
from flwr.superlink.servicer.runtime.runtime_handlers import pull_task_input

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
        _CARRIER[:-2] + "02",
        "00-" + "0" * 32 + "-" + "2" * 16 + "-01",
        "00-" + "1" * 32 + "-" + "0" * 16 + "-01",
    ],
)
def test_invalid_traceparent_is_discarded(value: object) -> None:
    """Only bounded version-00 carriers with nonzero IDs are accepted."""
    assert tracing.validate_traceparent(value) == ""


@pytest.mark.parametrize("flags", ["00", "01"])
def test_valid_traceparent(flags: str) -> None:
    """Sampled and unsampled version-00 contexts round trip unchanged."""
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


def test_task_carrier_connects_run_dispatch_app_and_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported trace spans persisted parent/child tasks and first text."""
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    active: ContextVar[str] = ContextVar("traceparent", default="")
    records: list[dict[str, Any]] = []
    backend = ModuleType("fake_tracing")

    @contextmanager
    def capture(
        name: str, *, traceparent: str, attributes: dict[str, object]
    ) -> Iterator[Mock]:
        parent = traceparent or active.get()
        trace_id = parent.split("-")[1] if parent else "f" * 32
        carrier = f"00-{trace_id}-{len(records) + 1:016x}-01"
        raw_span = Mock()
        records.append(
            {
                "name": name,
                "carrier": carrier,
                "attributes": attributes,
                "span": raw_span,
            }
        )
        token = active.set(carrier)
        try:
            yield raw_span
        finally:
            active.reset(token)

    backend.__dict__.update(
        trace_span=capture,
        current_traceparent=active.get,
        flush_traces=Mock(),
    )
    monkeypatch.setattr(tracing, "_load_backend", lambda: backend)
    state = InMemoryLinkState(Mock(), InMemoryObjectStore())
    with tracing.trace_span("run.create"):
        run_id = state.create_run(
            "flwr/test",
            "1",
            state.store_fab(Fab("", b"fab", {})),
            {},
            "@me/test",
            None,
            "account",
            TaskType.AGENT_APP,
            traceparent=tracing.current_traceparent(),
        )
    acquired = acquire_task(
        AcquireTaskRequest(supported_task_types=[TaskType.AGENT_APP]), state
    )
    task_input = pull_task_input(PullTaskInputRequest(), state, acquired.task)
    restored = PullTaskInputResponse.FromString(task_input.SerializeToString())
    assert restored.traceparent == acquired.task.traceparent
    with tracing.trace_span("task.dispatch", traceparent=acquired.task.traceparent):
        pass
    with tracing.trace_span("agentapp.execute", traceparent=restored.traceparent):
        exchange = _start_exchange(
            state,
            acquired.task,
            {"model": "model", "input": "prompt-secret", "stream": True},
        )
    child = state.get_tasks(task_ids=[exchange.model_task_id])[0]
    assert child.traceparent != acquired.task.traceparent
    assert child.traceparent.split("-")[1] == acquired.task.traceparent.split("-")[1]
    monkeypatch.setattr(TaskIdentity, "_task_id", child.task_id)
    monkeypatch.setattr(TaskIdentity, "_run_id", run_id)
    monkeypatch.setattr(TaskIdentity, "_node_id", 1)
    client = Mock()
    client.PullTaskMessage.return_value.messages = []
    client.PullTaskMessage.return_value.messages = [
        message_to_proto(
            state.get_task_message(dst_task_ids=[child.task_id], limit=1)[0]
        )
    ]

    def provider(_request: object, **kwargs: Any) -> dict[str, Any]:
        emit = kwargs["on_stream_event"]
        emit(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "reasoning-secret",
            }
        )
        for _ in range(20):
            emit({"type": "response.output_text.delta", "delta": "output-secret"})
        return {"object": "response", "status": "completed", "output": []}

    monkeypatch.setattr(model_task, "invoke_model_provider", provider)
    with tracing.trace_span("model.execute", traceparent=child.traceparent):
        model_task.handle_task(client)
    assert len({record["carrier"].split("-")[1] for record in records}) == 1
    provider_span = records[-1]["span"]
    assert [call.args for call in provider_span.add_event.call_args_list] == [
        ("provider.first_text",),
        ("provider.completed",),
    ]
    assert "secret" not in str(records)
    assert tracing.current_traceparent() == ""
