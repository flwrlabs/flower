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
"""Tests for the model task handler's streaming event batching."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import cast
from unittest.mock import MagicMock

import pytest

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskMessageResponse,
    PushTaskEventsRequest,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject, JSONValue

from . import task as task_module
from .provider import ModelProviderError

_PROGRESS_EVENT_TYPE = "response.flower_fusion.progress"
_TEXT_EVENT_TYPE = "response.output_text.delta"
_REASONING_EVENT_TYPE = "response.reasoning_summary_part.added"
_TOOL_EVENT_TYPE = "response.function_call_arguments.delta"

_ORDINARY_TYPES = (_TEXT_EVENT_TYPE, _REASONING_EVENT_TYPE, _TOOL_EVENT_TYPE)


def _progress_event(**extras: JSONValue) -> JSONObject:
    """Return a Fusion progress event."""
    event: JSONObject = {
        "type": _PROGRESS_EVENT_TYPE,
        "stage": "routing",
        "status": "in_progress",
        "message": "Choosing the best approach...",
        "elapsed_ms": 1,
    }
    event.update(cast(JSONObject, extras))
    return event


def _ordinary_event(type_: str = _TEXT_EVENT_TYPE, index: int = 0) -> JSONObject:
    """Return an ordinary batchable stream event."""
    event: JSONObject = {"type": type_, "delta": f"t{index}"}
    return event


def _terminal_event(event_type: str = "response.completed") -> JSONObject:
    """Return a terminal success event."""
    event: JSONObject = {
        "type": event_type,
        "response": {"id": "resp_1", "status": "completed"},
    }
    return event


def _make_request(
    *,
    stream: bool = True,
    src_task_id: int = 7,
    message_id: str = "msg_1",
) -> ModelRequest:
    """Create a model request and set its source task metadata."""
    request = ModelRequest(
        dst_task_id=123,
        input_=[{"role": "user", "content": "hello"}],
        model="flower-fusion",
        stream=stream,
    )
    request.metadata.src_task_id = src_task_id
    request.metadata.__dict__["_message_id"] = message_id
    return request


def _make_pull_response(request: ModelRequest) -> PullTaskMessageResponse:
    """Wrap a model request in a PullTaskMessage response."""
    return PullTaskMessageResponse(messages=[message_to_proto(request)])


def _mocked_stub(request: ModelRequest) -> MagicMock:
    """Return a stub whose first PullTaskMessage yields ``request``."""
    stub = MagicMock()
    stub.PullTaskMessage.return_value = _make_pull_response(request)
    return stub


def _monkeypatch_invoke(
    monkeypatch: pytest.MonkeyPatch,
    events: Sequence[JSONObject],
    response: JSONObject | None = None,
    exception: BaseException | None = None,
) -> None:
    """Replace ``invoke_model_provider`` with a no-network callable."""

    def _fake_invoke(
        _request_payload: JSONObject,
        *,
        on_stream_event: Callable[[JSONObject], None] | None = None,
    ) -> JSONObject:
        for event in events:
            if on_stream_event is not None:
                on_stream_event(event)
        if exception is not None:
            raise exception
        if response is not None:
            return response
        return cast(
            JSONObject,
            {"id": "resp_1", "object": "response", "status": "completed"},
        )

    monkeypatch.setattr(task_module, "invoke_model_provider", _fake_invoke)


def _event_type(event: TaskEvent) -> str:
    """Return the event type from a serialized TaskEvent."""
    return event.event


def _event_data(event: TaskEvent) -> JSONObject:
    """Return the parsed data payload from a serialized TaskEvent."""
    return cast(JSONObject, json.loads(event.data))


def _response_payload_from_push_call(stub: MagicMock) -> JSONObject:
    """Return the parsed ModelResponse payload from the PushTaskMessage call."""
    message_proto = stub.PushTaskMessage.call_args.args[0].message
    message = ModelResponse.from_message(message_from_proto(message_proto))
    return message.payload


def _assert_push_calllists(
    stub: MagicMock,
    expected_groups: Sequence[Sequence[JSONObject]],
) -> None:
    """Assert PushTaskEvents was called with groups of events in order."""
    calls = stub.PushTaskEvents.call_args_list
    assert len(calls) == len(expected_groups)
    for call, group in zip(calls, expected_groups, strict=True):
        request = call.args[0]
        assert isinstance(request, PushTaskEventsRequest)
        assert len(request.events) == len(group)
        for actual_event, expected_event in zip(request.events, group, strict=True):
            assert _event_type(actual_event) == expected_event["type"]
            assert _event_data(actual_event) == expected_event


@pytest.mark.parametrize("stream", [True, False])
def test_no_push_task_events_for_non_stream_or_empty_stream(
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """No streaming events should produce no PushTaskEvents calls."""
    request = _make_request(stream=stream)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(
        monkeypatch,
        events=[],
        response={"id": "resp_1", "object": "response", "status": "completed"},
    )

    if stream:
        task_module.handle_task(stub, task_id=5, run_id=1)
    else:
        task_module.handle_task(stub, task_id=5, run_id=1)

    stub.PushTaskEvents.assert_not_called()
    stub.PushTaskMessage.assert_called_once()


def test_single_progress_event_flushes_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single progress event must be pushed immediately."""
    event = _progress_event()
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=[event])

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [[event]])


def test_two_progress_events_flush_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each progress event must flush its own batch without coalescing."""
    p1 = _progress_event(elapsed_ms=1)
    p2 = _progress_event(elapsed_ms=2)
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=[p1, p2])

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [[p1], [p2]])


def test_sixteen_ordinary_events_flush_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exactly 16 ordinary events flush as one batch."""
    events = [_ordinary_event(index=i) for i in range(16)]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events)

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [events])


def test_fifteen_ordinary_events_flush_on_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial batch of 15 ordinary events flushes after the stream ends."""
    events = [_ordinary_event(index=i) for i in range(15)]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events)

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [events])


def test_progress_precedes_ordinary_partial_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress flushes first and any following ordinary events are appended."""
    p = _progress_event()
    t = _ordinary_event(index=0)
    p2 = _progress_event(elapsed_ms=2)
    events = [p, t, p2]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events)

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [[p], [t, p2]])


def test_fifteen_ordinary_plus_progress_coalesce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress appended to an existing partial batch coalesces and flushes."""
    ordinary = [_ordinary_event(index=i) for i in range(15)]
    p = _progress_event()
    events = [*ordinary, p]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events)

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [[*ordinary, p]])


def test_progress_then_ordinary_then_exception_flushes_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress flushes immediately; remaining ordinary events flush finally."""
    p = _progress_event()
    t = _ordinary_event(index=0)
    events = [p, t]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(
        monkeypatch,
        events=events,
        exception=ModelProviderError(detail={"message": "boom"}),
    )

    with pytest.raises(ModelProviderError):
        task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [[p], [t]])


def test_normal_events_do_not_flush_before_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer than 16 normal events produce no push until the terminal flush."""
    events = [_ordinary_event(index=i) for i in range(15)]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events)

    task_module.handle_task(stub, task_id=5, run_id=1)

    assert len(stub.PushTaskEvents.call_args_list) == 1
    _assert_push_calllists(stub, [events])


@pytest.mark.parametrize("event_type", _ORDINARY_TYPES)
def test_varied_ordinary_event_types_batch_normally(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
) -> None:
    """Text, reasoning, and tool events should be batchable."""
    events = [_ordinary_event(type_=event_type, index=i) for i in range(16)]
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events)

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [events])


def test_terminal_response_is_pushed_after_stream_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed stream should push the final response after flushing events."""
    events = [_ordinary_event(index=0), _terminal_event("response.completed")]
    final_response = cast(
        JSONObject, {"id": "resp_1", "object": "response", "status": "completed"}
    )
    request = _make_request(stream=True, src_task_id=7, message_id="msg_1")
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=events, response=final_response)

    task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [events])
    assert _response_payload_from_push_call(stub) == final_response


def test_exception_propagates_and_produces_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider errors should be re-raised and pushed as a failed response."""
    detail: JSONObject = {"message": "internal provider error"}
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(
        monkeypatch,
        events=[],
        exception=ModelProviderError(detail=detail),
    )

    with pytest.raises(ModelProviderError):
        task_module.handle_task(stub, task_id=5, run_id=1)

    parsed = _response_payload_from_push_call(stub)
    assert parsed["status"] == "failed"
    error = cast(JSONObject, parsed["error"])
    assert error["code"] == "model_provider_error"


def test_cancellation_flushes_partial_then_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation must flush partial batches and propagate unchanged.

    A progress event flushes immediately; a following ordinary event stays
    buffered until the cancellation triggers the ``finally`` flush. No
    error/terminal response should be pushed.
    """
    p = _progress_event()
    t = _ordinary_event(index=0)
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(
        monkeypatch,
        events=[p, t],
        exception=asyncio.CancelledError("request cancelled"),
    )

    with pytest.raises(asyncio.CancelledError, match="request cancelled"):
        task_module.handle_task(stub, task_id=5, run_id=1)

    _assert_push_calllists(stub, [[p], [t]])
    stub.PushTaskMessage.assert_not_called()


def test_payload_serialization_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Event data must equal compact JSON serialization of the original event."""
    event: JSONObject = {
        "type": _TEXT_EVENT_TYPE,
        "delta": "hello",
        "metadata": {"index": 1},
    }
    request = _make_request(stream=True)
    stub = _mocked_stub(request)
    _monkeypatch_invoke(monkeypatch, events=[event])

    task_module.handle_task(stub, task_id=5, run_id=1)

    pushed = stub.PushTaskEvents.call_args.args[0].events[0]
    # Compact JSON: keys in insertion order, no spaces, ascii-safe.
    expected = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
    assert pushed.data == expected
