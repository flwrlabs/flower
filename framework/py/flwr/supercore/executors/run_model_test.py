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
"""Tests for the `flwr-model` executor."""


from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from flwr.app.metadata import Metadata
from flwr.common import ConfigRecord, RecordDict
from flwr.common.constant import SubStatus
from flwr.common.message import make_message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputResponse,
    PullTaskMessageResponse,
    PushRunEventsRequest,
    PushTaskMessageRequest,
    PushTaskOutputRequest,
)
from flwr.supercore.executors.model_provider import (
    ModelProviderError,
    ModelProviderResult,
)
from flwr.supercore.executors.run_model import (
    MODEL_COMPLETED_EVENT,
    MODEL_FAILED_EVENT,
    MODEL_OUTPUT_DELTA_EVENT,
    MODEL_STARTED_EVENT,
    _run_model_task,
)
from flwr.supercore.task_message import (
    JsonObject,
    ModelTaskMessage,
    ModelTaskResultMessage,
)

SOURCE_TASK_ID = 101
MODEL_TASK_ID = 202
REQUEST_MESSAGE_ID = "request-message-id"


class FakeServerAppIoStub:
    """Fake ServerAppIo stub for model executor tests."""

    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.calls: list[str] = []
        self.run_events: list[tuple[str, dict[str, object]]] = []
        self.pushed_messages: list[Any] = []
        self.task_outputs: list[PushTaskOutputRequest] = []
        self.pull_limits: list[int] = []

    def PullTaskInput(self, request: object) -> PullTaskInputResponse:
        """Return model task input."""
        del request
        self.calls.append("PullTaskInput")
        return PullTaskInputResponse(task_id=MODEL_TASK_ID)

    def PullTaskMessage(self, request: Any) -> PullTaskMessageResponse:
        """Return configured task messages."""
        self.calls.append("PullTaskMessage")
        self.pull_limits.append(request.limit)
        return PullTaskMessageResponse(messages=self.messages)

    def PushRunEvents(self, request: PushRunEventsRequest) -> object:
        """Capture run events."""
        self.calls.append("PushRunEvents")
        for event in request.events:
            self.run_events.append((event.event, json.loads(event.data)))
        return object()

    def PushTaskMessage(self, request: PushTaskMessageRequest) -> object:
        """Capture task messages."""
        self.calls.append("PushTaskMessage")
        self.pushed_messages.append(request.message)
        return object()

    def PushTaskOutput(self, request: PushTaskOutputRequest) -> object:
        """Capture task output."""
        self.calls.append("PushTaskOutput")
        self.task_outputs.append(request)
        return object()


def test_run_model_task_pushes_success_result() -> None:
    """Test model task success request/reply flow."""
    stub = FakeServerAppIoStub([_model_request_proto(stream=False)])

    def invoke_model(
        request: JsonObject, on_stream_event: Callable[[JsonObject], None] | None
    ) -> ModelProviderResult:
        del on_stream_event
        assert request["model"] == "gpt-4.1-mini"
        return ModelProviderResult(
            response={
                "id": "resp-1",
                "output": [{"type": "message", "content": "hello"}],
                "usage": {"input_tokens": 1},
            },
            events=[],
        )

    _run_model_task(stub, invoke_model)

    result = _pushed_model_result(stub)
    assert stub.calls[:2] == ["PullTaskInput", "PullTaskMessage"]
    assert stub.pull_limits == [2]
    assert [event for event, _ in stub.run_events] == [
        MODEL_STARTED_EVENT,
        MODEL_COMPLETED_EVENT,
    ]
    assert result.dst_task_id == SOURCE_TASK_ID
    assert result.reply_to_message_id == REQUEST_MESSAGE_ID
    assert result.payload["response_id"] == "resp-1"
    assert result.payload["usage"] == {"input_tokens": 1}
    assert stub.task_outputs[-1].sub_status == SubStatus.COMPLETED
    assert stub.task_outputs[-1].details == ""


def test_run_model_task_emits_stream_events() -> None:
    """Test provider stream events are forwarded as run events."""
    stub = FakeServerAppIoStub([_model_request_proto(stream=True)])

    def invoke_model(
        request: JsonObject, on_stream_event: Callable[[JsonObject], None] | None
    ) -> ModelProviderResult:
        assert request["stream"] is True
        assert on_stream_event is not None
        on_stream_event({"type": "response.output_text.delta", "delta": "hi"})
        return ModelProviderResult(
            response={"id": "resp-1"},
            events=[{"type": "response.output_text.delta", "delta": "hi"}],
        )

    _run_model_task(stub, invoke_model)

    assert [event for event, _ in stub.run_events] == [
        MODEL_STARTED_EVENT,
        MODEL_OUTPUT_DELTA_EVENT,
        MODEL_COMPLETED_EVENT,
    ]
    assert stub.run_events[1][1]["event"] == {
        "type": "response.output_text.delta",
        "delta": "hi",
    }


def test_run_model_task_pushes_error_result_on_provider_failure() -> None:
    """Test provider failures are returned to the source task."""
    stub = FakeServerAppIoStub([_model_request_proto(stream=False)])

    def invoke_model(
        request: JsonObject, on_stream_event: Callable[[JsonObject], None] | None
    ) -> ModelProviderResult:
        del request, on_stream_event
        raise ModelProviderError({"type": "provider_error", "message": "down"})

    _run_model_task(stub, invoke_model)

    result = _pushed_model_result(stub)
    assert [event for event, _ in stub.run_events] == [
        MODEL_STARTED_EVENT,
        MODEL_FAILED_EVENT,
    ]
    assert result.payload["response"] == {"status": "failed"}
    assert result.payload["error"] == {"type": "provider_error", "message": "down"}
    assert stub.task_outputs[-1].sub_status == SubStatus.FAILED
    assert stub.task_outputs[-1].details == "down"


def test_run_model_task_fails_when_request_count_is_not_one() -> None:
    """Test missing model request fails the task without pushing a reply."""
    stub = FakeServerAppIoStub([])

    def invoke_model(
        request: JsonObject, on_stream_event: Callable[[JsonObject], None] | None
    ) -> ModelProviderResult:
        del request, on_stream_event
        raise AssertionError("provider must not be called")

    _run_model_task(stub, invoke_model)

    assert stub.pushed_messages == []
    assert [event for event, _ in stub.run_events] == [MODEL_FAILED_EVENT]
    assert stub.task_outputs[-1].sub_status == SubStatus.FAILED
    assert "Expected exactly one model task message" in stub.task_outputs[-1].details


def test_run_model_task_fails_when_request_count_is_more_than_one() -> None:
    """Test multiple model requests fail the task without pushing a reply."""
    stub = FakeServerAppIoStub(
        [_model_request_proto(stream=False), _model_request_proto(stream=False)]
    )

    def invoke_model(
        request: JsonObject, on_stream_event: Callable[[JsonObject], None] | None
    ) -> ModelProviderResult:
        del request, on_stream_event
        raise AssertionError("provider must not be called")

    _run_model_task(stub, invoke_model)

    assert stub.pushed_messages == []
    assert [event for event, _ in stub.run_events] == [MODEL_FAILED_EVENT]
    assert stub.task_outputs[-1].sub_status == SubStatus.FAILED
    assert "Expected exactly one model task message" in stub.task_outputs[-1].details


def test_run_model_task_replies_on_invalid_model_payload() -> None:
    """Test invalid typed request payload still gets a task-message reply."""
    stub = FakeServerAppIoStub([_invalid_model_request_proto()])

    def invoke_model(
        request: JsonObject, on_stream_event: Callable[[JsonObject], None] | None
    ) -> ModelProviderResult:
        del request, on_stream_event
        raise AssertionError("provider must not be called")

    _run_model_task(stub, invoke_model)

    result = _pushed_model_result(stub)
    assert [event for event, _ in stub.run_events] == [MODEL_FAILED_EVENT]
    assert result.dst_task_id == SOURCE_TASK_ID
    assert result.reply_to_message_id == REQUEST_MESSAGE_ID
    assert result.payload["response"] == {"status": "failed"}
    assert result.payload["error"] == {
        "type": "ValueError",
        "message": "Task message payload requires `model`.",
    }
    assert stub.task_outputs[-1].sub_status == SubStatus.FAILED


def _model_request_proto(stream: bool) -> Any:
    """Create a model request Message proto."""
    message = ModelTaskMessage.create(
        dst_task_id=MODEL_TASK_ID,
        input=[{"role": "user", "content": "hello"}],
        model="gpt-4.1-mini",
        stream=stream,
    ).to_message()
    message.metadata.src_task_id = SOURCE_TASK_ID
    message.metadata.__dict__["_message_id"] = REQUEST_MESSAGE_ID
    return message_to_proto(message)


def _invalid_model_request_proto() -> Any:
    """Create an invalid model request Message proto."""
    metadata = Metadata(
        run_id=0,
        message_id="",
        src_node_id=0,
        dst_node_id=0,
        reply_to_message_id="",
        group_id="",
        created_at=0.0,
        ttl=60.0,
        message_type="query.model",
        src_task_id=SOURCE_TASK_ID,
        dst_task_id=MODEL_TASK_ID,
    )
    message = make_message(
        metadata=metadata,
        content=RecordDict(
            {
                "payload": ConfigRecord(
                    {"json": json.dumps({"input": [], "stream": False})}
                )
            }
        ),
    )
    message.metadata.__dict__["_message_id"] = REQUEST_MESSAGE_ID
    return message_to_proto(message)


def _pushed_model_result(stub: FakeServerAppIoStub) -> ModelTaskResultMessage:
    """Return the first pushed model result."""
    assert len(stub.pushed_messages) == 1
    message = message_from_proto(stub.pushed_messages[0])
    return ModelTaskResultMessage.from_message(message)
