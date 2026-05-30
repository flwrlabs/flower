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
"""Tests for model task execution."""


from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import pytest

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import PullTaskMessageResponse  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject

from .app import run
from .provider import ModelProviderError


class _FakeStub:  # pylint: disable=invalid-name
    """Fake ServerAppIo stub."""

    def __init__(self, request: ModelRequest) -> None:
        self.request = request
        self.messages: list[ModelResponse] = []

    def PullTaskMessage(self, _request: object) -> PullTaskMessageResponse:
        """Fake task message pull."""
        return PullTaskMessageResponse(messages=[message_to_proto(self.request)])

    def PushTaskMessage(self, request: Any) -> object:
        """Record task messages."""
        self.messages.append(
            ModelResponse.from_message(message_from_proto(request.message))
        )
        return object()


def _request_message() -> ModelRequest:
    request = ModelRequest(
        dst_task_id=321,
        input_="Hello",
        model="gpt-5",
        stream=True,
    )
    request.metadata.__dict__["_run_id"] = 42
    request.metadata.__dict__["_message_id"] = "request-message-id"
    request.metadata.src_task_id = 123
    return request


def test_run_pushes_stream_and_success_responses() -> None:
    """A provider success should stream and reply to the source task."""
    stub = _FakeStub(_request_message())

    def invoke(
        request: JSONObject,
        *,
        on_stream_event: Callable[[JSONObject], None] | None = None,
    ) -> JSONObject:
        assert request == {"model": "gpt-5", "input": "Hello", "stream": True}
        assert on_stream_event is not None
        on_stream_event({"type": "response.output_text.delta", "delta": "Hi"})
        return {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": "gpt-5",
            "output": [{"type": "message", "role": "assistant", "content": []}],
        }

    with patch("flwr.supercore.task_process.model.app.invoke_model_provider", invoke):
        run(cast(ServerAppIoStub, stub), task_id=321, run_id=42)

    assert len(stub.messages) == 2
    stream_message = stub.messages[0]
    assert stream_message.metadata.src_task_id == 321
    assert stream_message.metadata.dst_task_id == 123
    assert stream_message.metadata.reply_to_message_id == "request-message-id"
    assert stream_message.payload["status"] == "in_progress"
    assert stream_message.payload["events"] == [
        {"type": "response.output_text.delta", "delta": "Hi", "model": "gpt-5"}
    ]
    final_message = stub.messages[1]
    assert final_message.metadata.src_task_id == 321
    assert final_message.metadata.dst_task_id == 123
    assert final_message.metadata.reply_to_message_id == "request-message-id"
    assert final_message.payload["id"] == "resp_1"
    assert "events" not in final_message.payload


def test_run_pushes_error_response_on_provider_failure() -> None:
    """A provider failure should reply with a failed response."""
    stub = _FakeStub(_request_message())
    provider_error = ModelProviderError(
        status_code=429,
        detail={"error": {"message": "quota exceeded"}},
    )

    with (
        patch(
            "flwr.supercore.task_process.model.app.invoke_model_provider",
            side_effect=provider_error,
        ),
        pytest.raises(ModelProviderError),
    ):
        run(cast(ServerAppIoStub, stub), task_id=321, run_id=42)

    assert len(stub.messages) == 1
    message = stub.messages[0]
    assert message.metadata.dst_task_id == 123
    assert message.metadata.reply_to_message_id == "request-message-id"
    assert message.payload["object"] == "response"
    assert message.payload["status"] == "failed"
    error = cast(JSONObject, message.payload["error"])
    assert error["provider_status_code"] == 429
