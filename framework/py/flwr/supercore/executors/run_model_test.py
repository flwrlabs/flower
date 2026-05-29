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

from collections.abc import Callable
from queue import Queue
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from flwr.common.constant import Status, SubStatus
from flwr.common.exit import ExitCode
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputResponse,
    PullTaskMessageResponse,
    PushTaskOutputRequest,
    SendTaskHeartbeatResponse,
)
from flwr.proto.run_pb2 import Run, RunStatus  # pylint: disable=E0611
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject

from .model.provider import ModelProviderError
from .run_model import run_model


class _FakeChannel:
    """Fake gRPC channel."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        """Record channel close."""
        self.closed = True


class _FakeStub:  # pylint: disable=invalid-name
    """Fake ServerAppIo stub."""

    def __init__(self, request: ModelRequest) -> None:
        self.request = request
        self.messages: list[ModelResponse] = []
        self.outputs: list[PushTaskOutputRequest] = []
        self.calls: list[str] = []

    def SendTaskHeartbeat(self, _request: object) -> SendTaskHeartbeatResponse:
        """Fake heartbeat."""
        return SendTaskHeartbeatResponse(success=True)

    def PullTaskInput(self, _request: object) -> PullTaskInputResponse:
        """Fake task input."""
        self.calls.append("PullTaskInput")
        return PullTaskInputResponse(
            run=Run(
                run_id=42,
                status=RunStatus(status=Status.RUNNING, sub_status="", details=""),
            ),
            task_id=321,
        )

    def PullTaskMessage(self, _request: object) -> PullTaskMessageResponse:
        """Fake task message pull."""
        self.calls.append("PullTaskMessage")
        return PullTaskMessageResponse(messages=[message_to_proto(self.request)])

    def PushTaskMessage(self, request: Any) -> object:
        """Record task messages."""
        self.calls.append("PushTaskMessage")
        self.messages.append(
            ModelResponse.from_message(message_from_proto(request.message))
        )
        return object()

    def PushTaskOutput(self, request: Any) -> object:
        """Record task output."""
        self.calls.append("PushTaskOutput")
        self.outputs.append(request)
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


def _run_with_stub(stub: _FakeStub) -> Mock:
    heartbeat_sender = Mock()
    heartbeat_sender.is_running = True
    flwr_exit = Mock(side_effect=RuntimeError("exit"))
    with (
        patch(
            "flwr.supercore.executors.run_model._create_serverappio_stub",
            return_value=(_FakeChannel(), stub, SimpleNamespace(max_tries=None)),
        ),
        patch(
            "flwr.supercore.executors.run_model.HeartbeatSender",
            return_value=heartbeat_sender,
        ),
        patch("flwr.supercore.executors.run_model.start_log_uploader"),
        patch("flwr.supercore.executors.run_model.flush_logs"),
        patch("flwr.supercore.executors.run_model.stop_log_uploader"),
        patch("flwr.supercore.executors.run_model.flwr_exit", flwr_exit),
        pytest.raises(RuntimeError, match="exit"),
    ):
        run_model(
            serverappio_api_address="127.0.0.1:9091",
            log_queue=Queue(),
            token="task-token",
        )
    heartbeat_sender.start.assert_called_once()
    heartbeat_sender.stop.assert_called_once()
    return flwr_exit


def test_run_model_pushes_stream_and_success_responses() -> None:
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

    with patch("flwr.supercore.executors.run_model.invoke_model_provider", invoke):
        flwr_exit = _run_with_stub(stub)

    assert stub.calls.index("PullTaskInput") < stub.calls.index("PullTaskMessage")
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
    assert stub.outputs[-1].sub_status == SubStatus.COMPLETED
    assert flwr_exit.call_args.args[0] == ExitCode.SUCCESS


def test_run_model_pushes_error_response_on_provider_failure() -> None:
    """A provider failure should reply with a failed response and fail the task."""
    stub = _FakeStub(_request_message())

    provider_error = ModelProviderError(
        status_code=429,
        detail={"error": {"message": "quota exceeded"}},
    )

    with patch(
        "flwr.supercore.executors.run_model.invoke_model_provider",
        side_effect=provider_error,
    ):
        flwr_exit = _run_with_stub(stub)

    assert len(stub.messages) == 1
    message = stub.messages[0]
    assert message.metadata.dst_task_id == 123
    assert message.metadata.reply_to_message_id == "request-message-id"
    assert message.payload["object"] == "response"
    assert message.payload["status"] == "failed"
    error = cast(JSONObject, message.payload["error"])
    assert error["provider_status_code"] == 429
    assert stub.outputs[-1].sub_status == SubStatus.FAILED
    assert "quota exceeded" in stub.outputs[-1].details
    assert flwr_exit.call_args.args[0] == ExitCode.SERVERAPP_EXCEPTION
