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
"""Tests for AgentApp executor sessions."""


from typing import cast

from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushTaskMessageRequest,
    PushTaskMessageResponse,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject

from .session import RuntimeAgentResponses


class _Stub:
    """ServerAppIoStub test double."""

    def __init__(self, *, agent_task_id: int, response: JSONObject) -> None:
        self.create_task_requests: list[CreateTaskRequest] = []
        self.push_task_message_requests: list[PushTaskMessageRequest] = []
        self.pull_task_message_requests: list[PullTaskMessageRequest] = []
        self._agent_task_id = agent_task_id
        self._response = response

    def CreateTask(self, request: CreateTaskRequest) -> CreateTaskResponse:
        """Record CreateTask requests."""
        self.create_task_requests.append(request)
        return CreateTaskResponse(task_id=456)

    def PushTaskMessage(
        self, request: PushTaskMessageRequest
    ) -> PushTaskMessageResponse:
        """Record PushTaskMessage requests."""
        self.push_task_message_requests.append(request)
        return PushTaskMessageResponse(message_id="request-message-id")

    def PullTaskMessage(
        self, request: PullTaskMessageRequest
    ) -> PullTaskMessageResponse:
        """Return a matching model response."""
        self.pull_task_message_requests.append(request)
        response = ModelResponse(
            dst_task_id=self._agent_task_id,
            response=self._response,
            reply_to_message_id="request-message-id",
        )
        return PullTaskMessageResponse(messages=[message_to_proto(response)])


def test_agent_responses_create_uses_private_task_message_helper() -> None:
    """Runtime AgentResponses should send one ModelRequest and return its response."""
    response_payload: JSONObject = {"object": "response", "id": "resp_123"}
    stub = _Stub(agent_task_id=123, response=response_payload)
    responses = RuntimeAgentResponses(
        stub=cast(ServerAppIoStub, stub),
        run_id=789,
        task_id=123,
    )

    result = responses.create({"model": "gpt-5", "input": "Hello"})

    assert result == response_payload
    assert len(stub.create_task_requests) == 1
    assert stub.create_task_requests[0].type == TaskType.MODEL
    assert stub.create_task_requests[0].model_ref == "gpt-5"
    assert len(stub.push_task_message_requests) == 1
    assert len(stub.pull_task_message_requests) == 1

    pushed = message_from_proto(stub.push_task_message_requests[0].message)
    request = ModelRequest.from_message(pushed)
    assert request.metadata.run_id == 789
    assert request.metadata.src_task_id == 123
    assert request.metadata.message_id != ""
    assert request.metadata.dst_task_id == 456
    assert request.payload == {
        "model": "gpt-5",
        "input": "Hello",
        "stream": False,
    }
