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
"""Executor-bound AgentApp session implementations."""


from __future__ import annotations

import time
from collections.abc import Sequence
from typing import cast

from flwr.agentapp import AgentResponses, AgentSession
from flwr.app import Message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    PullTaskMessageRequest,
    PushTaskMessageRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject

_DEFAULT_MODEL_REPLY_TIMEOUT = 300.0
_DEFAULT_MODEL_REPLY_POLL_INTERVAL = 0.25
_DEFAULT_PULL_LIMIT = 10


class RuntimeAgentSession(AgentSession):
    """AgentSession bound to one AgentApp task."""

    def __init__(self, responses: AgentResponses) -> None:
        self._responses = responses

    @property
    def responses(self) -> AgentResponses:
        """Model response creation API."""
        return self._responses


class RuntimeAgentResponses(AgentResponses):
    """AgentResponses implementation backed by AppIo task messages."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        stub: ServerAppIoStub,
        run_id: int,
        task_id: int,
    ) -> None:
        self._stub = stub
        self._messages = _RuntimeTaskMessages(
            stub=stub,
            run_id=run_id,
            task_id=task_id,
        )

    def create(self, request: JSONObject) -> JSONObject:
        """Create a model response through a child model task."""
        payload = dict(request)
        model = payload.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(
                "AgentResponses request requires a non-empty string 'model' field."
            )

        create_res = self._stub.CreateTask(
            CreateTaskRequest(type=TaskType.MODEL, model_ref=model)
        )
        if not create_res.HasField("task_id"):
            raise RuntimeError("Model task could not be created.")

        model_task_id = create_res.task_id
        message = ModelRequest(
            dst_task_id=model_task_id,
            input_=cast(str | Sequence[JSONObject], payload.get("input")),
            model=model,
            stream=cast(bool, payload.get("stream", False)),
            tools=cast(Sequence[JSONObject] | None, payload.get("tools")),
            tool_choice=payload.get("tool_choice"),
            reasoning=cast(JSONObject | None, payload.get("reasoning")),
            previous_response_id=cast(str | None, payload.get("previous_response_id")),
            instructions=cast(str | None, payload.get("instructions")),
            max_output_tokens=cast(int | None, payload.get("max_output_tokens")),
            metadata=cast(JSONObject | None, payload.get("metadata")),
            text=cast(JSONObject | None, payload.get("text")),
        )
        response_message = self._messages.send_and_receive(message)
        response = ModelResponse.from_message(response_message)
        return response.payload


class _RuntimeTaskMessages:
    """Private task-message runtime helper."""

    def __init__(
        self,
        *,
        stub: ServerAppIoStub,
        run_id: int,
        task_id: int,
    ) -> None:
        self._stub = stub
        self._run_id = run_id
        self._task_id = task_id
        self._inbox: list[Message] = []

    def send_and_receive(self, message: Message) -> Message:
        """Push one task message and wait for its matching reply."""
        message.metadata.__dict__["_run_id"] = self._run_id
        message.metadata.src_task_id = self._task_id
        message.metadata.__dict__["_message_id"] = message.object_id
        push_res = self._stub.PushTaskMessage(
            PushTaskMessageRequest(message=message_to_proto(message))
        )
        return self._pull_matching_reply(push_res.message_id)

    def _pull_matching_reply(self, reply_to_message_id: str) -> Message:
        deadline = time.monotonic() + _DEFAULT_MODEL_REPLY_TIMEOUT

        while True:
            for idx, message in enumerate(self._inbox):
                if _matches_reply(message, self._task_id, reply_to_message_id):
                    return self._inbox.pop(idx)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for model response.")

            res = self._stub.PullTaskMessage(
                PullTaskMessageRequest(limit=_DEFAULT_PULL_LIMIT)
            )
            messages = [message_from_proto(message) for message in res.messages]
            if not messages:
                time.sleep(min(_DEFAULT_MODEL_REPLY_POLL_INTERVAL, remaining))
                continue

            for message in messages:
                if _matches_reply(message, self._task_id, reply_to_message_id):
                    return message
                self._inbox.append(message)


def _matches_reply(
    message: Message,
    dst_task_id: int,
    reply_to_message_id: str,
) -> bool:
    return (
        message.metadata.dst_task_id == dst_task_id
        and message.metadata.reply_to_message_id == reply_to_message_id
    )
