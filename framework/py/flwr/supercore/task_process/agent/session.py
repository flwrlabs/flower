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
from flwr.app import Context, Message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    PullTaskMessageRequest,
    PushTaskMessageRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.model_message import ModelRequest, ModelResponse
from flwr.supercore.typing import JSONObject, JSONValue

from .context_items import append_items

_DEFAULT_MODEL_REPLY_TIMEOUT = 300.0
_DEFAULT_MODEL_REPLY_POLL_INTERVAL = 0.25
_DEFAULT_PULL_LIMIT = 1


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
        context: Context,
    ) -> None:
        self._stub = stub
        self._context = context
        self._run_id = run_id
        self._task_id = task_id

    def create(self, request: JSONObject) -> JSONObject:
        """Create a model response through a child model task."""
        model = request.get("model")
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
            input_=cast(str | Sequence[JSONObject], request.get("input")),
            model=model,
            stream=cast(bool, request.get("stream", False)),
            tools=cast(Sequence[JSONObject] | None, request.get("tools")),
            tool_choice=request.get("tool_choice"),
            reasoning=cast(JSONObject | None, request.get("reasoning")),
            previous_response_id=cast(str | None, request.get("previous_response_id")),
            instructions=cast(str | None, request.get("instructions")),
            max_output_tokens=cast(int | None, request.get("max_output_tokens")),
            metadata=cast(JSONObject | None, request.get("metadata")),
            text=cast(JSONObject | None, request.get("text")),
        )
        response_message = self._send_and_receive(message)
        response = ModelResponse.from_message(response_message)
        response_payload = response.payload
        output = response_payload.get("output")
        if _is_json_object_list(output):
            append_items(self._context, cast(list[JSONObject], output))
        return response_payload

    def _push_task_message(self, message: Message) -> None:
        """Push one task message and return its message ID."""
        message.metadata.__dict__["_run_id"] = self._run_id
        message.metadata.src_task_id = self._task_id
        message.metadata.__dict__["_message_id"] = message.object_id
        self._stub.PushTaskMessage(
            PushTaskMessageRequest(message=message_to_proto(message))
        )

    def _pull_task_messages(self) -> list[Message]:
        """Pull pending task messages."""
        res = self._stub.PullTaskMessage(PullTaskMessageRequest(limit=1))
        return [message_from_proto(msg) for msg in res.messages]

    def _send_and_receive(self, message: Message) -> Message:
        """Send one message and wait for its direct reply.

        For now, `flwr-agentapp` expects a strict one-request-one-reply exchange with
        `flwr-model`, so any non-matching pulled message is treated as an error.
        """
        # Push the message to the flwr-model
        self._push_task_message(message)
        message_id = message.metadata.message_id

        # Pull until a message arrives that replies to the pushed message, or timeout
        deadline = time.monotonic() + _DEFAULT_MODEL_REPLY_TIMEOUT
        while True:
            for pulled_msg in self._pull_task_messages():
                if pulled_msg.metadata.reply_to_message_id != message_id:
                    raise RuntimeError(
                        "Received a message that does not reply to the request."
                    )
                return pulled_msg

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for model response.")

            time.sleep(min(_DEFAULT_MODEL_REPLY_POLL_INTERVAL, remaining))


def _is_json_object_list(obj: JSONValue) -> bool:
    """Check if the given object is a list of JSON objects."""
    return isinstance(obj, list) and all(isinstance(item, dict) for item in obj)
