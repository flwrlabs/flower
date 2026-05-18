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
"""AgentApp model client."""


from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from flwr.common import Message
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    PullTaskMessageRequest,
    PushTaskMessageRequest,
)
from flwr.supercore.constant import TaskType
from flwr.supercore.task_message import (
    JsonObject,
    JsonValue,
    ModelTaskMessage,
    ModelTaskResultMessage,
)

from .exceptions import AgentModelError, AgentModelTimeoutError

DEFAULT_AGENT_MODEL = "gpt-4.1-mini"
DEFAULT_MODEL_RESPONSE_TIMEOUT = 300.0
DEFAULT_MODEL_RESPONSE_POLL_INTERVAL = 0.2
MODEL_REQUESTED_EVENT = "agent.model.requested"
MODEL_COMPLETED_EVENT = "agent.model.completed"


class ServerAppIoModelClientStub(Protocol):
    """Subset of ServerAppIo RPCs used by the AgentApp model client."""

    def CreateTask(self, request: CreateTaskRequest) -> Any:
        """Create a child task."""

    def PushTaskMessage(self, request: PushTaskMessageRequest) -> Any:
        """Push a task message."""

    def PullTaskMessage(self, request: PullTaskMessageRequest) -> Any:
        """Pull task messages."""


@dataclass(frozen=True)
class AgentModelResponse:
    """Responses-compatible model response returned to AgentApps."""

    response: JsonObject
    response_id: str | None
    output: JsonValue | None
    usage: JsonObject | None
    events: list[JsonObject]

    @property
    def output_text(self) -> str:
        """Return text extracted from the Responses-compatible output."""
        output_text = _optional_str(self.response.get("output_text"))
        if output_text is not None:
            return output_text
        if self.output is None:
            return ""
        return _extract_text(self.output)


class AgentModelClient:
    """Client for creating model tasks and awaiting model results."""

    def __init__(
        self,
        *,
        stub: ServerAppIoModelClientStub,
        task_id: int,
        emit_event: Callable[[str, JsonObject], None],
        default_model: str = DEFAULT_AGENT_MODEL,
        response_timeout: float = DEFAULT_MODEL_RESPONSE_TIMEOUT,
        poll_interval: float = DEFAULT_MODEL_RESPONSE_POLL_INTERVAL,
    ) -> None:
        if task_id <= 0:
            raise ValueError("`task_id` must be greater than zero.")
        if response_timeout <= 0:
            raise ValueError("`response_timeout` must be greater than zero.")
        if poll_interval < 0:
            raise ValueError("`poll_interval` must not be negative.")
        self._stub = stub
        self._task_id = task_id
        self._emit_event = emit_event
        self.default_model = default_model
        self.response_timeout = response_timeout
        self.poll_interval = poll_interval
        self._pending_messages: list[Message] = []

    def response(  # pylint: disable=too-many-arguments,too-many-positional-arguments,redefined-builtin
        self,
        *,
        input: JsonValue,
        model: str | None = None,
        stream: bool = True,
        tools: list[JsonObject] | None = None,
        tool_choice: JsonValue | None = None,
        reasoning: JsonObject | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: JsonObject | None = None,
        text: JsonObject | None = None,
        timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> AgentModelResponse:
        """Create a model task, send a request, and wait for its result."""
        selected_model = model or self.default_model
        if not selected_model:
            raise ValueError("`model` must be provided.")

        create_response = self._stub.CreateTask(
            CreateTaskRequest(type=TaskType.MODEL, model_ref=selected_model)
        )
        model_task_id = int(create_response.task_id)
        if model_task_id <= 0:
            raise RuntimeError("CreateTask did not return a valid model task_id.")

        spec = ModelTaskMessage.create(
            dst_task_id=model_task_id,
            input=input,
            model=selected_model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            previous_response_id=previous_response_id,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            text=text,
        )
        push_response = self._stub.PushTaskMessage(
            PushTaskMessageRequest(message=message_to_proto(spec.to_message()))
        )
        request_message_id = str(push_response.message_id)
        if not request_message_id:
            raise RuntimeError("PushTaskMessage did not return a message_id.")

        self._emit_event(
            MODEL_REQUESTED_EVENT,
            {"model_task_id": model_task_id, "model": selected_model},
        )

        result = self._pull_result(
            request_message_id=request_message_id,
            timeout=timeout if timeout is not None else self.response_timeout,
            poll_interval=(
                poll_interval if poll_interval is not None else self.poll_interval
            ),
        )

        self._emit_event(
            MODEL_COMPLETED_EVENT,
            {
                "model_task_id": model_task_id,
                "model": selected_model,
                **(
                    {"response_id": result.response_id}
                    if result.response_id is not None
                    else {}
                ),
            },
        )
        return result

    def _pull_result(
        self, *, request_message_id: str, timeout: float, poll_interval: float
    ) -> AgentModelResponse:
        """Poll task messages until the matching model result arrives."""
        if timeout <= 0:
            raise ValueError("`timeout` must be greater than zero.")
        if poll_interval < 0:
            raise ValueError("`poll_interval` must not be negative.")

        deadline = time.monotonic() + timeout
        while True:
            response = self._stub.PullTaskMessage(PullTaskMessageRequest(limit=10))
            messages = self._pending_messages + [
                message_from_proto(message_proto) for message_proto in response.messages
            ]
            self._pending_messages = []
            unmatched_messages = []
            for message in messages:
                if message.metadata.dst_task_id != self._task_id:
                    continue
                if message.metadata.reply_to_message_id != request_message_id:
                    unmatched_messages.append(message)
                    continue
                try:
                    result = ModelTaskResultMessage.from_message(message)
                except ValueError:
                    continue
                return _agent_model_response_from_result(result)
            self._pending_messages = unmatched_messages

            if time.monotonic() >= deadline:
                raise AgentModelTimeoutError(
                    {
                        "type": "model_timeout",
                        "message": "Timed out waiting for model task result.",
                    }
                )
            if poll_interval > 0:
                time.sleep(poll_interval)


def _agent_model_response_from_result(
    result: ModelTaskResultMessage,
) -> AgentModelResponse:
    """Convert a model result task message into an AgentModelResponse."""
    if error := _optional_json_object(result.payload.get("error")):
        raise AgentModelError(error)

    response = _required_json_object(result.payload.get("response"))
    events = _optional_json_object_list(result.payload.get("events"))
    return AgentModelResponse(
        response=response,
        response_id=_optional_str(result.payload.get("response_id")),
        output=result.payload.get("output"),
        usage=_optional_json_object(result.payload.get("usage")),
        events=events,
    )


def _extract_text(value: JsonValue) -> str:
    """Extract text from a Responses-compatible JSON value."""
    parts: list[str] = []
    _collect_text(value, parts)
    return "".join(parts)


def _collect_text(value: JsonValue, parts: list[str]) -> None:
    """Collect text leaves from Responses-compatible output structures."""
    if isinstance(value, str):
        parts.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text(item, parts)
        return
    if isinstance(value, dict):
        for key in ("output_text", "text", "content"):
            child = value.get(key)
            if isinstance(child, (str, list, dict)):
                _collect_text(child, parts)
                return
        output = value.get("output")
        if isinstance(output, (str, list, dict)):
            _collect_text(output, parts)


def _required_json_object(value: JsonValue | None) -> JsonObject:
    """Return value as a JSON object or raise."""
    if not isinstance(value, dict):
        raise AgentModelError(
            {
                "type": "invalid_model_result",
                "message": "Model task result payload requires `response`.",
            }
        )
    return cast(JsonObject, value)


def _optional_json_object(value: JsonValue | None) -> JsonObject | None:
    """Return value if it is a JSON object."""
    return cast(JsonObject, value) if isinstance(value, dict) else None


def _optional_json_object_list(value: JsonValue | None) -> list[JsonObject]:
    """Return value if it is a list of JSON objects."""
    if not isinstance(value, list):
        return []
    return [cast(JsonObject, item) for item in value if isinstance(item, dict)]


def _optional_str(value: JsonValue | None) -> str | None:
    """Return value if it is a string."""
    return value if isinstance(value, str) else None
