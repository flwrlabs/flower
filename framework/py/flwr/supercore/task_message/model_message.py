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
"""Typed model task messages."""


from __future__ import annotations

from collections.abc import Sequence

from flwr.app.message import Message
from flwr.supercore.task_message.base import TaskMessage
from flwr.supercore.task_message.constant import DEFAULT_TASK_MESSAGE_TTL
from flwr.supercore.task_message.validation import (
    set_optional,
    validate_json_object,
    validate_json_object_sequence,
    validate_non_empty_string,
    validate_optional_bool,
    validate_optional_int,
    validate_optional_string,
)
from flwr.supercore.typing import JSONObject, JSONValue


class ModelRequest(TaskMessage):
    """Task-routed model request in Open Responses create-request shape."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        *,
        dst_task_id: int,
        input_: str | Sequence[JSONObject],
        model: str,
        stream: bool = False,
        tools: Sequence[JSONObject] | None = None,
        tool_choice: JSONValue | None = None,
        reasoning: JSONObject | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: JSONObject | None = None,
        text: JSONObject | None = None,
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        payload: JSONObject = {
            "model": model,
            "input": input_,
            "stream": stream,
        }
        set_optional(payload, "tools", tools)
        set_optional(payload, "tool_choice", tool_choice)
        set_optional(payload, "reasoning", reasoning)
        set_optional(payload, "previous_response_id", previous_response_id)
        set_optional(payload, "instructions", instructions)
        set_optional(payload, "max_output_tokens", max_output_tokens)
        set_optional(payload, "metadata", metadata)
        set_optional(payload, "text", text)

        _validate_model_request_payload(payload)
        super().__init__(
            dst_task_id=dst_task_id,
            payload=payload,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ModelRequest:
        """Parse a generic message into a model request."""
        return cls._from_message(
            message,
            validate_payload=_validate_model_request_payload,
        )


class ModelResponse(TaskMessage):
    """Task-routed model response in Open Responses object shape."""

    def __init__(
        self,
        *,
        dst_task_id: int,
        response: JSONObject,
        reply_to_message_id: str,
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        if not reply_to_message_id:
            raise ValueError("ModelResponse requires reply_to_message_id.")
        _validate_model_response_payload(response)
        super().__init__(
            dst_task_id=dst_task_id,
            payload=response,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ModelResponse:
        """Parse a generic message into a model response."""
        return cls._from_message(
            message,
            validate_payload=_validate_model_response_payload,
            require_reply_to_message_id=True,
            reply_to_message_id_error="ModelResponse requires reply_to_message_id.",
        )


def _validate_model_request_payload(payload: JSONObject) -> None:
    """Validate the minimal Responses create-request shape."""
    validate_non_empty_string(payload, "model", owner="ModelRequest")
    if "input" not in payload:
        raise ValueError("ModelRequest payload requires field 'input'.")

    input_value = payload["input"]
    if not isinstance(input_value, str) and (
        not isinstance(input_value, Sequence)
        or not all(isinstance(item, dict) for item in input_value)
    ):
        raise ValueError(
            "ModelRequest payload field 'input' must be a string or sequence "
            "of JSON objects."
        )

    validate_optional_bool(payload, "stream", owner="ModelRequest")

    validate_json_object_sequence(payload, "tools", owner="ModelRequest")
    validate_json_object(payload, "reasoning", owner="ModelRequest", required=False)
    for field in ("previous_response_id", "instructions"):
        validate_optional_string(payload, field, owner="ModelRequest")
    validate_optional_int(payload, "max_output_tokens", owner="ModelRequest")
    for field in ("metadata", "text"):
        validate_json_object(payload, field, owner="ModelRequest", required=False)


def _validate_model_response_payload(payload: JSONObject) -> None:
    """Validate the minimal Open Responses object shape."""
    if payload.get("object") != "response":
        raise ValueError("ModelResponse payload field 'object' must be 'response'.")
    for field in ("id", "status"):
        validate_optional_string(payload, field, owner="ModelResponse")
    validate_json_object_sequence(payload, "output", owner="ModelResponse")
    validate_json_object(
        payload,
        "error",
        owner="ModelResponse",
        required=False,
        allow_none=True,
    )
