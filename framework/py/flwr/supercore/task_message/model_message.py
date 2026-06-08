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
from flwr.app.message_type import MessageType
from flwr.supercore.task_message.constant import DEFAULT_TASK_MESSAGE_TTL
from flwr.supercore.task_message.utils import (
    build_task_message_metadata_and_content,
    task_message_payload_from_content,
)
from flwr.supercore.typing import JSONObject, JSONValue


class ModelRequest(Message):
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
        _set_optional(payload, "tools", tools)
        _set_optional(payload, "tool_choice", tool_choice)
        _set_optional(payload, "reasoning", reasoning)
        _set_optional(payload, "previous_response_id", previous_response_id)
        _set_optional(payload, "instructions", instructions)
        _set_optional(payload, "max_output_tokens", max_output_tokens)
        _set_optional(payload, "metadata", metadata)
        _set_optional(payload, "text", text)

        _validate_model_request_payload(payload)
        message_metadata, content = build_task_message_metadata_and_content(
            dst_task_id,
            payload,
            "",
            ttl,
        )
        super().__init__(  # type: ignore[call-overload]
            metadata=message_metadata,
            content=content,
        )

    @property
    def payload(self) -> JSONObject:
        """Return this request's Responses create-request payload."""
        if not self.has_content():
            raise ValueError("Expected a message with content.")
        return task_message_payload_from_content(self.content)

    @classmethod
    def from_message(cls, message: Message) -> ModelRequest:
        """Parse a generic message into a model request."""
        if message.metadata.message_type != MessageType.QUERY:
            raise ValueError(
                f"Expected message type {MessageType.QUERY}, "
                f"got {message.metadata.message_type}."
            )
        if not message.has_content():
            raise ValueError("Expected a message with content.")

        payload = task_message_payload_from_content(message.content)
        _validate_model_request_payload(payload)
        request = cls.__new__(cls)
        request.__dict__.update(message.__dict__)
        return request


class ModelResponse(Message):
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
        metadata, content = build_task_message_metadata_and_content(
            dst_task_id,
            response,
            reply_to_message_id,
            ttl,
        )
        super().__init__(  # type: ignore[call-overload]
            metadata=metadata,
            content=content,
        )

    @property
    def payload(self) -> JSONObject:
        """Return this response's Open Responses object payload."""
        if not self.has_content():
            raise ValueError("Expected a message with content.")
        return task_message_payload_from_content(self.content)

    @classmethod
    def from_message(cls, message: Message) -> ModelResponse:
        """Parse a generic message into a model response."""
        if message.metadata.message_type != MessageType.QUERY:
            raise ValueError(
                f"Expected message type {MessageType.QUERY}, "
                f"got {message.metadata.message_type}."
            )
        if not message.metadata.reply_to_message_id:
            raise ValueError("ModelResponse requires reply_to_message_id.")
        if not message.has_content():
            raise ValueError("Expected a message with content.")

        payload = task_message_payload_from_content(message.content)
        _validate_model_response_payload(payload)
        response = cls.__new__(cls)
        response.__dict__.update(message.__dict__)
        return response


def _set_optional(payload: JSONObject, key: str, value: JSONValue | None) -> None:
    """Set optional payload value if present."""
    if value is not None:
        payload[key] = value


def _validate_json_object_sequence_field(
    payload: JSONObject, field: str, *, owner: str, required: bool = False
) -> None:
    """Validate that a payload field is a sequence of JSON objects."""
    if field not in payload:
        if required:
            raise ValueError(f"{owner} payload requires field '{field}'.")
        return

    value = payload[field]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not all(isinstance(item, dict) for item in value)
    ):
        raise ValueError(
            f"{owner} payload field '{field}' must be a sequence of JSON objects."
        )


def _validate_model_request_input_field(payload: JSONObject) -> None:
    """Validate that a model request input is a string or sequence of JSON objects."""
    if "input" not in payload:
        raise ValueError("ModelRequest payload requires field 'input'.")

    value = payload["input"]
    if isinstance(value, str):
        return
    if not isinstance(value, Sequence) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(
            "ModelRequest payload field 'input' must be a string or sequence "
            "of JSON objects."
        )


def _validate_model_request_payload(payload: JSONObject) -> None:
    """Validate the minimal Responses create-request shape."""
    if not isinstance(payload.get("model"), str):
        raise ValueError("ModelRequest payload requires a string field 'model'.")
    _validate_model_request_input_field(payload)
    if "stream" in payload and not isinstance(payload["stream"], bool):
        raise ValueError("ModelRequest payload field 'stream' must be a bool.")

    _validate_json_object_sequence_field(payload, "tools", owner="ModelRequest")
    if "reasoning" in payload and not isinstance(payload["reasoning"], dict):
        raise ValueError(
            "ModelRequest payload field 'reasoning' must be a JSON object."
        )
    for field in ("previous_response_id", "instructions"):
        if field in payload and not isinstance(payload[field], str):
            raise ValueError(f"ModelRequest payload field '{field}' must be a string.")
    if "max_output_tokens" in payload and not isinstance(
        payload["max_output_tokens"], int
    ):
        raise ValueError(
            "ModelRequest payload field 'max_output_tokens' must be an integer."
        )
    for field in ("metadata", "text"):
        if field in payload and not isinstance(payload[field], dict):
            raise ValueError(
                f"ModelRequest payload field '{field}' must be a JSON object."
            )


def _validate_model_response_payload(payload: JSONObject) -> None:
    """Validate the minimal Open Responses object shape."""
    if payload.get("object") != "response":
        raise ValueError("ModelResponse payload field 'object' must be 'response'.")
    for field in ("id", "status"):
        if field in payload and not isinstance(payload[field], str):
            raise ValueError(f"ModelResponse payload field '{field}' must be a string.")
    _validate_json_object_sequence_field(payload, "output", owner="ModelResponse")
    if (
        "error" in payload
        and payload["error"] is not None
        and not isinstance(payload["error"], dict)
    ):
        raise ValueError("ModelResponse payload field 'error' must be a JSON object.")
