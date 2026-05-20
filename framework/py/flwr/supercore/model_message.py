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

import json
from typing import ClassVar, TypeVar, cast

from flwr.app.metadata import Metadata
from flwr.common.message import Message, make_message
from flwr.common.record import ConfigRecord, RecordDict
from flwr.supercore.date import now
from flwr.supercore.inflatable.inflatable_object import InflatableObject

JSONValue = object
JSONObject = dict[str, JSONValue]

_PAYLOAD_RECORD_KEY = "payload"
_PAYLOAD_JSON_KEY = "json"
_DEFAULT_TASK_MESSAGE_TTL = 3600.0

T = TypeVar("T", bound=Message)


class ModelRequest(Message):
    """Task-routed model request in OpenAI Responses create-request shape."""

    MESSAGE_TYPE: ClassVar[str] = "query.model_request"

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        *,
        dst_task_id: int,
        input: list[JSONObject],  # pylint: disable=redefined-builtin
        model: str,
        stream: bool,
        tools: list[JSONObject] | None = None,
        tool_choice: JSONValue | None = None,
        reasoning: JSONObject | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: JSONObject | None = None,
        text: JSONObject | None = None,
        reply_to_message_id: str = "",
        ttl: float = _DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        payload: JSONObject = {
            "model": model,
            "input": input,
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
        _init_message_from_payload(
            self,
            dst_task_id=dst_task_id,
            message_type=self.MESSAGE_TYPE,
            payload=payload,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @property
    def payload(self) -> JSONObject:
        """Return this request's Responses create-request payload."""
        return _payload_from_message(self)

    @classmethod
    def create(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        cls,
        *,
        dst_task_id: int,
        input: list[JSONObject],  # pylint: disable=redefined-builtin
        model: str,
        stream: bool,
        tools: list[JSONObject] | None = None,
        tool_choice: JSONValue | None = None,
        reasoning: JSONObject | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: JSONObject | None = None,
        text: JSONObject | None = None,
        reply_to_message_id: str = "",
        ttl: float = _DEFAULT_TASK_MESSAGE_TTL,
    ) -> ModelRequest:
        """Create a model request message."""
        return cls(
            dst_task_id=dst_task_id,
            input=input,
            model=model,
            stream=stream,
            tools=tools,
            tool_choice=tool_choice,
            reasoning=reasoning,
            previous_response_id=previous_response_id,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            metadata=metadata,
            text=text,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ModelRequest:
        """Parse a generic message into a model request."""
        _validate_message_type(message, cls.MESSAGE_TYPE)
        payload = _payload_from_message(message)
        _validate_model_request_payload(payload)
        return _copy_as(cls, message)

    def deflate(self) -> bytes:
        """Deflate as the underlying transport `Message`."""
        return _deflate_as_message(self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> ModelRequest:
        """Inflate a model request from bytes."""
        return cls.from_message(Message.inflate(object_content, children))


class ModelResponse(Message):
    """Task-routed model response in OpenAI Responses object shape."""

    MESSAGE_TYPE: ClassVar[str] = "query.model_response"

    def __init__(
        self,
        *,
        dst_task_id: int,
        response: JSONObject,
        reply_to_message_id: str = "",
        ttl: float = _DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        _validate_model_response_payload(response)
        _init_message_from_payload(
            self,
            dst_task_id=dst_task_id,
            message_type=self.MESSAGE_TYPE,
            payload=response,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @property
    def payload(self) -> JSONObject:
        """Return this response's OpenAI Responses object payload."""
        return _payload_from_message(self)

    @classmethod
    def create(
        cls,
        *,
        dst_task_id: int,
        response: JSONObject,
        reply_to_message_id: str = "",
        ttl: float = _DEFAULT_TASK_MESSAGE_TTL,
    ) -> ModelResponse:
        """Create a model response message."""
        return cls(
            dst_task_id=dst_task_id,
            response=response,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ModelResponse:
        """Parse a generic message into a model response."""
        _validate_message_type(message, cls.MESSAGE_TYPE)
        payload = _payload_from_message(message)
        _validate_model_response_payload(payload)
        return _copy_as(cls, message)

    def deflate(self) -> bytes:
        """Deflate as the underlying transport `Message`."""
        return _deflate_as_message(self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> ModelResponse:
        """Inflate a model response from bytes."""
        return cls.from_message(Message.inflate(object_content, children))


def _set_optional(payload: JSONObject, key: str, value: JSONValue | None) -> None:
    """Set optional payload value if present."""
    if value is not None:
        payload[key] = value


def _init_message_from_payload(
    message: Message,
    *,
    dst_task_id: int,
    message_type: str,
    payload: JSONObject,
    reply_to_message_id: str,
    ttl: float,
) -> None:
    """Initialize a Message subclass from a task payload."""
    metadata = Metadata(
        run_id=0,
        message_id="",
        src_node_id=0,
        dst_node_id=0,
        reply_to_message_id=reply_to_message_id,
        group_id="",
        created_at=now().timestamp(),
        ttl=ttl,
        message_type=message_type,
        dst_task_id=dst_task_id,
    )
    metadata.delivered_at = ""
    message.__dict__.update(
        {
            "_metadata": metadata,
            "_content": RecordDict(
                {_PAYLOAD_RECORD_KEY: _config_record_from_payload(payload)}
            ),
            "_error": None,
        }
    )


def _config_record_from_payload(payload: JSONObject) -> ConfigRecord:
    """Serialize a JSON object payload into a ConfigRecord."""
    try:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError("Payload must be JSON serializable.") from err
    return ConfigRecord({_PAYLOAD_JSON_KEY: encoded})


def _payload_from_message(message: Message) -> JSONObject:
    """Parse a JSON object payload from a message."""
    if message.has_error():
        raise ValueError("Expected a message with content, got an error message.")
    if not message.has_content():
        raise ValueError("Expected a message with content.")

    record = message.content.config_records.get(_PAYLOAD_RECORD_KEY)
    if record is None:
        raise ValueError("Expected a payload ConfigRecord.")

    raw = record.get(_PAYLOAD_JSON_KEY)
    if not isinstance(raw, str):
        raise ValueError("Expected payload JSON to be a string.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as err:
        raise ValueError("Payload JSON is malformed.") from err

    if not isinstance(payload, dict):
        raise ValueError("Payload JSON must be an object.")
    return cast(JSONObject, payload)


def _validate_message_type(message: Message, expected: str) -> None:
    """Validate that the message type matches the expected value."""
    actual = message.metadata.message_type
    if actual != expected:
        raise ValueError(f"Expected message type {expected}, got {actual}.")


def _validate_model_request_payload(payload: JSONObject) -> None:
    """Validate the minimal Responses create-request shape."""
    if not isinstance(payload.get("model"), str):
        raise ValueError("ModelRequest payload requires string field 'model'.")
    if not isinstance(payload.get("input"), list):
        raise ValueError("ModelRequest payload requires list field 'input'.")
    if not isinstance(payload.get("stream"), bool):
        raise ValueError("ModelRequest payload requires bool field 'stream'.")

    if "tools" in payload and not isinstance(payload["tools"], list):
        raise ValueError("ModelRequest payload field 'tools' must be a list.")
    if "reasoning" in payload and not isinstance(payload["reasoning"], dict):
        raise ValueError("ModelRequest payload field 'reasoning' must be an object.")
    if "previous_response_id" in payload and not isinstance(
        payload["previous_response_id"], str
    ):
        raise ValueError(
            "ModelRequest payload field 'previous_response_id' must be a string."
        )
    if "instructions" in payload and not isinstance(payload["instructions"], str):
        raise ValueError("ModelRequest payload field 'instructions' must be a string.")
    if "max_output_tokens" in payload and not isinstance(
        payload["max_output_tokens"], int
    ):
        raise ValueError(
            "ModelRequest payload field 'max_output_tokens' must be an integer."
        )
    if "metadata" in payload and not isinstance(payload["metadata"], dict):
        raise ValueError("ModelRequest payload field 'metadata' must be an object.")
    if "text" in payload and not isinstance(payload["text"], dict):
        raise ValueError("ModelRequest payload field 'text' must be an object.")


def _validate_model_response_payload(payload: JSONObject) -> None:
    """Validate the minimal OpenAI Responses object shape."""
    if payload.get("object") != "response":
        raise ValueError("ModelResponse payload must be a Responses object.")
    if "id" in payload and not isinstance(payload["id"], str):
        raise ValueError("ModelResponse payload field 'id' must be a string.")
    if "status" in payload and not isinstance(payload["status"], str):
        raise ValueError("ModelResponse payload field 'status' must be a string.")
    if "output" in payload and not isinstance(payload["output"], list):
        raise ValueError("ModelResponse payload field 'output' must be a list.")
    if (
        "error" in payload
        and payload["error"] is not None
        and not isinstance(payload["error"], dict)
    ):
        raise ValueError("ModelResponse payload field 'error' must be an object.")


def _copy_as(cls: type[T], message: Message) -> T:
    """Return a Message subclass instance carrying the original message data."""
    typed = cls.__new__(cls)
    typed.__dict__.update(
        {
            "_metadata": message.metadata,
            "_content": message.content,
            "_error": None,
        }
    )
    return typed


def _deflate_as_message(message: Message) -> bytes:
    """Deflate a typed message using the plain Message transport header."""
    return make_message(metadata=message.metadata, content=message.content).deflate()
