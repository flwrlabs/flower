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
from collections.abc import Sequence
from typing import cast

from flwr.app.message import ConfigRecord, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.app.metadata import Metadata
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.date import now
from flwr.supercore.typing import JSONObject, JSONValue

_PAYLOAD_RECORD_KEY = "payload"
_PAYLOAD_JSON_KEY = "json"
_DEFAULT_TASK_MESSAGE_TTL = 3600.0


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
        ttl: float = _DEFAULT_TASK_MESSAGE_TTL,
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
        message_metadata, content = _build_metadata_and_content(
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
        return _payload_from_content(self.content)

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

        payload = _payload_from_content(message.content)
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
        ttl: float = _DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        if not reply_to_message_id:
            raise ValueError("ModelResponse requires reply_to_message_id.")
        _validate_model_response_payload(response)
        metadata, content = _build_metadata_and_content(
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
        return _payload_from_content(self.content)

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

        payload = _payload_from_content(message.content)
        _validate_model_response_payload(payload)
        response = cls.__new__(cls)
        response.__dict__.update(message.__dict__)
        return response


def _set_optional(payload: JSONObject, key: str, value: JSONValue | None) -> None:
    """Set optional payload value if present."""
    if value is not None:
        payload[key] = value


def _build_metadata_and_content(
    dst_task_id: int,
    payload: JSONObject,
    reply_to_message_id: str,
    ttl: float,
) -> tuple[Metadata, RecordDict]:
    """Build message metadata and content from a task payload."""
    metadata = Metadata(
        run_id=0,
        message_id="",
        src_node_id=SUPERLINK_NODE_ID,
        dst_node_id=SUPERLINK_NODE_ID,
        reply_to_message_id=reply_to_message_id,
        group_id="",
        created_at=now().timestamp(),
        ttl=ttl,
        message_type=MessageType.QUERY,
        dst_task_id=dst_task_id,
    )
    return metadata, _payload_to_content(payload)


def _payload_to_content(payload: JSONObject) -> RecordDict:
    """Serialize a JSON object payload into message content."""
    try:
        # Store compact, strict JSON without unnecessary whitespace;
        # Python's NaN/Infinity extensions are invalid.
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as err:
        raise ValueError("Payload must be JSON serializable.") from err
    return RecordDict({_PAYLOAD_RECORD_KEY: ConfigRecord({_PAYLOAD_JSON_KEY: encoded})})


def _reject_non_finite_json_number(value: str) -> None:
    """Reject non-finite JSON number constants accepted by Python's decoder."""
    raise ValueError(f"Payload JSON contains non-finite number {value}.")


def _payload_from_content(content: RecordDict) -> JSONObject:
    """Parse a JSON object payload from message content."""
    record = content.config_records.get(_PAYLOAD_RECORD_KEY)
    if record is None:
        raise ValueError("Expected a payload ConfigRecord.")

    raw = record.get(_PAYLOAD_JSON_KEY)
    if not isinstance(raw, str):
        raise ValueError("Expected payload JSON to be a string.")

    try:
        # Reject Python's NaN/Infinity extensions while parsing inbound JSON.
        payload = json.loads(raw, parse_constant=_reject_non_finite_json_number)
    except ValueError as err:
        raise ValueError("Payload JSON is malformed.") from err

    if not isinstance(payload, dict):
        raise ValueError("Payload JSON must be a JSON object.")
    return cast(JSONObject, payload)


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
