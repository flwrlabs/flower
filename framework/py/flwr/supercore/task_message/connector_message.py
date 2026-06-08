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
"""Typed connector task messages."""


from __future__ import annotations

from flwr.app.message import Message
from flwr.app.message_type import MessageType
from flwr.supercore.task_message.constant import DEFAULT_TASK_MESSAGE_TTL
from flwr.supercore.task_message.utils import (
    build_task_message_metadata_and_content,
    task_message_payload_from_content,
)
from flwr.supercore.typing import JSONObject, JSONValue


class ConnectorRequest(Message):
    """Task-routed connector request."""

    def __init__(
        self,
        *,
        dst_task_id: int,
        name: str,
        call_id: str,
        arguments: JSONObject,
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        payload: JSONObject = {
            "name": name,
            "call_id": call_id,
            "arguments": arguments,
        }
        _validate_connector_request_payload(payload)
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
        """Return this connector request payload."""
        if not self.has_content():
            raise ValueError("Expected a message with content.")
        return task_message_payload_from_content(self.content)

    @classmethod
    def from_message(cls, message: Message) -> ConnectorRequest:
        """Parse a generic message into a connector request."""
        if message.metadata.message_type != MessageType.QUERY:
            raise ValueError(
                f"Expected message type {MessageType.QUERY}, "
                f"got {message.metadata.message_type}."
            )
        if not message.has_content():
            raise ValueError("Expected a message with content.")

        payload = task_message_payload_from_content(message.content)
        _validate_connector_request_payload(payload)
        request = cls.__new__(cls)
        request.__dict__.update(message.__dict__)
        return request


class ConnectorResponse(Message):
    """Task-routed connector response."""

    def __init__(
        self,
        *,
        dst_task_id: int,
        name: str,
        call_id: str,
        output: JSONValue,
        error: JSONObject | None,
        reply_to_message_id: str,
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
        if not reply_to_message_id:
            raise ValueError("ConnectorResponse requires reply_to_message_id.")

        payload: JSONObject = {
            "name": name,
            "call_id": call_id,
            "output": output,
            "error": error,
        }
        _validate_connector_response_payload(payload)
        metadata, content = build_task_message_metadata_and_content(
            dst_task_id,
            payload,
            reply_to_message_id,
            ttl,
        )
        super().__init__(  # type: ignore[call-overload]
            metadata=metadata,
            content=content,
        )

    @property
    def payload(self) -> JSONObject:
        """Return this connector response payload."""
        if not self.has_content():
            raise ValueError("Expected a message with content.")
        return task_message_payload_from_content(self.content)

    @classmethod
    def from_message(cls, message: Message) -> ConnectorResponse:
        """Parse a generic message into a connector response."""
        if message.metadata.message_type != MessageType.QUERY:
            raise ValueError(
                f"Expected message type {MessageType.QUERY}, "
                f"got {message.metadata.message_type}."
            )
        if not message.metadata.reply_to_message_id:
            raise ValueError("ConnectorResponse requires reply_to_message_id.")
        if not message.has_content():
            raise ValueError("Expected a message with content.")

        payload = task_message_payload_from_content(message.content)
        _validate_connector_response_payload(payload)
        response = cls.__new__(cls)
        response.__dict__.update(message.__dict__)
        return response


def _validate_string_field(payload: JSONObject, field: str, *, owner: str) -> None:
    """Validate that a payload field is a non-empty string."""
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{owner} payload requires a non-empty string field '{field}'."
        )


def _validate_connector_request_payload(payload: JSONObject) -> None:
    """Validate the connector request payload shape."""
    _validate_string_field(payload, "name", owner="ConnectorRequest")
    _validate_string_field(payload, "call_id", owner="ConnectorRequest")
    if not isinstance(payload.get("arguments"), dict):
        raise ValueError(
            "ConnectorRequest payload requires a JSON object field 'arguments'."
        )


def _validate_connector_response_payload(payload: JSONObject) -> None:
    """Validate the connector response payload shape."""
    _validate_string_field(payload, "name", owner="ConnectorResponse")
    _validate_string_field(payload, "call_id", owner="ConnectorResponse")

    if "output" not in payload:
        raise ValueError("ConnectorResponse payload requires field 'output'.")
    if "error" not in payload:
        raise ValueError("ConnectorResponse payload requires field 'error'.")

    error = payload["error"]
    if error is not None and not isinstance(error, dict):
        raise ValueError(
            "ConnectorResponse payload field 'error' must be a JSON object."
        )
    if error is not None and payload["output"] is not None:
        raise ValueError(
            "ConnectorResponse payload field 'output' must be null when "
            "'error' is set."
        )
