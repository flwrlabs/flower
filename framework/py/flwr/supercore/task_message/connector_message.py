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
from flwr.supercore.task_message.base import TaskMessage
from flwr.supercore.task_message.constant import DEFAULT_TASK_MESSAGE_TTL
from flwr.supercore.task_message.validation import (
    require_json_object,
    require_non_empty_string,
    require_present,
)
from flwr.supercore.typing import JSONObject, JSONValue


class ConnectorRequest(TaskMessage):
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
        super().__init__(
            dst_task_id=dst_task_id,
            payload=payload,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ConnectorRequest:
        """Parse a generic message into a connector request."""
        return cls._from_message(
            message,
            validate_payload=_validate_connector_request_payload,
        )


class ConnectorResponse(TaskMessage):
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
        super().__init__(
            dst_task_id=dst_task_id,
            payload=payload,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ConnectorResponse:
        """Parse a generic message into a connector response."""
        return cls._from_message(
            message,
            validate_payload=_validate_connector_response_payload,
            require_reply_to_message_id=True,
            reply_to_message_id_error="ConnectorResponse requires reply_to_message_id.",
        )


def _validate_connector_request_payload(payload: JSONObject) -> None:
    """Validate the connector request payload shape."""
    require_non_empty_string(payload, "name", owner="ConnectorRequest")
    require_non_empty_string(payload, "call_id", owner="ConnectorRequest")
    require_json_object(payload, "arguments", owner="ConnectorRequest")


def _validate_connector_response_payload(payload: JSONObject) -> None:
    """Validate the connector response payload shape."""
    require_non_empty_string(payload, "name", owner="ConnectorResponse")
    require_non_empty_string(payload, "call_id", owner="ConnectorResponse")

    require_present(payload, "output", owner="ConnectorResponse")
    require_present(payload, "error", owner="ConnectorResponse")

    error = payload["error"]
    require_json_object(
        payload,
        "error",
        owner="ConnectorResponse",
        required=False,
        allow_none=True,
    )
    if error is not None and payload["output"] is not None:
        raise ValueError(
            "ConnectorResponse payload field 'output' must be null when "
            "'error' is set."
        )
