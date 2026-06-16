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
        super().__init__(
            dst_task_id=dst_task_id,
            payload=payload,
            ttl=ttl,
        )

    @classmethod
    def from_message(cls, message: Message) -> ConnectorRequest:
        """Parse a generic message into a connector request."""
        return cls._from_message(message)

    @classmethod
    def _validate_payload(cls, payload: JSONObject) -> None:
        """Validate the connector request payload shape."""
        cls._validate_non_empty_string(payload, "name")
        cls._validate_non_empty_string(payload, "call_id")
        if not isinstance(payload.get("arguments"), dict):
            raise ValueError(
                f"{cls.__name__} payload requires a JSON object field 'arguments'."
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
            require_reply_to_message_id=True,
        )

    @classmethod
    def _validate_payload(cls, payload: JSONObject) -> None:
        """Validate the connector response payload shape."""
        cls._validate_non_empty_string(payload, "name")
        cls._validate_non_empty_string(payload, "call_id")

        for field in ("output", "error"):
            if field not in payload:
                raise ValueError(f"{cls.__name__} payload requires field '{field}'.")

        error = payload["error"]
        if error is not None and not isinstance(error, dict):
            raise ValueError(
                f"{cls.__name__} payload field 'error' must be a JSON object."
            )
        if error is not None and payload["output"] is not None:
            raise ValueError(
                f"{cls.__name__} payload field 'output' must be null when "
                "'error' is set."
            )
