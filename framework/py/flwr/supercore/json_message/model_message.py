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

from flwr.app.constants import DEFAULT_TTL
from flwr.supercore.json_message.base import JSONMessage
from flwr.supercore.typing import JSONObject, JSONValue


class ModelRequest(JSONMessage):
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
        ttl: float = DEFAULT_TTL,
    ) -> None:
        payload: JSONObject = {
            "model": model,
            "input": input_,
            "stream": stream,
            "tools": tools,
            "tool_choice": tool_choice,
            "reasoning": reasoning,
            "previous_response_id": previous_response_id,
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "metadata": metadata,
            "text": text,
        }
        payload = self._remove_none(payload)

        super().__init__(
            dst_task_id=dst_task_id,
            payload=payload,
            ttl=ttl,
        )

    @classmethod
    def _validate_payload(cls, payload: JSONObject) -> None:
        """Validate the minimal Responses create-request shape."""
        cls._validate_non_empty_string(payload, "model")
        if "input" not in payload:
            raise ValueError(f"{cls.__name__} payload requires field 'input'.")

        input_value = payload["input"]
        if not isinstance(input_value, str) and (
            not isinstance(input_value, Sequence)
            or not all(isinstance(item, dict) for item in input_value)
        ):
            raise ValueError(
                f"{cls.__name__} payload field 'input' must be a string or sequence "
                "of JSON objects."
            )

        cls._validate_optional_bool(payload, "stream")

        cls._validate_optional_json_object_sequence(payload, "tools")
        cls._validate_optional_json_object(payload, "reasoning")
        for field in ("previous_response_id", "instructions"):
            cls._validate_optional_string(payload, field)
        cls._validate_optional_int(payload, "max_output_tokens")
        for field in ("metadata", "text"):
            cls._validate_optional_json_object(payload, field)


class ModelResponse(JSONMessage):
    """Task-routed model response in Open Responses object shape."""

    def __init__(
        self,
        *,
        dst_task_id: int,
        response: JSONObject,
        reply_to_message_id: str,
        ttl: float = DEFAULT_TTL,
    ) -> None:
        if not reply_to_message_id:
            raise ValueError("ModelResponse requires reply_to_message_id.")
        super().__init__(
            dst_task_id=dst_task_id,
            payload=response,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    @classmethod
    def _validate_payload(cls, payload: JSONObject) -> None:
        """Validate the minimal Open Responses object shape."""
        if payload.get("object") != "response":
            raise ValueError(
                f"{cls.__name__} payload field 'object' must be 'response'."
            )
        for field in ("id", "status"):
            cls._validate_optional_string(payload, field)
        cls._validate_optional_json_object_sequence(payload, "output")
        if (
            "error" in payload
            and payload["error"] is not None
            and not isinstance(payload["error"], dict)
        ):
            raise ValueError(
                f"{cls.__name__} payload field 'error' must be a JSON object."
            )
