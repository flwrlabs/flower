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
"""Internal task-routed message wrappers."""


from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import ClassVar, TypeAlias, TypeVar, cast

from flwr.app.metadata import Metadata
from flwr.common import ConfigRecord, Message, RecordDict
from flwr.common.message import make_message
from flwr.supercore.date import now

JsonObject: TypeAlias = "dict[str, JsonValue]"
JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | JsonObject"
TaskMessageSpecT = TypeVar("TaskMessageSpecT", bound="TaskMessageSpec")

DEFAULT_TASK_MESSAGE_TTL = 3600.0

_PAYLOAD_RECORD_KEY = "payload"
_PAYLOAD_JSON_KEY = "json"


@dataclass(frozen=True)
class TaskMessageSpec:
    """Common typed wrapper for task-routed messages.

    This wrapper gives executors a typed view over a task message while preserving
    `message.proto.Message` as the transport and storage format.
    """

    MESSAGE_TYPE: ClassVar[str] = ""

    dst_task_id: int
    message_type: str
    payload: JsonObject
    reply_to_message_id: str = ""
    ttl: float = DEFAULT_TASK_MESSAGE_TTL

    def __post_init__(self) -> None:
        """Validate task-message wrapper fields."""
        if self.dst_task_id <= 0:
            raise ValueError("`dst_task_id` must be greater than zero.")
        if self.ttl <= 0:
            raise ValueError("`ttl` must be greater than zero.")
        if not self.message_type:
            raise ValueError("`message_type` is required.")
        if self.MESSAGE_TYPE and self.message_type != self.MESSAGE_TYPE:
            raise ValueError(
                f"Expected message type {self.MESSAGE_TYPE}, got "
                f"{self.message_type}."
            )
        _validate_json_value(self.payload, "payload")
        self._validate_payload(self.payload)

    def to_message(self) -> Message:
        """Convert this wrapper into a task-routed `Message`."""
        metadata = Metadata(
            run_id=0,
            message_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message_id=self.reply_to_message_id,
            group_id="",
            created_at=now().timestamp(),
            ttl=self.ttl,
            message_type=self.message_type,
            dst_task_id=self.dst_task_id,
        )
        return make_message(
            metadata=metadata,
            content=RecordDict(
                {
                    _PAYLOAD_RECORD_KEY: ConfigRecord(
                        {_PAYLOAD_JSON_KEY: _compact_json(self.payload)}
                    )
                }
            ),
        )

    @classmethod
    def from_message(cls: type[TaskMessageSpecT], message: Message) -> TaskMessageSpecT:
        """Parse a task-routed `Message` into this wrapper type."""
        if cls.MESSAGE_TYPE and message.metadata.message_type != cls.MESSAGE_TYPE:
            raise ValueError(
                f"Expected message type {cls.MESSAGE_TYPE}, got "
                f"{message.metadata.message_type}."
            )

        dst_task_id = message.metadata.dst_task_id
        if dst_task_id is None:
            raise ValueError("`Message.metadata.dst_task_id` is required.")

        return cls(
            dst_task_id=dst_task_id,
            message_type=message.metadata.message_type,
            payload=_decode_payload(message),
            reply_to_message_id=message.metadata.reply_to_message_id,
            ttl=message.metadata.ttl,
        )

    def _validate_payload(self, payload: JsonObject) -> None:
        """Validate subclass-specific payload structure."""


@dataclass(frozen=True)
class ModelTaskMessage(TaskMessageSpec):
    """Task message carrying a Responses-compatible model request."""

    MESSAGE_TYPE: ClassVar[str] = "query.model"

    @classmethod
    def create(  # pylint: disable=too-many-arguments,too-many-positional-arguments,redefined-builtin
        cls,
        *,
        dst_task_id: int,
        input: JsonValue,
        model: str,
        stream: bool,
        tools: list[JsonObject] | None = None,
        tool_choice: JsonValue | None = None,
        reasoning: JsonObject | None = None,
        previous_response_id: str | None = None,
        instructions: str | None = None,
        max_output_tokens: int | None = None,
        metadata: JsonObject | None = None,
        text: JsonObject | None = None,
        reply_to_message_id: str = "",
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> ModelTaskMessage:
        """Create a Responses-compatible model request task message."""
        payload: JsonObject = {
            "input": input,
            "model": model,
            "stream": stream,
        }
        if tools is not None:
            payload["tools"] = cast(JsonValue, tools)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if instructions is not None:
            payload["instructions"] = instructions
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if metadata is not None:
            payload["metadata"] = metadata
        if text is not None:
            payload["text"] = text

        return cls(
            dst_task_id=dst_task_id,
            message_type=cls.MESSAGE_TYPE,
            payload=payload,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    def _validate_payload(self, payload: JsonObject) -> None:
        """Validate model request payload structure."""
        _require_key(payload, "input")
        _require_type(payload, "model", str)
        _require_type(payload, "stream", bool)
        _require_optional_type(payload, "tools", list)
        _require_optional_type(payload, "reasoning", dict)
        _require_optional_type(payload, "previous_response_id", str)
        _require_optional_type(payload, "instructions", str)
        _require_optional_int(payload, "max_output_tokens")
        _require_optional_type(payload, "metadata", dict)
        _require_optional_type(payload, "text", dict)


@dataclass(frozen=True)
class ModelTaskResultMessage(TaskMessageSpec):
    """Task message carrying a Responses-compatible model result."""

    MESSAGE_TYPE: ClassVar[str] = "query.model_result"

    @classmethod
    def create(
        cls,
        *,
        dst_task_id: int,
        response: JsonObject,
        response_id: str | None = None,
        usage: JsonObject | None = None,
        finish_reason: str | None = None,
        output: JsonValue | None = None,
        events: list[JsonObject] | None = None,
        error: JsonObject | None = None,
        reply_to_message_id: str = "",
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> ModelTaskResultMessage:
        """Create a Responses-compatible model result task message."""
        payload: JsonObject = {"response": response}
        if response_id is not None:
            payload["response_id"] = response_id
        if usage is not None:
            payload["usage"] = usage
        if finish_reason is not None:
            payload["finish_reason"] = finish_reason
        if output is not None:
            payload["output"] = output
        if events is not None:
            payload["events"] = cast(JsonValue, events)
        if error is not None:
            payload["error"] = error

        return cls(
            dst_task_id=dst_task_id,
            message_type=cls.MESSAGE_TYPE,
            payload=payload,
            reply_to_message_id=reply_to_message_id,
            ttl=ttl,
        )

    def _validate_payload(self, payload: JsonObject) -> None:
        """Validate model result payload structure."""
        _require_type(payload, "response", dict)
        _require_optional_type(payload, "response_id", str)
        _require_optional_type(payload, "usage", dict)
        _require_optional_type(payload, "finish_reason", str)
        _require_optional_type(payload, "events", list)
        _require_optional_type(payload, "error", dict)


def _compact_json(payload: JsonObject) -> str:
    """Return compact JSON for a validated payload object."""
    _validate_json_value(payload, "payload")
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _decode_payload(message: Message) -> JsonObject:
    """Decode the JSON payload record from a task-routed `Message`."""
    if not message.has_content():
        raise ValueError("Task message must contain content.")

    record = message.content.config_records.get(_PAYLOAD_RECORD_KEY)
    if record is None:
        raise ValueError("Task message content must contain a `payload` record.")

    raw = record.get(_PAYLOAD_JSON_KEY)
    if not isinstance(raw, str):
        raise ValueError("Task message payload `json` field must be a string.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Task message payload must contain valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Task message payload JSON must be an object.")

    _validate_json_value(payload, "payload")
    return cast(JsonObject, payload)


def _validate_json_value(value: object, path: str) -> None:
    """Validate that `value` can be represented as strict JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Task message payload contains non-finite float at {path}."
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Task message payload contains non-string key at {path}."
                )
            _validate_json_value(item, f"{path}.{key}")
        return

    raise ValueError(f"Task message payload contains non-JSON value at {path}.")


def _require_key(payload: JsonObject, key: str) -> JsonValue:
    """Return a required payload value."""
    if key not in payload:
        raise ValueError(f"Task message payload requires `{key}`.")
    return payload[key]


def _require_type(
    payload: JsonObject,
    key: str,
    expected_type: type[object],
) -> None:
    """Validate the type of a required payload value."""
    value = _require_key(payload, key)
    if not isinstance(value, expected_type):
        raise ValueError(f"Task message payload `{key}` must be {expected_type}.")


def _require_optional_type(
    payload: JsonObject,
    key: str,
    expected_type: type[object],
) -> None:
    """Validate the type of an optional payload value."""
    if key not in payload:
        return
    if not isinstance(payload[key], expected_type):
        raise ValueError(f"Task message payload `{key}` must be {expected_type}.")


def _require_optional_int(payload: JsonObject, key: str) -> None:
    """Validate the type of an optional integer payload value."""
    if key not in payload:
        return
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Task message payload `{key}` must be int.")
