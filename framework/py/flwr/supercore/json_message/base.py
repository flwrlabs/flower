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
"""Base class for typed task-routed JSON messages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

from flwr.app.constants import DEFAULT_TTL
from flwr.app.message import ConfigRecord, Message, RecordDict
from flwr.app.message_type import MessageType
from flwr.app.metadata import Metadata
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.date import now
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps, strict_json_loads

from .constant import TASK_MESSAGE_PAYLOAD_JSON_KEY, TASK_MESSAGE_PAYLOAD_RECORD_KEY


class JSONMessage(Message, ABC):
    """Task-routed message carrying one JSON object payload."""

    def __init__(
        self,
        *,
        dst_task_id: int,
        payload: JSONObject,
        reply_to_message_id: str = "",
        ttl: float = DEFAULT_TTL,
    ) -> None:
        type(self)._validate_payload(payload)
        metadata, content = _build_metadata_and_content(
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
        """Return this task message's JSON object payload."""
        if not self.has_content():
            raise ValueError("Expected a message with content.")
        return _payload_from_content(self.content)

    @classmethod
    def from_message(cls, message: Message) -> Self:
        """Parse a generic message into a typed task message."""
        if not message.has_content():
            raise ValueError("Expected a message with content.")

        payload = _payload_from_content(message.content)
        cls._validate_payload(payload)
        typed_message = cls.__new__(cls)
        typed_message.__dict__.update(message.__dict__)
        return typed_message

    @classmethod
    @abstractmethod
    def _validate_payload(cls, payload: JSONObject) -> None:
        """Validate this task message type's payload."""

    @staticmethod
    def _remove_none(payload: JSONObject) -> JSONObject:
        """Return the payload without fields set to None."""
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def _validate_non_empty_string(cls, payload: JSONObject, field: str) -> None:
        """Validate that a payload field exists and is a non-empty string."""
        value = payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{cls.__name__} payload requires a non-empty string field '{field}'."
            )

    @classmethod
    def _validate_optional_string(cls, payload: JSONObject, field: str) -> None:
        """Validate that an optional payload field is a string when present."""
        if field in payload and not isinstance(payload[field], str):
            raise ValueError(
                f"{cls.__name__} payload field '{field}' must be a string."
            )

    @classmethod
    def _validate_optional_bool(cls, payload: JSONObject, field: str) -> None:
        """Validate that an optional payload field is a bool when present."""
        if field in payload and not isinstance(payload[field], bool):
            raise ValueError(f"{cls.__name__} payload field '{field}' must be a bool.")

    @classmethod
    def _validate_optional_int(cls, payload: JSONObject, field: str) -> None:
        """Validate that an optional payload field is an integer when present."""
        if field in payload and not isinstance(payload[field], int):
            raise ValueError(
                f"{cls.__name__} payload field '{field}' must be an integer."
            )

    @classmethod
    def _validate_optional_json_object(cls, payload: JSONObject, field: str) -> None:
        """Validate that an optional payload field is a JSON object when present."""
        if field in payload and not isinstance(payload[field], dict):
            raise ValueError(
                f"{cls.__name__} payload field '{field}' must be a JSON object."
            )

    @classmethod
    def _validate_optional_json_object_sequence(
        cls, payload: JSONObject, field: str
    ) -> None:
        """Validate that an optional payload field is a sequence of JSON objects."""
        if field not in payload:
            return

        value = payload[field]
        # Strings are sequences in Python, but they are not valid object sequences.
        if (
            not isinstance(value, Sequence)
            or isinstance(value, str)
            or not all(isinstance(item, dict) for item in value)
        ):
            raise ValueError(
                f"{cls.__name__} payload field '{field}' must be a sequence "
                "of JSON objects."
            )


def _build_metadata_and_content(
    dst_task_id: int,
    payload: JSONObject,
    reply_to_message_id: str,
    ttl: float,
) -> tuple[Metadata, RecordDict]:
    """Build task message metadata and content from a JSON object payload."""
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
    """Serialize a task message JSON object payload into message content."""
    try:
        encoded = strict_json_dumps(payload, compact=True)
    except (TypeError, ValueError) as err:
        raise ValueError("Payload must be JSON serializable.") from err
    return RecordDict(
        {
            TASK_MESSAGE_PAYLOAD_RECORD_KEY: ConfigRecord(
                {TASK_MESSAGE_PAYLOAD_JSON_KEY: encoded}
            )
        }
    )


def _payload_from_content(content: RecordDict) -> JSONObject:
    """Parse a task message JSON object payload from message content."""
    record = content.config_records.get(TASK_MESSAGE_PAYLOAD_RECORD_KEY)
    if record is None:
        raise ValueError("Expected a payload ConfigRecord.")

    raw = record.get(TASK_MESSAGE_PAYLOAD_JSON_KEY)
    if not isinstance(raw, str):
        raise ValueError("Expected payload JSON to be a string.")

    try:
        payload = strict_json_loads(raw)
    except ValueError as err:
        raise ValueError("Payload JSON is malformed.") from err

    if not isinstance(payload, dict):
        raise ValueError("Payload JSON must be a JSON object.")
    return payload
