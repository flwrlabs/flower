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
"""Base class for typed task-routed messages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

from flwr.app.message import Message
from flwr.app.message_type import MessageType
from flwr.supercore.task_message.constant import DEFAULT_TASK_MESSAGE_TTL
from flwr.supercore.task_message.utils import (
    build_task_message_metadata_and_content,
    task_message_payload_from_content,
)
from flwr.supercore.typing import JSONObject


class TaskMessage(Message):
    """Task-routed message carrying one JSON object payload."""

    def __init__(
        self,
        *,
        dst_task_id: int,
        payload: JSONObject,
        reply_to_message_id: str = "",
        ttl: float = DEFAULT_TASK_MESSAGE_TTL,
    ) -> None:
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
        """Return this task message's JSON object payload."""
        if not self.has_content():
            raise ValueError("Expected a message with content.")
        return task_message_payload_from_content(self.content)

    @classmethod
    def _from_message(
        cls,
        message: Message,
        *,
        validate_payload: Callable[[JSONObject], None],
        require_reply_to_message_id: bool = False,
        reply_to_message_id_error: str = "TaskMessage requires reply_to_message_id.",
    ) -> Self:
        """Parse a generic message into a typed task message."""
        if message.metadata.message_type != MessageType.QUERY:
            raise ValueError(
                f"Expected message type {MessageType.QUERY}, "
                f"got {message.metadata.message_type}."
            )
        if require_reply_to_message_id and not message.metadata.reply_to_message_id:
            raise ValueError(reply_to_message_id_error)
        if not message.has_content():
            raise ValueError("Expected a message with content.")

        payload = task_message_payload_from_content(message.content)
        validate_payload(payload)
        typed_message = cls.__new__(cls)
        typed_message.__dict__.update(message.__dict__)
        return typed_message
