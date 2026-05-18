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
"""AgentApp conversation client."""


from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Protocol

from flwr.proto.appio_pb2 import PushConversationItemsRequest  # pylint: disable=E0611
from flwr.proto.conversation_pb2 import ConversationItemPayload  # pylint: disable=E0611
from flwr.supercore.task_message import JsonObject


class ServerAppIoConversationClientStub(Protocol):
    """Subset of ServerAppIo RPCs used by the conversation client."""

    def PushConversationItems(self, request: PushConversationItemsRequest) -> Any:
        """Push conversation items."""


class AgentConversationClient:
    """Client for persisting AgentApp conversation items."""

    def __init__(
        self, *, stub: ServerAppIoConversationClientStub, conversation_id: str
    ) -> None:
        if not conversation_id:
            raise ValueError("`conversation_id` must be a non-empty string.")
        self._stub = stub
        self.conversation_id = conversation_id

    def add_items(self, items: Sequence[JsonObject]) -> list[int]:
        """Persist conversation items and return assigned item indices."""
        if not items:
            return []

        payloads = [
            ConversationItemPayload(item_json=_compact_json_object(item))
            for item in items
        ]
        response = self._stub.PushConversationItems(
            PushConversationItemsRequest(
                conversation_id=self.conversation_id,
                items=payloads,
            )
        )
        return list(response.item_indices)


def _compact_json_object(payload: JsonObject) -> str:
    """Return compact strict JSON for a conversation item."""
    _validate_json_value(payload, "item")
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def _validate_json_value(value: object, path: str) -> None:
    """Validate that `value` can be represented as strict JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Conversation item contains non-finite float at {path}.")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Conversation item contains non-string key at {path}."
                )
            _validate_json_value(item, f"{path}.{key}")
        return
    raise ValueError(f"Conversation item contains non-JSON value at {path}.")
