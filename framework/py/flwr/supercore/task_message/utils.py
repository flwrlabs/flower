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
"""Helpers for task-routed message payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps, strict_json_loads

from .constant import TASK_MESSAGE_PAYLOAD_JSON_KEY, TASK_MESSAGE_PAYLOAD_RECORD_KEY

if TYPE_CHECKING:
    from flwr.app.message import RecordDict
    from flwr.app.metadata import Metadata


def build_task_message_metadata_and_content(
    dst_task_id: int,
    payload: JSONObject,
    reply_to_message_id: str,
    ttl: float,
) -> tuple[Metadata, RecordDict]:
    """Build task message metadata and content from a JSON object payload."""
    from flwr.app.message_type import MessageType
    from flwr.app.metadata import Metadata
    from flwr.common.constant import SUPERLINK_NODE_ID
    from flwr.supercore.date import now

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
    return metadata, task_message_payload_to_content(payload)


def task_message_payload_to_content(payload: JSONObject) -> RecordDict:
    """Serialize a task message JSON object payload into message content."""
    from flwr.app.message import ConfigRecord, RecordDict

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


def task_message_payload_from_content(content: RecordDict) -> JSONObject:
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
    return cast(JSONObject, payload)
