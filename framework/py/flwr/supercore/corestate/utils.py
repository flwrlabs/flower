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
"""Utility functions for CoreState."""


from datetime import datetime
from os import urandom

from flwr.app import Context, Message
from flwr.common import serde
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.supercore.date import now
from flwr.supercore.utils import strict_json_loads

# unix timestamp of 28 February 2025 00h:00m:00s UTC
_MIN_VALID_MESSAGE_CREATED_AT = 1740700800.0


def generate_rand_int_from_bytes(
    num_bytes: int, exclude: set[int] | None = None
) -> int:
    """Generate a random unsigned integer from `num_bytes` bytes.

    If `exclude` is set, this function guarantees such number is not returned.
    """
    num = int.from_bytes(urandom(num_bytes), "little", signed=False)

    if exclude:
        while num in exclude:
            num = int.from_bytes(urandom(num_bytes), "little", signed=False)
    return num


def timestamp_to_iso(value: datetime | str | None) -> str:
    """Return a timestamp row value as an ISO-formatted string.

    A TIMESTAMP column in the database can be represented as a `datetime` object or an
    ISO-formatted string. This function ensures that the returned value is always
    an ISO-formatted string. If the input value is None, return an empty string.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return value


def context_to_bytes(context: Context) -> bytes:
    """Serialize `Context` to bytes."""
    return serde.context_to_proto(context).SerializeToString()


def context_from_bytes(context_bytes: bytes) -> Context:
    """Deserialize `Context` from bytes."""
    return serde.context_from_proto(ProtoContext.FromString(context_bytes))


def validate_task_event_data(data: str) -> None:
    """Validate that task event data is a JSON object string."""
    payload = strict_json_loads(data)
    if not isinstance(payload, dict):
        raise ValueError("Task event data must be a JSON object.")


def validate_task_message(message: Message) -> list[str]:  # pylint: disable=R0912
    """Validate a task Message."""
    validation_errors = []
    metadata = message.metadata

    if metadata.message_id == "":
        validation_errors.append("empty `metadata.message_id`")

    if metadata.run_id == 0:
        validation_errors.append("`metadata.run_id` is not set.")

    if metadata.src_task_id is None:
        validation_errors.append("`metadata.src_task_id` is not set.")

    if metadata.dst_task_id is None:
        validation_errors.append("`metadata.dst_task_id` is not set.")

    if metadata.src_task_id == metadata.dst_task_id:
        validation_errors.append(
            "`metadata.src_task_id` and `metadata.dst_task_id` must be different."
        )

    # Temporary: task messages are only supported in SuperLink for now.
    if metadata.src_node_id != SUPERLINK_NODE_ID:
        validation_errors.append(
            f"`metadata.src_node_id` is not {SUPERLINK_NODE_ID} (SuperLink node ID)"
        )

    if metadata.dst_node_id != SUPERLINK_NODE_ID:
        validation_errors.append(
            f"`metadata.dst_node_id` is not {SUPERLINK_NODE_ID} (SuperLink node ID)"
        )

    if metadata.created_at < _MIN_VALID_MESSAGE_CREATED_AT:
        validation_errors.append(
            "`metadata.created_at` must be a float that records the unix timestamp "
            "in seconds when the message was created."
        )

    if metadata.ttl <= 0:
        validation_errors.append("`metadata.ttl` must be higher than zero")
    elif metadata.created_at + metadata.ttl <= now().timestamp():
        validation_errors.append("Message TTL has expired")

    if metadata.message_type == "":
        validation_errors.append("`metadata.message_type` MUST be set")

    if message.has_content() == message.has_error():
        validation_errors.append(
            "Either message `content` or `error` MUST be set (but not both)"
        )

    return validation_errors
