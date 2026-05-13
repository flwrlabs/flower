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


from os import urandom
from typing import Any

from flwr.app.metadata import Metadata
from flwr.common import Message
from flwr.common.message import make_message
from flwr.common.serde import recorddict_from_proto, recorddict_to_proto
from flwr.common.serde_utils import error_from_proto, error_to_proto
from flwr.proto.error_pb2 import Error as ProtoError  # pylint: disable=E0611
from flwr.proto.recorddict_pb2 import (
    RecordDict as ProtoRecordDict,  # pylint: disable=E0611
)


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


def task_message_to_dict(message: Message, run_id: int) -> dict[str, Any]:
    """Transform a task-addressed Message to a storage dictionary."""
    result = {
        "message_id": message.metadata.message_id,
        "run_id": run_id,
        "src_task_id": message.metadata.src_node_id,
        "dst_task_id": message.metadata.dst_node_id,
        "reply_to_message_id": message.metadata.reply_to_message_id,
        "created_at": message.metadata.created_at,
        "ttl": message.metadata.ttl,
        "message_type": message.metadata.message_type,
        "content": None,
        "error": None,
    }

    if message.has_content():
        result["content"] = recorddict_to_proto(message.content).SerializeToString()
    elif message.has_error():
        result["error"] = error_to_proto(message.error).SerializeToString()

    return result


def task_message_from_dict(message_dict: dict[str, Any]) -> Message:
    """Transform a storage dictionary to a task-addressed Message."""
    content, error = None, None
    if (b_content := message_dict.pop("content", None)) is not None:
        content = recorddict_from_proto(ProtoRecordDict.FromString(b_content))
    if (b_error := message_dict.pop("error", None)) is not None:
        error = error_from_proto(ProtoError.FromString(b_error))

    metadata = Metadata(
        run_id=message_dict["run_id"],
        message_id=message_dict["message_id"],
        src_node_id=message_dict["src_task_id"],
        dst_node_id=message_dict["dst_task_id"],
        reply_to_message_id=message_dict["reply_to_message_id"] or "",
        group_id="",
        created_at=message_dict["created_at"],
        ttl=message_dict["ttl"],
        message_type=message_dict["message_type"],
    )
    return make_message(metadata=metadata, content=content, error=error)


def has_valid_task_message_payload(message: Message) -> bool:
    """Return True if the task message carries the required payload fields."""
    return (
        message.metadata.message_id != ""
        and message.metadata.src_node_id != 0
        and message.metadata.dst_node_id != 0
        and message.metadata.ttl > 0
        and message.metadata.message_type != ""
        and message.has_content() != message.has_error()
    )
