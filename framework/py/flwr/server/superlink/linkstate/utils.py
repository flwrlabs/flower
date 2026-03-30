# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for State."""


from os import urandom
from typing import Any

from flwr.common import ConfigRecord, Context, Error, Message, Metadata, now, serde
from flwr.common.constant import (
    HEARTBEAT_PATIENCE,
    SUPERLINK_NODE_ID,
    ErrorCode,
    Status,
    SubStatus,
)
from flwr.common.message import make_message
from flwr.common.serde import recorddict_from_proto, recorddict_to_proto
from flwr.common.serde_utils import error_from_proto, error_to_proto
from flwr.common.typing import RunStatus

# pylint: disable=E0611
from flwr.proto.error_pb2 import Error as ProtoError
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.recorddict_pb2 import ConfigRecord as ProtoConfigRecord
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict
from flwr.supercore.constant import SYSTEM_MESSAGE_TYPE
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64

# pylint: enable=E0611
VALID_RUN_STATUS_TRANSITIONS = {
    (Status.PENDING, Status.STARTING),
    (Status.STARTING, Status.RUNNING),
    (Status.RUNNING, Status.FINISHED),
    # Any non-FINISHED status can transition to FINISHED
    (Status.PENDING, Status.FINISHED),
    (Status.STARTING, Status.FINISHED),
}
VALID_RUN_SUB_STATUSES = {
    SubStatus.COMPLETED,
    SubStatus.FAILED,
    SubStatus.STOPPED,
}
MESSAGE_UNAVAILABLE_ERROR_REASON = (
    "Error: Message Unavailable - The requested message could not be found in the "
    "database. It may have expired due to its TTL, been deleted because the "
    "destination SuperNode was removed from the federation, or never existed."
)
REPLY_MESSAGE_UNAVAILABLE_ERROR_REASON = (
    "Error: Reply Message Unavailable - The reply message has expired."
)
NODE_UNAVAILABLE_ERROR_REASON = (
    "Error: Node Unavailable — The destination node failed to report a heartbeat "
    f"within {HEARTBEAT_PATIENCE} × its expected interval."
)


def generate_rand_int_from_bytes(
    num_bytes: int, exclude: list[int] | None = None
) -> int:
    """Generate a random unsigned integer from `num_bytes` bytes.

    If `exclude` is set, this function guarantees such number is not returned.
    """
    num = int.from_bytes(urandom(num_bytes), "little", signed=False)

    if exclude:
        while num in exclude:
            num = int.from_bytes(urandom(num_bytes), "little", signed=False)
    return num


def convert_uint64_values_in_dict_to_sint64(
    data_dict: dict[str, int], keys: list[str]
) -> None:
    """Convert uint64 values to sint64 in the given dictionary.

    Parameters
    ----------
    data_dict : dict[str, int]
        A dictionary where the values are integers to be converted.
    keys : list[str]
        A list of keys in the dictionary whose values need to be converted.
    """
    for key in keys:
        if key in data_dict:
            data_dict[key] = uint64_to_int64(data_dict[key])


def convert_sint64_values_in_dict_to_uint64(
    data_dict: dict[str, int], keys: list[str]
) -> None:
    """Convert sint64 values to uint64 in the given dictionary.

    Parameters
    ----------
    data_dict : dict[str, int]
        A dictionary where the values are integers to be converted.
    keys : list[str]
        A list of keys in the dictionary whose values need to be converted.
    """
    for key in keys:
        if key in data_dict:
            data_dict[key] = int64_to_uint64(data_dict[key])


def context_to_bytes(context: Context) -> bytes:
    """Serialize `Context` to bytes."""
    return serde.context_to_proto(context).SerializeToString()


def context_from_bytes(context_bytes: bytes) -> Context:
    """Deserialize `Context` from bytes."""
    return serde.context_from_proto(ProtoContext.FromString(context_bytes))


def configrecord_to_bytes(config_record: ConfigRecord) -> bytes:
    """Serialize a `ConfigRecord` to bytes."""
    return serde.config_record_to_proto(config_record).SerializeToString()


def configrecord_from_bytes(configrecord_bytes: bytes) -> ConfigRecord:
    """Deserialize `ConfigRecord` from bytes."""
    return serde.config_record_from_proto(
        ProtoConfigRecord.FromString(configrecord_bytes)
    )


def is_valid_transition(current_status: RunStatus, new_status: RunStatus) -> bool:
    """Check if a transition between two run statuses is valid.

    Parameters
    ----------
    current_status : RunStatus
        The current status of the run.
    new_status : RunStatus
        The new status to transition to.

    Returns
    -------
    bool
        True if the transition is valid, False otherwise.
    """
    # Transition to FINISHED from a non-RUNNING status is only allowed
    # if the sub-status is not COMPLETED
    if (
        current_status.status in [Status.PENDING, Status.STARTING]
        and new_status.status == Status.FINISHED
    ):
        return new_status.sub_status != SubStatus.COMPLETED

    return (
        current_status.status,
        new_status.status,
    ) in VALID_RUN_STATUS_TRANSITIONS


def has_valid_sub_status(status: RunStatus) -> bool:
    """Check if the 'sub_status' field of the given status is valid.

    Parameters
    ----------
    status : RunStatus
        The status object to be checked.

    Returns
    -------
    bool
        True if the status object has a valid sub-status, False otherwise.

    Notes
    -----
    Only an empty string (i.e., "") is considered a valid sub-status for
    non-finished statuses. The sub-status of a finished status cannot be empty.
    """
    if status.status == Status.FINISHED:
        return status.sub_status in VALID_RUN_SUB_STATUSES
    return status.sub_status == ""


def create_message_error_unavailable_res_message(
    ins_metadata: Metadata, error_type: str
) -> Message:
    """Generate an error Message that the SuperLink returns carrying the specified
    error."""
    current_time = now().timestamp()
    ttl = max(ins_metadata.ttl - (current_time - ins_metadata.created_at), 0)
    metadata = Metadata(
        run_id=ins_metadata.run_id,
        message_id="",
        src_node_id=SUPERLINK_NODE_ID,
        dst_node_id=SUPERLINK_NODE_ID,
        reply_to_message_id=ins_metadata.message_id,
        group_id=ins_metadata.group_id,
        message_type=ins_metadata.message_type,
        created_at=current_time,
        ttl=ttl,
    )

    msg = make_message(
        metadata=metadata,
        error=Error(
            code=(
                ErrorCode.REPLY_MESSAGE_UNAVAILABLE
                if error_type == "msg_unavail"
                else ErrorCode.NODE_UNAVAILABLE
            ),
            reason=(
                REPLY_MESSAGE_UNAVAILABLE_ERROR_REASON
                if error_type == "msg_unavail"
                else NODE_UNAVAILABLE_ERROR_REASON
            ),
        ),
    )
    msg.metadata.__dict__["_message_id"] = msg.object_id
    return msg


def create_message_error_unavailable_ins_message(reply_to_message_id: str) -> Message:
    """Error to indicate that the enquired Message had expired before reply arrived or
    that it isn't found."""
    metadata = Metadata(
        run_id=0,  # Unknown
        message_id="",
        src_node_id=SUPERLINK_NODE_ID,
        dst_node_id=SUPERLINK_NODE_ID,
        reply_to_message_id=reply_to_message_id,
        group_id="",  # Unknown
        message_type=SYSTEM_MESSAGE_TYPE,
        created_at=now().timestamp(),
        ttl=0,
    )

    msg = make_message(
        metadata=metadata,
        error=Error(
            code=ErrorCode.MESSAGE_UNAVAILABLE,
            reason=MESSAGE_UNAVAILABLE_ERROR_REASON,
        ),
    )
    msg.metadata.__dict__["_message_id"] = msg.object_id
    return msg


def message_ttl_has_expired(message_metadata: Metadata, current_time: float) -> bool:
    """Check if the Message has expired."""
    return message_metadata.ttl + message_metadata.created_at < current_time


def verify_message_ids(
    inquired_message_ids: set[str],
    found_message_ins_dict: dict[str, Message],
    current_time: float | None = None,
    update_set: bool = True,
) -> dict[str, Message]:
    """Verify found Messages and generate error Messages for invalid ones.

    Parameters
    ----------
    inquired_message_ids : set[str]
        Set of Message IDs for which to generate error Message if invalid.
    found_message_ins_dict : dict[str, Message]
        Dictionary containing all found Message indexed by their IDs.
    current_time : Optional[float] (default: None)
        The current time to check for expiration. If set to `None`, the current time
        will automatically be set to the current timestamp using `now().timestamp()`.
    update_set : bool (default: True)
        If True, the `inquired_message_ids` will be updated to remove invalid ones,
        by default True.

    Returns
    -------
    dict[str, Message]
        A dictionary of error Message indexed by the corresponding ID of the message
        they are a reply of.
    """
    ret_dict = {}
    current = current_time if current_time else now().timestamp()
    for message_id in list(inquired_message_ids):
        # Generate error message if the inquired message doesn't exist or has expired
        message_ins = found_message_ins_dict.get(message_id)
        if message_ins is None or message_ttl_has_expired(
            message_ins.metadata, current
        ):
            if update_set:
                inquired_message_ids.remove(message_id)
            message_res = create_message_error_unavailable_ins_message(message_id)
            ret_dict[message_id] = message_res
    return ret_dict


def verify_found_message_replies(
    inquired_message_ids: set[str],
    found_message_ins_dict: dict[str, Message],
    found_message_res_list: list[Message],
    current_time: float | None = None,
    update_set: bool = True,
) -> dict[str, Message]:
    """Verify found Message replies and generate error Message for invalid ones.

    Parameters
    ----------
    inquired_message_ids : set[str]
        Set of Message IDs for which to generate error Message if invalid.
    found_message_ins_dict : dict[str, Message]
        Dictionary containing all found instruction Messages indexed by their IDs.
    found_message_res_list : dict[Message, Message]
        List of found Message to be verified.
    current_time : Optional[float] (default: None)
        The current time to check for expiration. If set to `None`, the current time
        will automatically be set to the current timestamp using `now().timestamp()`.
    update_set : bool (default: True)
        If True, the `inquired_message_ids` will be updated to remove ones
        that have a reply Message, by default True.

    Returns
    -------
    dict[str, Message]
        A dictionary of Message indexed by the corresponding Message ID.
    """
    ret_dict: dict[str, Message] = {}
    current = current_time if current_time else now().timestamp()
    for message_res in found_message_res_list:
        message_ins_id = message_res.metadata.reply_to_message_id
        if update_set:
            inquired_message_ids.remove(message_ins_id)
        # Check if the reply Message has expired
        if message_ttl_has_expired(message_res.metadata, current):
            # No need to insert the error Message
            message_res = create_message_error_unavailable_res_message(
                found_message_ins_dict[message_ins_id].metadata, "msg_unavail"
            )
        ret_dict[message_ins_id] = message_res
    return ret_dict


def check_node_availability_for_in_message(
    inquired_in_message_ids: set[str],
    found_in_message_dict: dict[str, Message],
    node_id_to_online_until: dict[int, float],
    current_time: float | None = None,
    update_set: bool = True,
) -> dict[str, Message]:
    """Check node availability for given Message and generate error reply Message if
    unavailable. A Message error indicating node unavailability will be generated for
    each given Message whose destination node is offline or non-existent.

    Parameters
    ----------
    inquired_in_message_ids : set[str]
        Set of Message IDs for which to check destination node availability.
    found_in_message_dict : dict[str, Message]
        Dictionary containing all found Message indexed by their IDs.
    node_id_to_online_until : dict[int, float]
        Dictionary mapping node IDs to their online-until timestamps.
    current_time : Optional[float] (default: None)
        The current time to check for expiration. If set to `None`, the current time
        will automatically be set to the current timestamp using `now().timestamp()`.
    update_set : bool (default: True)
        If True, the `inquired_in_message_ids` will be updated to remove invalid ones,
        by default True.

    Returns
    -------
    dict[str, Message]
        A dictionary of error Message indexed by the corresponding Message ID.
    """
    ret_dict = {}
    current = current_time if current_time else now().timestamp()
    for in_message_id in list(inquired_in_message_ids):
        in_message = found_in_message_dict[in_message_id]
        node_id = in_message.metadata.dst_node_id
        online_until = node_id_to_online_until.get(node_id)
        # Generate a reply message containing an error reply
        # if the node is offline or doesn't exist.
        if online_until is None or online_until < current:
            if update_set:
                inquired_in_message_ids.remove(in_message_id)
            reply_message = create_message_error_unavailable_res_message(
                in_message.metadata, "node_unavail"
            )
            ret_dict[in_message_id] = reply_message
    return ret_dict


def message_to_dict(message: Message) -> dict[str, Any]:
    """Transform Message to dict."""
    result = {
        "message_id": message.metadata.message_id,
        "group_id": message.metadata.group_id,
        "run_id": message.metadata.run_id,
        "src_node_id": message.metadata.src_node_id,
        "dst_node_id": message.metadata.dst_node_id,
        "reply_to_message_id": message.metadata.reply_to_message_id,
        "created_at": message.metadata.created_at,
        "delivered_at": message.metadata.delivered_at,
        "ttl": message.metadata.ttl,
        "message_type": message.metadata.message_type,
        "content": None,
        "error": None,
    }

    if message.has_content():
        result["content"] = recorddict_to_proto(message.content).SerializeToString()
    else:
        result["error"] = error_to_proto(message.error).SerializeToString()

    return result


def dict_to_message(message_dict: dict[str, Any]) -> Message:
    """Transform dict to Message."""
    content, error = None, None
    if (b_content := message_dict.pop("content", None)) is not None:
        content = recorddict_from_proto(ProtoRecordDict.FromString(b_content))
    if (b_error := message_dict.pop("error", None)) is not None:
        error = error_from_proto(ProtoError.FromString(b_error))

    # Metadata constructor doesn't allow passing created_at. We set it later
    metadata = Metadata(
        **{k: v for k, v in message_dict.items() if k not in ["delivered_at"]}
    )
    msg = make_message(metadata=metadata, content=content, error=error)
    msg.metadata.delivered_at = message_dict.get("delivered_at", "")
    return msg
