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
"""Validators."""


from flwr.app import Message
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.date import now


# pylint: disable-next=too-many-branches
def validate_message(message: Message, is_reply_message: bool) -> list[str]:
    """Validate a Message."""
    validation_errors = []
    metadata = message.metadata

    if metadata.message_id == "":
        validation_errors.append("empty `metadata.message_id`")

    # Created/delivered/TTL/Pushed
    if (
        metadata.created_at < 1740700800.0
    ):  # unix timestamp of 28 February 2025 00h:00m:00s UTC
        validation_errors.append(
            "`metadata.created_at` must be a float that records the unix timestamp "
            "in seconds when the message was created."
        )
    if metadata.delivered_at != "":
        validation_errors.append("`metadata.delivered_at` must be an empty str")
    if metadata.ttl <= 0:
        validation_errors.append("`metadata.ttl` must be higher than zero")

    # Verify TTL and created_at time
    current_time = now().timestamp()
    if metadata.created_at + metadata.ttl <= current_time:
        validation_errors.append("Message TTL has expired")

    # Source node is set and is not zero
    if not metadata.src_node_id:
        validation_errors.append("`metadata.src_node_id` is not set.")

    # Destination node is set and is not zero
    if not metadata.dst_node_id:
        validation_errors.append("`metadata.dst_node_id` is not set.")

    # Message type
    if metadata.message_type == "":
        validation_errors.append("`metadata.message_type` MUST be set")

    # Content
    if not message.has_content() != message.has_error():
        validation_errors.append(
            "Either message `content` or `error` MUST be set (but not both)"
        )

    # Link respose to original message
    if not is_reply_message:
        if metadata.reply_to_message_id != "":
            validation_errors.append("`metadata.reply_to_message_id` MUST not be set.")
        if metadata.src_node_id != SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.src_node_id` is not {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )
        if metadata.dst_node_id == SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.dst_node_id` is {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )
    else:
        if metadata.reply_to_message_id == "":
            validation_errors.append("`metadata.reply_to_message_id` MUST be set.")
        if metadata.src_node_id == SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.src_node_id` is {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )
        if metadata.dst_node_id != SUPERLINK_NODE_ID:
            validation_errors.append(
                f"`metadata.dst_node_id` is not {SUPERLINK_NODE_ID} (SuperLink node ID)"
            )

    return validation_errors
