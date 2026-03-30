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
"""Metadata."""


from __future__ import annotations

from typing import cast

from flwr.app.message_type import MessageType
from flwr.supercore.constant import SYSTEM_MESSAGE_TYPE

from ..common.constant import MessageTypeLegacy


class Metadata:  # pylint: disable=too-many-instance-attributes
    """The class representing metadata associated with the current message.

    Parameters
    ----------
    run_id : int
        An identifier for the current run.
    message_id : str
        An identifier for the current message.
    src_node_id : int
        An identifier for the node sending this message.
    dst_node_id : int
        An identifier for the node receiving this message.
    reply_to_message_id : str
        An identifier for the message to which this message is a reply.
    group_id : str
        An identifier for grouping messages. In some settings,
        this is used as the FL round.
    created_at : float
        Unix timestamp when the message was created.
    ttl : float
        Time-to-live for this message in seconds.
    message_type : str
        A string that encodes the action to be executed on
        the receiving end.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        run_id: int,
        message_id: str,
        src_node_id: int,
        dst_node_id: int,
        reply_to_message_id: str,
        group_id: str,
        created_at: float,
        ttl: float,
        message_type: str,
    ) -> None:
        var_dict = {
            "_run_id": run_id,
            "_message_id": message_id,
            "_src_node_id": src_node_id,
            "_dst_node_id": dst_node_id,
            "_reply_to_message_id": reply_to_message_id,
            "_group_id": group_id,
            "_created_at": created_at,
            "_ttl": ttl,
            "_message_type": message_type,
        }
        self.__dict__.update(var_dict)
        self.message_type = message_type  # Trigger validation

    @property
    def run_id(self) -> int:
        """An identifier for the current run."""
        return cast(int, self.__dict__["_run_id"])

    @property
    def message_id(self) -> str:
        """An identifier for the current message."""
        return cast(str, self.__dict__["_message_id"])

    @property
    def src_node_id(self) -> int:
        """An identifier for the node sending this message."""
        return cast(int, self.__dict__["_src_node_id"])

    @property
    def reply_to_message_id(self) -> str:
        """An identifier for the message to which this message is a reply."""
        return cast(str, self.__dict__["_reply_to_message_id"])

    @property
    def dst_node_id(self) -> int:
        """An identifier for the node receiving this message."""
        return cast(int, self.__dict__["_dst_node_id"])

    @dst_node_id.setter
    def dst_node_id(self, value: int) -> None:
        """Set dst_node_id."""
        self.__dict__["_dst_node_id"] = value

    @property
    def group_id(self) -> str:
        """An identifier for grouping messages."""
        return cast(str, self.__dict__["_group_id"])

    @group_id.setter
    def group_id(self, value: str) -> None:
        """Set group_id."""
        self.__dict__["_group_id"] = value

    @property
    def created_at(self) -> float:
        """Unix timestamp when the message was created."""
        return cast(float, self.__dict__["_created_at"])

    @created_at.setter
    def created_at(self, value: float) -> None:
        """Set creation timestamp of this message."""
        self.__dict__["_created_at"] = value

    @property
    def delivered_at(self) -> str:
        """Unix timestamp when the message was delivered."""
        return cast(str, self.__dict__["_delivered_at"])

    @delivered_at.setter
    def delivered_at(self, value: str) -> None:
        """Set delivery timestamp of this message."""
        self.__dict__["_delivered_at"] = value

    @property
    def ttl(self) -> float:
        """Time-to-live for this message."""
        return cast(float, self.__dict__["_ttl"])

    @ttl.setter
    def ttl(self, value: float) -> None:
        """Set ttl."""
        self.__dict__["_ttl"] = value

    @property
    def message_type(self) -> str:
        """A string that encodes the action to be executed on the receiving end."""
        return cast(str, self.__dict__["_message_type"])

    @message_type.setter
    def message_type(self, value: str) -> None:
        """Set message_type."""
        # Validate message type
        if validate_legacy_message_type(value):
            pass  # Backward compatibility for legacy message types
        elif not validate_message_type(value):
            raise ValueError(
                f"Invalid message type: '{value}'. "
                "Expected format: '<category>' or '<category>.<action>', "
                "where <category> must be 'train', 'evaluate', or 'query', "
                "and <action> must be a valid Python identifier."
            )

        self.__dict__["_message_type"] = value

    def __repr__(self) -> str:
        """Return a string representation of this instance."""
        view = ", ".join([f"{k.lstrip('_')}={v!r}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__qualname__}({view})"

    def __eq__(self, other: object) -> bool:
        """Compare two instances of the class."""
        if not isinstance(other, self.__class__):
            raise NotImplementedError
        return self.__dict__ == other.__dict__


def validate_message_type(message_type: str) -> bool:
    """Validate if the message type is valid.

    A valid message type format must be one of the following:

    - "<category>"
    - "<category>.<action>"

    where `category` must be one of "train", "evaluate", or "query",
    and `action` must be a valid Python identifier.
    """
    # Check if conforming to the format "<category>"
    valid_types = {
        MessageType.TRAIN,
        MessageType.EVALUATE,
        MessageType.QUERY,
        SYSTEM_MESSAGE_TYPE,
    }
    if message_type in valid_types:
        return True

    # Check if conforming to the format "<category>.<action>"
    if message_type.count(".") != 1:
        return False

    category, action = message_type.split(".")
    if category in valid_types and action.isidentifier():
        return True

    return False


def validate_legacy_message_type(message_type: str) -> bool:
    """Validate if the legacy message type is valid."""
    # Backward compatibility for legacy message types
    if message_type in (
        MessageTypeLegacy.GET_PARAMETERS,
        MessageTypeLegacy.GET_PROPERTIES,
        "reconnect",
    ):
        return True

    return False
