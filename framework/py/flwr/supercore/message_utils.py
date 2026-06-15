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
"""Internal Message helpers for transport and persistence boundaries."""


from typing import cast
from uuid import uuid4

from flwr.app.message import Message

MESSAGE_OBJECT_ID_KEY = "_message_object_id"


def set_message_id(message: Message, message_id: str) -> None:
    """Set a Message's logical message ID."""
    message.metadata.__dict__["_message_id"] = message_id


def assign_message_id(message: Message) -> None:
    """Assign a fresh logical UUID message ID."""
    set_message_id(message, str(uuid4()))


def set_message_object_id(message: Message, object_id: str) -> None:
    """Set the ObjectStore root ID associated with a Message."""
    message.__dict__[MESSAGE_OBJECT_ID_KEY] = object_id


def get_message_object_id(message: Message) -> str:
    """Get the ObjectStore root ID associated with a Message."""
    if MESSAGE_OBJECT_ID_KEY in message.__dict__:
        return cast(str, message.__dict__[MESSAGE_OBJECT_ID_KEY])
    return message.object_id
