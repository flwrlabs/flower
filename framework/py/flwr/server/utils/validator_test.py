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
"""Validator tests."""


import unittest
from uuid import uuid4

from parameterized import parameterized

from flwr.app.message import make_message
from flwr.common import DEFAULT_TTL, Error, Message, Metadata, RecordDict, now
from flwr.common.constant import SUPERLINK_NODE_ID

from .validator import validate_message


def create_message(  # pylint: disable=R0913, R0917
    message_id: str = str(uuid4()),
    src_node_id: int = SUPERLINK_NODE_ID,
    dst_node_id: int = 456,
    ttl: int = DEFAULT_TTL,
    reply_to_message_id: str = "",
    has_content: bool = True,
    has_error: bool = False,
    msg_type: str = "mock",
) -> Message:
    """Create a Message for testing.

    By default, it creates a valid instruction message containing a RecordDict.
    """
    metadata = Metadata(
        run_id=0,
        message_id=message_id,
        src_node_id=src_node_id,
        dst_node_id=dst_node_id,
        reply_to_message_id=reply_to_message_id,
        group_id="",
        created_at=now().timestamp(),
        ttl=ttl,
        message_type="train",  # Bypass message type validation
    )
    metadata.__dict__["_message_type"] = msg_type
    ret = make_message(metadata=metadata, content=RecordDict())
    if not has_content:
        ret.__dict__["_content"] = None
    if has_error:
        ret.__dict__["_error"] = Error(0)
    return ret


class ValidatorTest(unittest.TestCase):
    """Test validation code in state."""

    @parameterized.expand(  # type: ignore
        [
            # Valid messages
            (create_message(), False, False),
            (create_message(has_content=False, has_error=True), False, False),
            # `message_id` is not set
            (create_message(message_id=""), False, True),
            # `ttl` is zero
            (create_message(ttl=0), False, True),
            # `src_node_id` is not set
            (create_message(src_node_id=0), False, True),
            # `dst_node_id` is not set
            (create_message(dst_node_id=0), False, True),
            # `dst_node_id` is SUPERLINK
            (create_message(dst_node_id=SUPERLINK_NODE_ID), False, True),
            # `message_type` is not set
            (create_message(msg_type=""), False, True),
            # Both `content` and `error` are not set
            (create_message(has_content=False), False, True),
            # Both `content` and `error` are set
            (create_message(has_error=True), False, True),
            # `reply_to_message_id` is set in a non-reply message
            (create_message(reply_to_message_id="789"), False, True),
            # `reply_to_message_id` is not set in reply message
            (create_message(), True, True),
            # `dst_node_id` is not SuperLink in reply message
            (create_message(src_node_id=123, reply_to_message_id="blabla"), True, True),
        ]
    )
    def test_message(self, message: Message, is_reply: bool, should_fail: bool) -> None:
        """Test is_valid message."""
        # Execute
        val_errors = validate_message(message, is_reply_message=is_reply)

        # Assert
        if should_fail:
            self.assertTrue(val_errors)
        else:
            self.assertFalse(val_errors)
