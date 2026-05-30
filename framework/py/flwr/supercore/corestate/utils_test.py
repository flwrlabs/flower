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
"""CoreState utils tests."""


import unittest

from flwr.app import Error, Message, Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common import now
from flwr.common.constant import SUPERLINK_NODE_ID

from .utils import generate_rand_int_from_bytes, validate_task_message


def create_task_message(  # pylint: disable=too-many-arguments
    src_task_id: int | None = 1,
    dst_task_id: int | None = 2,
    run_id: int = 1,
    *,
    reply_to_message_id: str = "",
    created_at: float | None = None,
    ttl: float = 60.0,
    message_type: str = "train",
    content: RecordDict | None = None,
    has_error: bool = False,
) -> Message:
    """Create a task Message for testing."""
    metadata = Metadata(
        run_id=run_id,
        message_id="",
        src_node_id=SUPERLINK_NODE_ID,
        dst_node_id=SUPERLINK_NODE_ID,
        reply_to_message_id=reply_to_message_id,
        group_id="",
        created_at=created_at if created_at is not None else now().timestamp(),
        ttl=ttl,
        message_type=message_type,
        src_task_id=src_task_id,
        dst_task_id=dst_task_id,
    )

    if has_error:
        msg = make_message(metadata=metadata, error=Error(0))
    else:
        msg = make_message(
            metadata=metadata,
            content=content if content is not None else RecordDict(),
        )
    msg.metadata.__dict__["_message_id"] = msg.object_id
    return msg


def _assert_has_error(errors: list[str], expected: str) -> None:
    """Assert that an expected validation error fragment exists."""
    assert any(expected in error for error in errors)


class UtilsTest(unittest.TestCase):
    """Test CoreState utils code."""

    def test_generate_rand_int_from_bytes_unsigned_int(self) -> None:
        """Test that the generated integer is unsigned (non-negative)."""
        for num_bytes in range(1, 9):
            with self.subTest(num_bytes=num_bytes):
                rand_int = generate_rand_int_from_bytes(num_bytes)
                self.assertGreaterEqual(rand_int, 0)

    def test_validate_task_message_accepts_valid_message(self) -> None:
        """Test that a valid task message passes validation."""
        for has_error in [False, True]:
            with self.subTest(has_error=has_error):
                message = create_task_message(has_error=has_error)

                self.assertEqual(validate_task_message(message), [])

    def test_validate_task_message_rejects_missing_message_id(self) -> None:
        """Test that message_id must be set."""
        message = create_task_message()
        message.metadata.__dict__["_message_id"] = ""

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.message_id")

    def test_validate_task_message_rejects_unset_run_id(self) -> None:
        """Test that run_id must be set."""
        message = create_task_message(run_id=0)

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.run_id")

    def test_validate_task_message_rejects_missing_src_task_id(self) -> None:
        """Test that source task ID must be set."""
        message = create_task_message(src_task_id=None)

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.src_task_id")

    def test_validate_task_message_rejects_missing_dst_task_id(self) -> None:
        """Test that destination task ID must be set."""
        message = create_task_message(dst_task_id=None)

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.dst_task_id")

    def test_validate_task_message_rejects_same_task_ids(self) -> None:
        """Test that source and destination task IDs must differ."""
        message = create_task_message(src_task_id=1, dst_task_id=1)

        errors = validate_task_message(message)

        _assert_has_error(errors, "must be different")

    def test_validate_task_message_rejects_invalid_created_at(self) -> None:
        """Test that created_at must be a plausible timestamp."""
        message = create_task_message(created_at=0.0)

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.created_at")

    def test_validate_task_message_rejects_non_positive_ttl(self) -> None:
        """Test that ttl must be positive."""
        message = create_task_message(ttl=0.0)

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.ttl")

    def test_validate_task_message_rejects_expired_ttl(self) -> None:
        """Test that task messages must not be expired."""
        message = create_task_message(created_at=now().timestamp() - 10.0, ttl=1.0)

        errors = validate_task_message(message)

        _assert_has_error(errors, "TTL has expired")

    def test_validate_task_message_rejects_missing_message_type(self) -> None:
        """Test that message_type must be set."""
        message = create_task_message()
        message.metadata.__dict__["_message_type"] = ""

        errors = validate_task_message(message)

        _assert_has_error(errors, "metadata.message_type")

    def test_validate_task_message_rejects_missing_content_and_error(self) -> None:
        """Test that either content or error must be set."""
        message = create_task_message()
        message.__dict__["_content"] = None

        errors = validate_task_message(message)

        _assert_has_error(errors, "content` or `error")

    def test_validate_task_message_rejects_content_and_error(self) -> None:
        """Test that content and error must not both be set."""
        message = create_task_message()
        message.__dict__["_error"] = Error(0)

        errors = validate_task_message(message)

        _assert_has_error(errors, "content` or `error")
