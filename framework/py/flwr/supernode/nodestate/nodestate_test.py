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
"""Tests all NodeState implementations have to conform to."""


import hashlib
from datetime import timedelta
from typing import Any
from unittest.mock import patch

from parameterized import parameterized

from flwr.common import ConfigRecord, Context, Message, Metadata, RecordDict, now
from flwr.common.constant import ErrorCode
from flwr.common.message import make_message
from flwr.common.typing import Fab, Run
from flwr.supercore.corestate.corestate_test import StateTest as CoreStateTest
from flwr.supercore.object_store import ObjectStoreFactory

from . import InMemoryNodeState, NodeState


class StateTest(CoreStateTest):  # pylint: disable=R0904
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    def setUp(self) -> None:
        """Set up the test case."""
        self.state: NodeState = self.state_factory()

    def state_factory(self) -> NodeState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_get_set_node_id(self) -> None:
        """Test set_node_id."""
        # Prepare
        node_id = 123

        # Execute
        self.state.set_node_id(node_id)

        retrieved_node_id = self.state.get_node_id()

        # Assert
        assert node_id == retrieved_node_id

    def test_get_node_id_fails(self) -> None:
        """Test get_node_id fails correctly if node_id is not set."""
        # Execute and assert
        with self.assertRaises(ValueError):
            self.state.get_node_id()

    def test_store_and_get_run(self) -> None:
        """Test storing and retrieving a run."""
        # Prepare
        run = Run.create_empty(61016)
        self.state.store_run(run)

        # Execute
        retrieved = self.state.get_run(61016)

        # Assert
        self.assertEqual(retrieved, run)

    def test_store_and_get_fab(self) -> None:
        """Test storing and retrieving a FAB."""
        content = b"fab-content"
        fab = Fab(hashlib.sha256(content).hexdigest(), content, {"meta": "data"})

        fab_hash = self.state.store_fab(fab)
        retrieved = self.state.get_fab(fab_hash)

        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.hash_str, fab_hash)
        self.assertEqual(retrieved.content, fab.content)
        self.assertEqual(retrieved.verifications, fab.verifications)

        # Retrieved FAB should be a defensive copy.
        retrieved.verifications["meta"] = "mutated"
        reloaded = self.state.get_fab(fab_hash)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.verifications, {"meta": "data"})

        # Also verify write-time hash validation rejects mismatched hashes.
        with self.assertRaisesRegex(ValueError, "FAB hash mismatch"):
            self.state.store_fab(Fab("not-the-content-hash", b"fab-content", {}))

    def test_store_and_get_context(self) -> None:
        """Test storing and retrieving a context."""
        # Prepare
        ctx = Context(
            run_id=99,
            node_id=1,
            node_config={"key1": "value1"},
            state=RecordDict({"cfg": ConfigRecord({"key2": "value2"})}),
            run_config={"key3": "value3"},
        )
        self.state.store_context(ctx)

        # Execute
        retrieved = self.state.get_context(99)

        # Assert
        self.assertEqual(retrieved, ctx)

    def test_store_and_get_message_basic(self) -> None:
        """Test storing and retrieving a message."""
        # Prepare
        msg = make_dummy_message(msg_id="test_msg")

        # Execute
        self.state.store_message(msg)

        # Basic retrieval with no filters
        retrieved_msg = self.state.get_messages()[0]

        self.assertIn("test_msg", retrieved_msg.metadata.message_id)
        self.assertEqual(retrieved_msg, msg)

        # Ensure message won't be retrieved again
        result = self.state.get_messages()
        self.assertEqual(len(result), 0)

    @parameterized.expand(  # type: ignore
        [
            ({"run_ids": [1]}, {"msg1", "msg2"}),
            ({"run_ids": [1], "is_reply": False}, {"msg2"}),
            ({"run_ids": [1], "limit": 1}, {"msg1", "msg2"}),
            ({"run_ids": [2, 3]}, {"msg3", "msg4"}),
            ({"is_reply": True}, {"msg1", "msg4"}),
            ({"is_reply": True, "limit": 1}, {"msg1", "msg4"}),
        ]
    )
    def test_get_message_with_filters(
        self, filters: dict[str, Any], expected: set[str]
    ) -> None:
        """Test retrieving messages with various filters."""
        # Prepare
        # Run 1: 1 instruction, 1 reply
        self.state.store_message(make_dummy_message(1, True, "msg1"))
        self.state.store_message(make_dummy_message(1, False, "msg2"))
        # Run 2: 1 instruction
        self.state.store_message(make_dummy_message(2, False, "msg3"))
        # Run 3: 1 reply
        self.state.store_message(make_dummy_message(3, True, "msg4"))

        # Execute
        result = self.state.get_messages(**filters)
        result_ids = {msg.metadata.message_id for msg in result}

        # Assert
        if (limit := filters.get("limit")) is not None:
            self.assertEqual(len(result), limit)
            self.assertTrue(result_ids.issubset(expected))
        else:
            self.assertEqual(result_ids, expected)

    def test_delete_message(self) -> None:
        """Test deleting messages."""
        # Prepare
        msg1 = make_dummy_message(msg_id="msg1")
        msg2 = make_dummy_message(msg_id="msg2")
        self.state.store_message(msg1)
        self.state.store_message(msg2)

        # Execute: delete one message
        self.state.delete_messages(message_ids=["msg1"])

        # Assert: msg1 should be deleted, msg2 should remain
        msgs = self.state.get_messages()
        msg_ids = {msg.metadata.message_id for msg in msgs}
        self.assertNotIn("msg1", msg_ids)
        self.assertIn("msg2", msg_ids)

    def test_get_run_ids_with_pending_messages(self) -> None:
        """Test retrieving run IDs with pending messages."""
        # Prepare: store messages for runs 1, 2, and 3
        # Run 1 has a pending message, run 2 has a token, run 3 has a reply,
        # run 4 has a retrieved message (not pending),
        #  and run 5 was assigned a token but was later deleted due to
        # `flwr-clientapp` finishing the handling of a message.
        self.state.store_message(make_dummy_message(1, False, "msg1"))
        self.state.store_message(make_dummy_message(2, False, "msg2"))
        self.state.store_message(make_dummy_message(3, True, "msg3"))
        self.state.store_message(make_dummy_message(4, False, "msg4"))
        self.state.store_message(make_dummy_message(5, False, "msg5"))
        self.state.get_messages(run_ids=[4])
        self.state.create_token(2)
        self.state.create_token(5)
        self.state.delete_token(5)

        # Execute
        run_ids = self.state.get_run_ids_with_pending_messages()

        # Assert: run 1 and run 5 should be returned
        self.assertEqual(set(run_ids), {1, 5})

    def test_get_error_reply_when_token_expires(self) -> None:
        """Test that error replies are created when tokens expire."""
        # Prepare: Create a token for a run
        run_id = 110
        created_at = now()
        token = self.state.create_token(run_id)
        assert token is not None

        # Prepare: store and retrieve a message for the run
        msg = make_dummy_message(run_id=run_id)
        self.state.store_message(msg)
        assert self.state.get_messages(run_ids=[run_id])

        # Execute: retrieve
        with patch("datetime.datetime") as mock_datetime:
            # Simulate time passage beyond token TTL
            mock_datetime.now.return_value = created_at + timedelta(seconds=1e5)

            # Retrieve replies
            replies = self.state.get_messages(is_reply=True)

        # Assert: error replies should be created for the message
        self.assertEqual(len(replies), 1)
        self.assertEqual(replies[0].metadata.reply_to_message_id, msg.object_id)
        self.assertTrue(replies[0].has_error())
        self.assertEqual(replies[0].error.code, ErrorCode.CLIENT_APP_CRASHED)

    def test_record_message_processing_timing(self) -> None:
        """Test recording message processing start and end times."""
        # Prepare
        msg_id = "test_msg_123"
        msg = make_dummy_message(msg_id=msg_id)
        self.state.store_message(msg)

        # Execute: record start time
        self.state.record_message_processing_start(msg_id)

        patched_dt = now() + timedelta(seconds=10)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt

            # Execute: record end time
            self.state.record_message_processing_end(msg_id)

            # Execute: get duration
            duration = self.state.get_message_processing_duration(msg_id)

            # Assert
            assert duration is not None
            self.assertGreater(duration, 0.0)

    def test_get_message_processing_duration_missing_message(self) -> None:
        """Test getting duration for non-existent message raises error."""
        # Execute and assert
        msg_id = "non_existent_msg"
        with self.assertRaises(ValueError) as ctx:
            self.state.get_message_processing_duration(msg_id)
        self.assertIn(f"Message ID {msg_id} not found", str(ctx.exception))

    def test_record_message_processing_end_missing_start(self) -> None:
        """Test recording end time without start time raises error."""
        # Execute and assert
        msg_id = "msg_without_start"
        with self.assertRaises(ValueError) as ctx:
            self.state.record_message_processing_end(msg_id)
        self.assertIn(
            f"Cannot record end time: Message ID {msg_id} not found.",
            str(ctx.exception),
        )

    def test_get_message_processing_duration_incomplete_timing(self) -> None:
        """Test getting duration when only start time is recorded raises error."""
        # Prepare
        msg_id = "incomplete_msg"
        msg = make_dummy_message(msg_id=msg_id)
        self.state.store_message(msg)

        self.state.record_message_processing_start(msg_id)

        # Execute and assert: should raise error since end time is missing
        with self.assertRaises(ValueError) as ctx:
            self.state.get_message_processing_duration(msg_id)
        self.assertIn(
            f"Start time or end time for message ID {msg_id} is missing.",
            str(ctx.exception),
        )

    def test_message_processing_timing_multiple_messages(self) -> None:
        """Test recording timing for multiple messages independently."""
        # Prepare
        msg1_id = "msg1"
        msg1 = make_dummy_message(msg_id=msg1_id)
        self.state.store_message(msg1)

        msg2_id = "msg2"
        msg2 = make_dummy_message(msg_id=msg2_id)
        self.state.store_message(msg2)

        # Execute: record timing for first message
        self.state.record_message_processing_start(msg1_id)
        patched_dt_1 = now() + timedelta(seconds=10)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt_1
            self.state.record_message_processing_end(msg1_id)

        # Execute: record timing for second message
        self.state.record_message_processing_start(msg2_id)

        patched_dt_2 = now() + timedelta(seconds=20)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt_2
            self.state.record_message_processing_end(msg2_id)

        # Get durations
        duration1 = self.state.get_message_processing_duration(msg1_id)
        duration2 = self.state.get_message_processing_duration(msg2_id)

        # Assert
        assert duration1 is not None
        assert duration2 is not None
        self.assertGreater(duration2, duration1)

    def test_long_running_message_processing_not_cleaned_up(self) -> None:
        """Test that long-running message timing entries are not cleaned up."""
        # Prepare
        msg_id = "test_msg_123"
        msg = make_dummy_message(msg_id=msg_id)
        self.state.store_message(msg)

        # Execute & Assert
        self.state.record_message_processing_start(msg_id)
        patched_dt = now() + timedelta(hours=12)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            self.state.record_message_processing_end(msg_id)
            duration = self.state.get_message_processing_duration(msg_id)

            assert duration is not None
            self.assertGreater(duration, 0.0)

    def test_cleanup_old_message_times(self) -> None:
        """Test that old message timing entries are cleaned up."""
        # Prepare
        msg_id = "old_test_msg_123"
        msg = make_dummy_message(msg_id=msg_id)
        self.state.store_message(msg)

        # Record timing for an "old" completed message (2 hours ago)
        patched_dt = now() - timedelta(hours=2)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            self.state.record_message_processing_start(msg_id)
            # Finish 1 second later
            mock_dt.now.return_value = patched_dt + timedelta(seconds=1)
            self.state.record_message_processing_end(msg_id)

        # Assert: old message should be cleaned up and raise error
        with self.assertRaises(ValueError) as ctx:
            self.state.get_message_processing_duration(msg_id)

        # Verify it was cleaned up (not just missing end time)
        self.assertIn(f"Message ID {msg_id} not found.", str(ctx.exception))

    def test_cleanup_orphaned_message_times(self) -> None:
        """Test that timing entries without corresponding messages are cleaned up."""
        # Prepare: create a timing entry
        orphan_msg_id = "orphaned_msg_123"
        msg = make_dummy_message(msg_id=orphan_msg_id)
        self.state.store_message(msg)
        self.state.record_message_processing_start(orphan_msg_id)
        self.state.record_message_processing_end(orphan_msg_id)

        # Delete the message from msg_store (simulating orphaned timing entry)
        self.state.delete_messages(message_ids=[orphan_msg_id])

        # Create another message to trigger cleanup
        other_msg_id = "other_msg"
        other_msg = make_dummy_message(msg_id=other_msg_id)
        self.state.store_message(other_msg)
        self.state.record_message_processing_start(other_msg_id)
        self.state.record_message_processing_end(other_msg_id)

        # Execute: accessing duration should trigger cleanup
        self.state.get_message_processing_duration(other_msg_id)

        # Assert: orphaned message should be cleaned up
        with self.assertRaises(ValueError) as ctx:
            self.state.get_message_processing_duration(orphan_msg_id)
        self.assertIn(f"Message ID {orphan_msg_id} not found.", str(ctx.exception))


def make_dummy_message(
    run_id: int = 110, is_reply: bool = False, msg_id: str = ""
) -> Message:
    """Create a dummy message for testing."""
    metadata = Metadata(
        run_id=run_id,
        # This is for testing purposes, in a real scenario this would be `.object_id`
        message_id=msg_id,
        src_node_id=0,
        dst_node_id=120,
        reply_to_message_id="mock id" if is_reply else "",
        group_id="Mock mock",
        created_at=123456789,
        ttl=999,
        message_type="query",
    )
    content = RecordDict({"cfg": ConfigRecord({"key": "value"})})
    msg = make_message(metadata, content)
    # Set message ID if not provided
    if msg_id == "":
        # pylint: disable-next=W0212
        msg.metadata._message_id = msg.object_id  # type: ignore
    return msg


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> NodeState:
        """Return InMemoryState."""
        return InMemoryNodeState(ObjectStoreFactory().store())
