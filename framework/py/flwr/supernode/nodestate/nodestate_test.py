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
from unittest.mock import patch

from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate.corestate_test import StateTest as CoreStateTest
from flwr.supercore.corestate.utils_test import create_task_message
from flwr.supercore.date import now
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import get_object_tree
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.run import Run

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
        state = InMemoryNodeState(ObjectStoreFactory().store())

        with self.assertRaises(ValueError):
            state.get_node_id()

    def test_store_task_message_uses_supernode_id(self) -> None:
        """Task Messages should use the node ID of their CoreState."""
        node_id = 123
        self.state.set_node_id(node_id)
        src_task_id = self.state.create_task(task_type=TaskType.CLIENT_APP, run_id=42)
        dst_task_id = self.state.create_task(task_type=TaskType.MODEL, run_id=42)
        assert src_task_id is not None and dst_task_id is not None
        message = create_task_message(
            src_task_id=src_task_id,
            dst_task_id=dst_task_id,
            run_id=42,
            node_id=node_id,
        )

        self.assertTrue(self.state.store_task_message(message))
        pulled = self.state.get_task_message(dst_task_ids=[dst_task_id])

        self.assertEqual(len(pulled), 1)
        self.assertEqual(pulled[0].metadata.src_node_id, node_id)
        self.assertEqual(pulled[0].metadata.dst_node_id, node_id)

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

    def test_store_message_and_object_tree(self) -> None:
        """Test storing a message and preregistering its object tree."""
        # Prepare
        msg = create_task_message()
        session_id = self.state.start_session(msg.metadata.run_id)

        # Execute
        stored, missing_objects = self.state.store_message_and_object_tree(
            msg, get_object_tree(msg), session_id
        )

        # Assert
        self.assertTrue(stored)
        self.assertIn(msg.metadata.message_id, missing_objects)
        self.assertTrue(msg.metadata.message_id in self.state.object_store)
        self.assertEqual(
            self.state.get_messages(dst_node_id=msg.metadata.dst_node_id)[0], msg
        )

    def test_push_session_expiry_cancels_message(self) -> None:
        """Test cancelling a Message belonging to an expired push session."""
        msg = create_task_message()
        self.state.store_messages([msg])
        session_id = self.state.start_session(msg.metadata.run_id)
        self.state.preregister_object_tree(get_object_tree(msg), session_id)

        self.state._cleanup_push_session(  # pylint: disable=protected-access
            session_id, cleanup_messages=True
        )

        self.assertEqual(
            self.state.get_messages(dst_node_id=msg.metadata.dst_node_id), []
        )

    def test_record_message_processing_timing(self) -> None:
        """Test recording message processing start and end times."""
        # Prepare
        msg = create_task_message()
        msg_id = msg.metadata.message_id
        self.state.store_messages([msg])

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
        """Test getting duration for non-existent message returns zero."""
        # Execute and assert
        msg_id = "non_existent_msg"
        with self.assertLogs("flwr", level="ERROR") as logs:
            duration = self.state.get_message_processing_duration(msg_id)

        self.assertEqual(duration, 0.0)
        self.assertIn(f"Message ID {msg_id} not found", logs.output[0])

    def test_record_message_processing_end_missing_start(self) -> None:
        """Test recording end time without start time logs an error."""
        # Execute and assert
        msg_id = "msg_without_start"
        with self.assertLogs("flwr", level="ERROR") as logs:
            self.state.record_message_processing_end(msg_id)
        self.assertIn(
            f"Cannot record end time: Message ID {msg_id} not found.",
            logs.output[0],
        )

    def test_get_message_processing_duration_incomplete_timing(self) -> None:
        """Test getting duration when only start time is recorded returns zero."""
        # Prepare
        msg = create_task_message()
        msg_id = msg.metadata.message_id
        self.state.store_messages([msg])

        self.state.record_message_processing_start(msg_id)

        # Execute and assert: should return zero since end time is missing
        with self.assertLogs("flwr", level="ERROR") as logs:
            duration = self.state.get_message_processing_duration(msg_id)
        self.assertEqual(duration, 0.0)
        self.assertIn(
            f"Start time or end time for message ID {msg_id} is missing.",
            logs.output[0],
        )

    def test_message_processing_timing_multiple_messages(self) -> None:
        """Test recording timing for multiple messages independently."""
        # Prepare
        msg1 = create_task_message()
        msg1_id = msg1.metadata.message_id
        self.state.store_messages([msg1])

        msg2 = create_task_message(dst_task_id=3)
        msg2_id = msg2.metadata.message_id
        self.state.store_messages([msg2])

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
        msg = create_task_message()
        msg_id = msg.metadata.message_id
        self.state.store_messages([msg])

        # Execute & Assert
        self.state.record_message_processing_start(msg_id)
        patched_dt = now() + timedelta(hours=12)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            with self.assertLogs("flwr", level="ERROR") as logs:
                self.assertEqual(
                    self.state.get_message_processing_duration(msg_id), 0.0
                )
            self.assertIn("is missing.", logs.output[0])
            self.state.record_message_processing_end(msg_id)
            duration = self.state.get_message_processing_duration(msg_id)

            assert duration is not None
            self.assertGreater(duration, 0.0)

    def test_cleanup_old_message_times(self) -> None:
        """Test that old message timing entries are cleaned up."""
        # Prepare
        msg = create_task_message()
        msg_id = msg.metadata.message_id
        self.state.store_messages([msg])

        # Record timing for an "old" completed message (2 hours ago)
        patched_dt = now() - timedelta(hours=2)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            self.state.record_message_processing_start(msg_id)
            # Finish 1 second later
            mock_dt.now.return_value = patched_dt + timedelta(seconds=1)
            self.state.record_message_processing_end(msg_id)

        # Assert: old message should be cleaned up and return zero
        with self.assertLogs("flwr", level="ERROR") as logs:
            duration = self.state.get_message_processing_duration(msg_id)

        # Verify it was cleaned up (not just missing end time)
        self.assertEqual(duration, 0.0)
        self.assertIn(f"Message ID {msg_id} not found.", logs.output[0])


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> NodeState:
        """Return InMemoryState."""
        state = InMemoryNodeState(ObjectStoreFactory().store())
        state.set_node_id(SUPERLINK_NODE_ID)
        return state
