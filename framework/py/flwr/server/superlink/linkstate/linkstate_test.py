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
"""Tests all LinkState implemenations have to conform to."""
# pylint: disable=invalid-name, too-many-lines, R0904, R0913


import hashlib
import multiprocessing
import os
import secrets
import shutil
import tempfile
import threading
import time
import unittest
from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

from parameterized import parameterized

from flwr.app.user_config import UserConfig
from flwr.common import DEFAULT_TTL, Context, Error, Message, RecordDict, now
from flwr.common.constant import (
    HEARTBEAT_DEFAULT_INTERVAL,
    RUN_FAILURE_DETAILS_NO_HEARTBEAT,
    SUPERLINK_NODE_ID,
    ErrorCode,
    Status,
    SubStatus,
)
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.common.typing import Fab, RunStatus
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611

# pylint: disable=E0611
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.proto.message_pb2 import Metadata as ProtoMetadata
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict

# pylint: enable=E0611
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState, SqlLinkState
from flwr.supercore.constant import NOOP_FEDERATION, NodeStatus, RunType
from flwr.supercore.corestate.corestate_test import StateTest as CoreStateTest
from flwr.supercore.object_store.object_store_factory import ObjectStoreFactory
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes
from flwr.superlink.federation import NoOpFederationManager


class StateTest(CoreStateTest):
    """Test all state implementations."""

    # This is to True in each child class
    __test__ = False

    @abstractmethod
    def state_factory(self) -> LinkState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def create_public_key(self) -> bytes:
        """Create a P-384 public key for node creation."""
        _, public_key = generate_key_pairs()
        return public_key_to_bytes(public_key)

    def test_store_and_get_fab(self) -> None:
        """Test storing and retrieving a FAB."""
        state = self.state_factory()
        content = b"fab-content"
        fab = Fab(hashlib.sha256(content).hexdigest(), content, {"meta": "data"})

        fab_hash = state.store_fab(fab)
        retrieved = state.get_fab(fab_hash)

        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.hash_str, fab_hash)
        self.assertEqual(retrieved.content, fab.content)
        self.assertEqual(retrieved.verifications, fab.verifications)

        # Retrieved FAB should be a defensive copy.
        retrieved.verifications["meta"] = "mutated"
        reloaded = state.get_fab(fab_hash)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        self.assertEqual(reloaded.verifications, {"meta": "data"})

    def test_store_fab_deduplicates_by_hash(self) -> None:
        """Test storing the same FAB content reuses the same hash."""
        state = self.state_factory()
        content = b"fab-content"
        hash_str = hashlib.sha256(content).hexdigest()

        fab_hash = state.store_fab(Fab(hash_str, content, {"meta": "data"}))
        other_hash = state.store_fab(Fab(hash_str, content, {"meta": "next"}))
        retrieved = state.get_fab(fab_hash)

        self.assertEqual(fab_hash, other_hash)
        self.assertIsNotNone(retrieved)
        assert retrieved is not None
        self.assertEqual(retrieved.verifications, {"meta": "next"})

    def test_get_fab_missing_returns_none(self) -> None:
        """Test missing FAB retrieval."""
        state = self.state_factory()
        self.assertIsNone(state.get_fab("missing-fab-hash"))

    def test_store_fab_rejects_hash_mismatch(self) -> None:
        """Test storing a FAB fails when provided hash doesn't match content."""
        state = self.state_factory()
        with self.assertRaisesRegex(ValueError, "FAB hash mismatch"):
            state.store_fab(Fab("not-the-content-hash", b"fab-content", {}))

    def test_create_and_get_run_info(self) -> None:
        """Test if create_run and get_run_info work correctly."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = state.create_run(
            None,
            None,
            "9f86d08",
            {"test_key": "test_value"},
            "health-federation",
            None,
            "i1r9f",
            RunType.SERVER_APP,
        )

        # Execute
        run = state.get_run_info(run_ids=[run_id])[0]

        # Assert
        assert run.run_id == run_id
        assert run.fab_hash == "9f86d08"
        assert run.federation == "health-federation"
        assert run.override_config["test_key"] == "test_value"
        assert run.flwr_aid == "i1r9f"

    def test_get_run_info_without_filters_returns_all_runs(self) -> None:
        """Test get_run_info returns all runs when no filter is provided."""
        # Prepare
        state = self.state_factory()
        run_id1 = create_dummy_run(state, flwr_aid="aid-1", federation="federation-1")
        run_id2 = create_dummy_run(state, flwr_aid="aid-2", federation="federation-2")

        # Execute
        runs = state.get_run_info()

        # Assert
        self.assertSetEqual({run.run_id for run in runs}, {run_id1, run_id2})

    def test_get_run_info_filter_by_run_ids(self) -> None:
        """Test get_run_info filters correctly by run_ids."""
        # Prepare
        state = self.state_factory()
        run_id1 = create_dummy_run(state)
        _ = create_dummy_run(state)
        run_id3 = create_dummy_run(state)

        # Execute
        runs = state.get_run_info(run_ids=[run_id1, run_id3])

        # Assert
        self.assertSetEqual({run.run_id for run in runs}, {run_id1, run_id3})

    def test_get_run_info_filter_logic(self) -> None:
        """Test get_run_info ORs within each filter and ANDs across filters."""
        # Prepare
        state = self.state_factory()

        _ = create_dummy_run(state, flwr_aid="aid-1", federation="federation-a")
        run_id2 = create_dummy_run(state, flwr_aid="aid-1", federation="federation-b")
        run_id3 = create_dummy_run(state, flwr_aid="aid-2", federation="federation-a")
        run_id4 = create_dummy_run(state, flwr_aid="aid-2", federation="federation-b")

        transition_run_status(state, run_id2, 1)  # STARTING
        transition_run_status(state, run_id3, 1)  # STARTING
        transition_run_status(state, run_id4, 2)  # RUNNING

        # Execute
        runs = state.get_run_info(
            statuses=[Status.STARTING, Status.RUNNING],
            flwr_aids=["aid-2"],
            federations=["federation-a", "federation-b"],
        )

        # Assert
        self.assertSetEqual({run.run_id for run in runs}, {run_id3, run_id4})

    def test_get_run_info_filter_by_statuses(self) -> None:
        """Test get_run_info filters correctly by statuses only."""
        # Prepare
        state = self.state_factory()
        pending_run_id = create_dummy_run(state)
        starting_run_id = create_dummy_run(state)
        running_run_id = create_dummy_run(state)
        finished_run_id = create_dummy_run(state)

        transition_run_status(state, starting_run_id, 1)
        transition_run_status(state, running_run_id, 2)
        transition_run_status(state, finished_run_id, 3)

        expected_runs = {
            Status.PENDING: {pending_run_id},
            Status.STARTING: {starting_run_id},
            Status.RUNNING: {running_run_id},
            Status.FINISHED: {finished_run_id},
        }

        # Execute & Assert
        for status, expected_run_ids in expected_runs.items():
            with self.subTest(status=status):
                runs = state.get_run_info(statuses=[status])
                self.assertSetEqual(
                    {run.run_id for run in runs},
                    expected_run_ids,
                )

    def test_get_run_info_filter_by_federations(self) -> None:
        """Test get_run_info filters correctly by federations only."""
        # Prepare
        state = self.state_factory()
        run_id1 = create_dummy_run(state, federation="federation-a")
        _ = create_dummy_run(state, federation="federation-b")
        run_id3 = create_dummy_run(state, federation="federation-a")

        # Execute
        runs = state.get_run_info(federations=["federation-a"])

        # Assert
        self.assertSetEqual({run.run_id for run in runs}, {run_id1, run_id3})

    def test_get_run_info_filter_by_flwr_aids(self) -> None:
        """Test get_run_info filters correctly by flwr_aids only."""
        # Prepare
        state = self.state_factory()
        run_id1 = create_dummy_run(state, flwr_aid="aid-1")
        _ = create_dummy_run(state, flwr_aid="aid-2")
        run_id3 = create_dummy_run(state, flwr_aid="aid-1")

        # Execute
        runs = state.get_run_info(flwr_aids=["aid-1"])

        # Assert
        self.assertSetEqual({run.run_id for run in runs}, {run_id1, run_id3})

    def test_get_run_info_filter_by_nonexistent_run_ids(self) -> None:
        """Test get_run_info returns empty for non-existent run_ids."""
        # Prepare
        state = self.state_factory()
        _ = create_dummy_run(state)

        # Execute
        runs = state.get_run_info(run_ids=[999999])

        # Assert
        self.assertEqual(list(runs), [])

    def test_get_run_info_order_by_pending_at_and_limit(self) -> None:
        """Test get_run_info ordering by pending_at and applying limit."""
        # Prepare
        state = self.state_factory()
        run_id1 = create_dummy_run(state)
        time.sleep(1e-6)
        run_id2 = create_dummy_run(state)
        time.sleep(1e-6)
        run_id3 = create_dummy_run(state)
        run_ids = [run_id1, run_id2, run_id3]

        # Execute
        ascending_runs = state.get_run_info(order_by="pending_at", ascending=True)
        descending_runs = state.get_run_info(order_by="pending_at", ascending=False)
        limited_runs = state.get_run_info(
            order_by="pending_at", ascending=True, limit=2
        )

        # Assert
        self.assertEqual([run.run_id for run in ascending_runs], run_ids)
        self.assertEqual([run.run_id for run in descending_runs], run_ids[::-1])
        self.assertEqual([run.run_id for run in limited_runs], run_ids[:2])

    @parameterized.expand([(1,), (2,), (9999,)])  # type: ignore
    def test_get_run_info_limit_without_order_by(self, limit: int) -> None:
        """Test get_run_info applies limit when no order_by is specified."""
        # Prepare
        state = self.state_factory()
        run_ids = {create_dummy_run(state) for _ in range(3)}
        expected_count = min(limit, len(run_ids))

        # Execute
        runs = state.get_run_info(limit=limit)
        returned_run_ids = {run.run_id for run in runs}

        # Assert
        self.assertEqual(len(runs), expected_count)
        self.assertEqual(len(returned_run_ids), expected_count)
        self.assertTrue(returned_run_ids.issubset(run_ids))

    def test_get_run_info_empty_filters(self) -> None:
        """Test get_run_info returns empty when any filter list is empty."""
        # Prepare
        state = self.state_factory()
        _ = create_dummy_run(state, flwr_aid="aid-1", federation="federation-a")
        _ = create_dummy_run(state, flwr_aid="aid-2", federation="federation-b")

        # Execute & Assert
        runs_statuses_empty = state.get_run_info(statuses=[])
        self.assertEqual(list(runs_statuses_empty), [])

        runs_flwr_aids_empty = state.get_run_info(flwr_aids=[])
        self.assertEqual(list(runs_flwr_aids_empty), [])

        runs_federations_empty = state.get_run_info(federations=[])
        self.assertEqual(list(runs_federations_empty), [])

        runs_run_ids_empty = state.get_run_info(run_ids=[])
        self.assertEqual(list(runs_run_ids_empty), [])

    def test_get_and_update_run_status(self) -> None:
        """Test if get_run_status and update_run_status work correctly."""
        # Prepare
        state = self.state_factory()
        run_id1 = create_dummy_run(state)
        run_id2 = create_dummy_run(state)
        state.update_run_status(run_id2, RunStatus(Status.STARTING, "", ""))
        state.update_run_status(run_id2, RunStatus(Status.RUNNING, "", ""))

        # Execute
        run_status_dict = state.get_run_status({run_id1, run_id2})
        status1 = run_status_dict[run_id1]
        status2 = run_status_dict[run_id2]

        # Assert
        assert status1.status == Status.PENDING
        assert status2.status == Status.RUNNING

    @parameterized.expand(
        [("get_run_info",), ("get_run_status",), ("update_run_status",)]
    )  # type: ignore
    def test_run_failed_due_to_heartbeat(self, test_method: str) -> None:
        """Test methods work correctly when the run has no heartbeat."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        assert state.create_token(run_id) is not None
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))

        # Execute
        # The run should be marked as failed after HEARTBEAT_DEFAULT_INTERVAL
        patched_dt = now() + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL + 1)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt

            if test_method == "get_run_info":
                run = state.get_run_info(run_ids=[run_id])[0]
                status = run.status
            elif test_method == "get_run_status":
                status = state.get_run_status({run_id})[run_id]
            elif test_method == "update_run_status":
                # The updation should fail because the run is already finished
                assert not state.update_run_status(
                    run_id, RunStatus(Status.FINISHED, SubStatus.FAILED, "")
                )
                status = state.get_run_status({run_id})[run_id]
            else:
                raise AssertionError

        # Assert
        assert status.status == Status.FINISHED
        assert status.sub_status == SubStatus.FAILED
        assert status.details == RUN_FAILURE_DETAILS_NO_HEARTBEAT

    @parameterized.expand([(0,), (1,), (2,)])  # type: ignore
    def test_status_transition_valid(
        self, num_transitions_before_finishing: int
    ) -> None:
        """Test valid run status transactions."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)

        # Execute and assert
        status = state.get_run_status({run_id})[run_id]
        assert status.status == Status.PENDING

        if num_transitions_before_finishing > 0:
            assert state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
            status = state.get_run_status({run_id})[run_id]
            assert status.status == Status.STARTING

        if num_transitions_before_finishing > 1:
            assert state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
            status = state.get_run_status({run_id})[run_id]
            assert status.status == Status.RUNNING

        assert state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.FAILED, "mock failure")
        )

        status = state.get_run_status({run_id})[run_id]
        assert status.status == Status.FINISHED

    def test_status_transition_invalid(self) -> None:
        """Test invalid run status transitions."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        run_statuses = [
            RunStatus(Status.PENDING, "", ""),
            RunStatus(Status.STARTING, "", ""),
            RunStatus(Status.PENDING, "", ""),
            RunStatus(Status.FINISHED, SubStatus.COMPLETED, ""),
        ]

        # Execute and assert
        # Cannot transition from RunStatus.PENDING to RunStatus.PENDING,
        # RunStatus.RUNNING, or RunStatus.FINISHED with COMPLETED substatus
        for run_status in [s for s in run_statuses if s.status != Status.STARTING]:
            assert not state.update_run_status(run_id, run_status)
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        # Cannot transition from RunStatus.STARTING to RunStatus.PENDING,
        # RunStatus.STARTING, or RunStatus.FINISHED with COMPLETED substatus
        for run_status in [s for s in run_statuses if s.status != Status.RUNNING]:
            assert not state.update_run_status(run_id, run_status)
        state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        # Cannot transition from RunStatus.RUNNING
        # to RunStatus.PENDING, RunStatus.STARTING, or RunStatus.RUNNING
        for run_status in [s for s in run_statuses if s.status != Status.FINISHED]:
            assert not state.update_run_status(run_id, run_status)
        state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")
        )
        # Cannot transition to any status from RunStatus.FINISHED
        run_statuses += [
            RunStatus(Status.FINISHED, SubStatus.FAILED, ""),
            RunStatus(Status.FINISHED, SubStatus.STOPPED, ""),
        ]
        for run_status in run_statuses:
            assert not state.update_run_status(run_id, run_status)

    @parameterized.expand([(1,), (2,), (3,)])  # type: ignore
    def test_usage_report_hook_called_on_each_successful_transition(
        self, num_transitions: int
    ) -> None:
        """Test report_run_usage hook is called only on the FINISHED transition."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        state.federation_manager.report_run_usage = Mock()  # type: ignore
        # Execute
        transition_run_status(state, run_id, num_transitions)
        # Assert: hook is called only when the run reaches FINISHED (num_transitions==3)
        expected_calls = 1 if num_transitions == 3 else 0
        assert state.federation_manager.report_run_usage.call_count == expected_calls

    def test_usage_report_hook_called_on_tokens_expired(self) -> None:
        """Test report_run_usage hook is called when tokens expire."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        assert state.create_token(run_id) is not None
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        state.federation_manager.report_run_usage = Mock()  # type: ignore
        # Execute: advance time past token expiry and trigger cleanup via verify_token
        patched_dt = now() + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL + 1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            state.verify_token(run_id, "dummy_token")
        # Assert
        state.federation_manager.report_run_usage.assert_called_once()

    def test_usage_report_hook_not_called_when_no_runs_updated(self) -> None:
        """Test report_run_usage is not called when expired tokens match no runs."""
        # Prepare: create a run already in FINISHED state
        state = self.state_factory()
        run_id = create_dummy_run(state)
        assert state.create_token(run_id) is not None
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.COMPLETED, "done")
        )
        state.federation_manager.report_run_usage = Mock()  # type: ignore
        # Execute: advance time past token expiry and trigger cleanup
        patched_dt = now() + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL + 1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            state.verify_token(run_id, "dummy_token")
        # Assert: hook should NOT be called since the run was already finished
        state.federation_manager.report_run_usage.assert_not_called()

    def test_get_message_ins_empty(self) -> None:
        """Validate that a new state has no input Messages."""
        # Prepare
        state = self.state_factory()

        # Assert
        assert state.num_message_ins() == 0

    def test_get_message_res_empty(self) -> None:
        """Validate that a new state has no reply Messages."""
        # Prepare
        state = self.state_factory()

        # Assert
        assert state.num_message_res() == 0

    def test_store_message_ins_one(self) -> None:
        """Test store_message_ins."""
        # Prepare
        state = self.state_factory()
        dt = datetime.now(tz=timezone.utc)
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )

        # Execute
        state.store_message_ins(message=msg)
        message_ins_list = state.get_message_ins(node_id=node_id, limit=10)

        # Assert
        # One returned Message
        assert len(message_ins_list) == 1
        assert message_ins_list[0].metadata.delivered_at != ""

        # Attempt to fetch a second time returns empty Message list
        assert len(state.get_message_ins(node_id=node_id, limit=10)) == 0

        actual_message_ins = message_ins_list[0]

        assert datetime.fromisoformat(actual_message_ins.metadata.delivered_at) > dt
        assert actual_message_ins.metadata.ttl > 0

    def test_store_message_ins_invalid_node_id(self) -> None:
        """Test store_message_ins with invalid node_id."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        node_id2 = create_dummy_node(state)
        state.delete_node("mock_flwr_aid", node_id2)
        node_id3 = create_dummy_node(state, activate=False)
        node_id4 = create_dummy_node(state)
        invalid_node_id = 61016
        assert invalid_node_id not in {node_id, node_id2, node_id3, node_id4}
        run_id = create_dummy_run(state)
        # A message for a node that doesn't exist
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=invalid_node_id,
                run_id=run_id,
            )
        )
        # A message with src_node_id that's not that of the SuperLink
        msg2 = message_from_proto(
            create_ins_message(src_node_id=61016, dst_node_id=node_id, run_id=run_id)
        )
        # A message for a node that is unregistered
        msg3 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id2, run_id=run_id
            )
        )
        # A message for a node of "registered" status
        msg4 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id3, run_id=run_id
            )
        )
        # A message for a node outside the federation
        mock_has_node = Mock(side_effect=lambda nid, _: nid != node_id4)
        state.federation_manager.has_node = mock_has_node  # type: ignore
        msg5 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id4, run_id=run_id
            )
        )

        # Execute and assert
        assert state.store_message_ins(msg) is None
        assert state.store_message_ins(msg2) is None
        assert state.store_message_ins(msg3) is None
        assert state.store_message_ins(msg4) is None
        assert state.store_message_ins(msg5) is None

    def test_store_and_delete_messages(self) -> None:
        """Test delete_message."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)
        msg0 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )
        msg1 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )
        msg2 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )

        # Insert three Messages
        msg_id_0 = state.store_message_ins(message=msg0)
        msg_id_1 = state.store_message_ins(message=msg1)
        msg_id_2 = state.store_message_ins(message=msg2)

        assert msg_id_0
        assert msg_id_1
        assert msg_id_2

        # Get Message to mark them delivered
        msg_ins_list = state.get_message_ins(node_id=node_id, limit=None)

        # Insert one reply Message and retrieve it to mark it as delivered
        msg_res_0 = Message(Error(0), reply_to=msg_ins_list[0])
        # pylint: disable-next=W0212
        msg_res_0.metadata._message_id = str(uuid4())  # type: ignore

        _ = state.store_message_res(message=msg_res_0)
        retrieved_msg_res_0 = state.get_message_res(
            message_ids={msg_res_0.metadata.reply_to_message_id}
        )[0]
        assert retrieved_msg_res_0.error.code == 0

        # Insert one reply Message, but don't retrieve it
        msg_res_1 = Message(RecordDict(), reply_to=msg_ins_list[1])
        # pylint: disable-next=W0212
        msg_res_1.metadata._message_id = str(uuid4())  # type: ignore
        _ = state.store_message_res(message=msg_res_1)

        # Situation now:
        # - State has three Message, all of them delivered
        # - State has two Message replies, one of them delivered, the other not
        assert state.num_message_ins() == 3
        assert state.num_message_res() == 2

        state.delete_messages({msg_id_0})
        assert state.num_message_ins() == 2
        assert state.num_message_res() == 1

        state.delete_messages({msg_id_1})
        assert state.num_message_ins() == 1
        assert state.num_message_res() == 0

        state.delete_messages({msg_id_2})
        assert state.num_message_ins() == 0
        assert state.num_message_res() == 0

    def test_get_message_ids_from_run_id(self) -> None:
        """Test get_message_ids_from_run_id."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id_0 = create_dummy_run(state)
        # Insert Message with the same run_id
        msg0 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id_0,
            )
        )
        msg1 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id_0,
            )
        )
        # Insert a Message with a different run_id
        # then, ensure it does not appear in result
        run_id_1 = create_dummy_run(state)
        msg2 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id_1,
            )
        )

        # Insert three Messages
        msg_id_0 = state.store_message_ins(message=msg0)
        msg_id_1 = state.store_message_ins(message=msg1)
        msg_id_2 = state.store_message_ins(message=msg2)

        assert msg_id_0
        assert msg_id_1
        assert msg_id_2

        expected_message_ids = {msg_id_0, msg_id_1}

        # Execute
        result = state.get_message_ids_from_run_id(run_id_0)
        bad_result = state.get_message_ids_from_run_id(15)

        self.assertEqual(len(bad_result), 0)
        self.assertSetEqual(result, expected_message_ids)

    # Init tests
    def test_init_state(self) -> None:
        """Test that state is initialized correctly."""
        # Execute
        state = self.state_factory()

        # Assert
        assert isinstance(state, LinkState)

    def test_message_ins_store_identity_and_retrieve_identity(self) -> None:
        """Store identity Message and retrieve it."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )
        # Execute
        message_ins_uuid = state.store_message_ins(msg)
        message_ins_list = state.get_message_ins(node_id=node_id, limit=None)

        # Assert
        assert len(message_ins_list) == 1

        retrieved_message_ins = message_ins_list[0]
        assert retrieved_message_ins.metadata.message_id == str(message_ins_uuid)

    def test_message_ins_store_delivered_and_fail_retrieving(self) -> None:
        """Fail retrieving delivered message."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )
        # Execute
        _ = state.store_message_ins(msg)

        # 1st get: set to delivered
        message_ins_list = state.get_message_ins(node_id=node_id, limit=None)

        assert len(message_ins_list) == 1

        # 2nd get: no Message because it was already delivered before
        message_ins_list = state.get_message_ins(node_id=node_id, limit=None)

        # Assert
        assert len(message_ins_list) == 0

    def test_get_message_ins_limit_throws_for_limit_zero(self) -> None:
        """Fail call with limit=0."""
        # Prepare
        state: LinkState = self.state_factory()

        # Execute & Assert
        with self.assertRaises(AssertionError):
            state.get_message_ins(node_id=2, limit=0)

    def test_message_ins_store_invalid_run_id_and_fail(self) -> None:
        """Store Message with invalid run_id and fail."""
        # Prepare
        state: LinkState = self.state_factory()
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=1234,
                run_id=61016,
            )
        )

        # Execute
        message_id = state.store_message_ins(msg)

        # Assert
        assert message_id is None

    def test_node_ids_initial_state(self) -> None:
        """Test retrieving all node_ids and empty initial state."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)

        # Execute
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_create_node_and_get_nodes(self) -> None:
        """Test creating nodes and get activated nodes."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        node_ids = []

        # Execute
        for _ in range(10):
            node_ids.append(create_dummy_node(state))
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        for i in retrieved_node_ids:
            assert i in node_ids

    def test_get_nodes_filtered_by_federation(self) -> None:
        """Test that get_nodes respects federation manager filtering."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state, federation="test-federation")

        # Create 5 nodes
        node_ids = [create_dummy_node(state) for _ in range(5)]

        # Mock filter_nodes to return only a subset (first 2 nodes)
        subset_node_ids = set(node_ids[:2])
        mock_filter = Mock(return_value=subset_node_ids)
        state.federation_manager.filter_nodes = mock_filter  # type: ignore

        # Execute
        retrieved_node_ids = state.get_nodes(run_id)

        # Assert
        mock_filter.assert_called_once_with(set(node_ids), "test-federation")
        assert retrieved_node_ids == subset_node_ids

    def test_create_node_public_key(self) -> None:
        """Test creating a client node with public key."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"

        # Execute
        expected_registered_at = now().timestamp()
        node_id = state.create_node("fake_aid", "fake_name", public_key, 10)
        node = state.get_node_info(node_ids=[node_id])[0]
        actual_registered_at = datetime.fromisoformat(node.registered_at).timestamp()

        # Assert
        assert node.node_id == node_id
        assert node.public_key == public_key
        self.assertAlmostEqual(actual_registered_at, expected_registered_at, 2)

    def test_create_node_public_key_twice(self) -> None:
        """Test creating a client node with same public key twice."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        node_id = state.create_node("fake_aid", "fake_name", public_key, 10)

        # Execute
        with self.assertRaises(ValueError):
            state.create_node("fake_aid2", "fake_name", public_key, 10)
        retrieved_nodes = state.get_node_info()

        # Assert
        assert len(retrieved_nodes) == 1
        assert retrieved_nodes[0].node_id == node_id
        assert retrieved_nodes[0].public_key == public_key

        # Assert node_ids and public_key_to_node_id are synced
        if isinstance(state, InMemoryLinkState):
            assert len(state.nodes) == 1
            assert len(state.node_public_key_to_node_id) == 1

    def test_get_node_info_no_filters(self) -> None:
        """Test get_node_info returns all nodes when no filters are provided."""
        state: LinkState = self.state_factory()

        # Prepare: create several nodes
        node_ids = [create_dummy_node(state, activate=False) for _ in range(5)]

        # Execute
        infos = state.get_node_info()

        # Assert
        returned_ids = [info.node_id for info in infos]
        self.assertSetEqual(set(returned_ids), set(node_ids))

    def test_get_node_info_filter_by_node_ids(self) -> None:
        """Test get_node_info filters correctly by node_ids."""
        state: LinkState = self.state_factory()
        node_ids = [create_dummy_node(state, activate=False) for _ in range(5)]

        # Execute: only query the first two
        infos = state.get_node_info(node_ids=node_ids[:2])

        # Assert
        returned_ids = [info.node_id for info in infos]
        self.assertSetEqual(set(returned_ids), set(node_ids[:2]))

    def test_get_node_info_filter_by_owner_aids(self) -> None:
        """Test get_node_info filters correctly by owner_aids."""
        state: LinkState = self.state_factory()
        node_id1 = create_dummy_node(state, owner_aid="alice", activate=False)
        _ = create_dummy_node(state, owner_aid="bob", activate=False)

        infos = state.get_node_info(owner_aids=["alice"])
        returned_ids = [info.node_id for info in infos]

        self.assertEqual(returned_ids, [node_id1])

    def test_get_node_info_filter_by_status(self) -> None:
        """Test get_node_info filters correctly by statuses."""
        state: LinkState = self.state_factory()
        _ = create_dummy_node(state, activate=False)
        _ = create_dummy_node(state)
        node_deleted = create_dummy_node(state)

        # Transition nodes
        state.delete_node("mock_flwr_aid", node_deleted)

        # Execute
        infos = state.get_node_info(statuses=[NodeStatus.REGISTERED, NodeStatus.ONLINE])
        returned_statuses = {info.status for info in infos}

        # Assert: should only contain CREATED and ONLINE
        self.assertTrue(NodeStatus.REGISTERED in returned_statuses)
        self.assertTrue(NodeStatus.ONLINE in returned_statuses)
        self.assertFalse(NodeStatus.UNREGISTERED in returned_statuses)

    def test_get_node_info_multiple_filters(self) -> None:
        """Test get_node_info applies AND logic across filters."""
        # Prepare
        state: LinkState = self.state_factory()
        node1 = create_dummy_node(state, owner_aid="alice")
        _ = create_dummy_node(state, owner_aid="bob")
        _ = create_dummy_node(state, owner_aid="bob", activate=False)

        # Query: owner_aid=alice AND status=ONLINE
        infos = state.get_node_info(owner_aids=["alice"], statuses=[NodeStatus.ONLINE])
        returned_ids = [info.node_id for info in infos]

        self.assertEqual(returned_ids, [node1])

    def test_get_node_info_empty_list_filters(self) -> None:
        """Test get_node_info with empty list filters returns no results."""
        state: LinkState = self.state_factory()
        create_dummy_node(state)

        infos = state.get_node_info(node_ids=[])
        self.assertEqual(infos, [])

    def test_delete_node(self) -> None:
        """Test deleting a client node."""
        # Prepare
        state: LinkState = self.state_factory()
        node_id = create_dummy_node(state)

        # Execute
        expected_unregistered_at = now().timestamp()
        state.delete_node("mock_flwr_aid", node_id)
        retrieved_nodes = state.get_node_info(node_ids=[node_id])
        assert len(retrieved_nodes) == 1
        node = retrieved_nodes[0]
        actual_unregistered_at = datetime.fromisoformat(
            node.unregistered_at
        ).timestamp()

        # Assert
        assert len(retrieved_nodes) == 1
        assert retrieved_nodes[0].status == NodeStatus.UNREGISTERED
        self.assertAlmostEqual(actual_unregistered_at, expected_unregistered_at, 2)
        self.assertAlmostEqual(node.online_until, expected_unregistered_at, 2)

    def test_delete_node_owner_mismatch(self) -> None:
        """Test deleting a client node with owner mismatch."""
        # Prepare
        state: LinkState = self.state_factory()
        _ = create_dummy_run(state)
        node_id = create_dummy_node(state)

        # Execute
        with self.assertRaises(ValueError):
            state.delete_node("wrong_owner_aid", node_id)

    def test_activate_node(self) -> None:
        """Test node activation transitions."""
        # Prepare
        state: LinkState = self.state_factory()
        heartbeat_interval = 30.0

        # Test successful activation from REGISTERED
        node_id = create_dummy_node(state, activate=False)
        assert state.activate_node(node_id, heartbeat_interval)
        assert state.get_node_info(node_ids=[node_id])[0].status == NodeStatus.ONLINE

        # Test successful activation from OFFLINE
        state.deactivate_node(node_id)
        assert state.activate_node(node_id, heartbeat_interval)
        assert state.get_node_info(node_ids=[node_id])[0].status == NodeStatus.ONLINE

        # Test failed activation when already ONLINE
        assert not state.activate_node(node_id, heartbeat_interval)

        # Test failed activation when UNREGISTERED
        state.delete_node("mock_flwr_aid", node_id)
        assert not state.activate_node(node_id, heartbeat_interval)

    def test_deactivate_node(self) -> None:
        """Test node deactivation transitions."""
        # Prepare
        state: LinkState = self.state_factory()
        node_id = create_dummy_node(state)

        # Test successful deactivation from ONLINE
        assert state.deactivate_node(node_id)
        assert state.get_node_info(node_ids=[node_id])[0].status == NodeStatus.OFFLINE

        # Test failed deactivation when already OFFLINE
        assert not state.deactivate_node(node_id)

        # Test failed deactivation from REGISTERED
        node_id2 = create_dummy_node(state, activate=False)
        assert not state.deactivate_node(node_id2)

        # Test failed deactivation when UNREGISTERED
        state.delete_node("mock_flwr_aid", node_id)
        assert not state.deactivate_node(node_id)

    def test_get_nodes_invalid_run_id(self) -> None:
        """Test retrieving all node_ids with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        create_dummy_run(state)
        invalid_run_id = 61016
        create_dummy_node(state)

        # Execute
        retrieved_node_ids = state.get_nodes(invalid_run_id)

        # Assert
        assert len(retrieved_node_ids) == 0

    def test_get_node_id_by_public_key(self) -> None:
        """Test get_node_id_by_public_key."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        node_id = state.create_node("fake_aid", "fake_name", public_key, 10)

        # Execute
        retrieved_node_id = state.get_node_id_by_public_key(public_key)

        # Assert
        assert retrieved_node_id is not None
        assert retrieved_node_id == node_id

    def test_get_node_id_by_public_key_of_deleted_node(self) -> None:
        """Test get_node_id_by_public_key of a deleted node."""
        # Prepare
        state: LinkState = self.state_factory()
        public_key = b"mock"
        node_id = state.create_node("fake_aid", "fake_name", public_key, 10)

        # Execute
        state.delete_node("fake_aid", node_id)
        retrieved_node_id = state.get_node_id_by_public_key(public_key)

        # Assert
        assert retrieved_node_id is None

    def test_num_message_ins(self) -> None:
        """Test if num_message_ins returns correct number of not delivered Messages."""
        # Prepare
        state: LinkState = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)
        msg0 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )
        msg1 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )

        # Insert Messages
        _ = state.store_message_ins(message=msg0)
        _ = state.store_message_ins(message=msg1)

        # Execute
        num = state.num_message_ins()

        # Assert
        assert num == 2

    def test_num_message_res(self) -> None:
        """Test if num_message_res returns correct number of not delivered Message
        replies."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        node_id = create_dummy_node(state)

        msg0 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )
        msg1 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id,
                run_id=run_id,
            )
        )

        # Insert Messages
        _ = state.store_message_ins(message=msg0)
        _ = state.store_message_ins(message=msg1)

        # Store replies
        msg_rp0 = Message(RecordDict(), reply_to=msg0)
        # pylint: disable-next=W0212
        msg_rp0.metadata._message_id = str(uuid4())  # type: ignore
        state.store_message_res(msg_rp0)
        msg_rp1 = Message(RecordDict(), reply_to=msg1)
        # pylint: disable-next=W0212
        msg_rp1.metadata._message_id = str(uuid4())  # type: ignore
        state.store_message_res(msg_rp1)

        # Execute
        num = state.num_message_res()

        # Assert
        assert num == 2

    def test_acknowledge_node_heartbeat(self) -> None:
        """Test if acknowledge_ping works and get_nodes return online nodes.

        We permit HEARTBEAT_PATIENCE - 1 missed heartbeats before marking
        the node offline. In time units, nodes are considered online within
        `last heartbeat time + HEARTBEAT_PATIENCE x heartbeat_interval (in seconds)`.
        """
        # Prepare
        state: LinkState = self.state_factory()
        node_ids = [create_dummy_node(state, activate=False) for _ in range(10)]
        expected_activated_at = now().timestamp()
        expected_deactivated_at = (now() + timedelta(seconds=60)).timestamp()
        for node_id in node_ids[:7]:
            assert state.acknowledge_node_heartbeat(node_id, heartbeat_interval=30)
        for node_id in node_ids[7:]:
            assert state.acknowledge_node_heartbeat(node_id, heartbeat_interval=90)

        # Execute
        # Test with current_time + 90s
        # node_ids[:7] are online until current_time + 60s (HEARTBEAT_PATIENCE * 30s)
        # node_ids[7:] are online until current_time + 180s (HEARTBEAT_PATIENCE * 90s)
        # As a result, only node_ids[7:] will be returned by get_nodes().
        future_dt = now() + timedelta(seconds=90)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt
            nodes = state.get_node_info(node_ids=node_ids)
            online_node_ids = {
                node.node_id for node in nodes if node.status == NodeStatus.ONLINE
            }

        # Assert
        # Allow up to 1 decimal place difference due to file-based SQLite DB speed.
        # CI runs on cracky old machines, so minor delays are expected.
        self.assertSetEqual(online_node_ids, set(node_ids[7:]))
        for node in nodes:
            actual = datetime.fromisoformat(node.last_activated_at).timestamp()
            self.assertAlmostEqual(actual, expected_activated_at, 1)
            if node.status == NodeStatus.OFFLINE:
                actual = datetime.fromisoformat(node.last_deactivated_at).timestamp()
                self.assertAlmostEqual(actual, expected_deactivated_at, 1)

    def test_acknowledge_node_heartbeat_failed(self) -> None:
        """Test that acknowledge_node_heartbeat returns False when the heartbeat
        fails."""
        # Prepare
        state: LinkState = self.state_factory()

        # Execute
        is_successful = state.acknowledge_node_heartbeat(0, heartbeat_interval=30)

        # Assert
        assert not is_successful

    def test_node_unavailable_error(self) -> None:  # pylint: disable=too-many-locals
        """Test if get_message_res return Message containing node unavailable error."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        node_id_0 = create_dummy_node(state)
        node_id_1 = create_dummy_node(state)
        node_id_2 = create_dummy_node(state)

        # Run acknowledge heartbeat
        state.acknowledge_node_heartbeat(node_id_0, heartbeat_interval=90)
        state.acknowledge_node_heartbeat(node_id_1, heartbeat_interval=30)
        state.acknowledge_node_heartbeat(node_id_2, heartbeat_interval=30)

        # Create and store Messages
        in_message_0 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id_0,
                run_id=run_id,
            )
        )
        in_message_1 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id_1,
                run_id=run_id,
            )
        )
        in_message_2 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID,
                dst_node_id=node_id_2,
                run_id=run_id,
            )
        )
        message_id_0 = state.store_message_ins(in_message_0)
        message_id_1 = state.store_message_ins(in_message_1)
        message_id_2 = state.store_message_ins(in_message_2)
        assert message_id_0 and message_id_1 and message_id_2

        # Get Message to mark them delivered
        state.get_message_ins(node_id=node_id_0, limit=None)
        state.get_message_ins(node_id=node_id_1, limit=None)

        # Delete the 3rd node to simulate unavailability
        state.delete_node("mock_flwr_aid", node_id_2)

        # Create and store reply Messages
        res_message_0 = Message(content=RecordDict(), reply_to=in_message_0)
        # pylint: disable-next=W0212
        res_message_0.metadata._message_id = res_message_0.object_id  # type: ignore
        assert state.store_message_res(res_message_0) is not None

        # Execute
        # Test with current_time + 100s
        # node_id_0 remain online until current_time + 180s (HEARTBEAT_PATIENCE * 90s)
        # node_id_1 remain online until current_time + 60s (HEARTBEAT_PATIENCE * 30s)
        # As a result, a reply message with NODE_UNAVAILABLE
        # error will generate for node_id_1.
        future_dt = now() + timedelta(seconds=100)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt
            res_message_list = state.get_message_res(
                {message_id_0, message_id_1, message_id_2}
            )
            msgs = {msg.metadata.reply_to_message_id: msg for msg in res_message_list}

        # Assert
        assert len(res_message_list) == 3
        reply_1 = msgs[message_id_1]  # Offline due to heartbeat timeout
        assert reply_1.has_error()
        assert reply_1.error.code == ErrorCode.NODE_UNAVAILABLE
        reply_2 = msgs[message_id_2]  # Deleted node
        assert reply_2.has_error()
        assert reply_2.error.code == ErrorCode.NODE_UNAVAILABLE

    def test_store_message_res_message_ins_expired(self) -> None:
        """Test behavior of store_message_res when the Message it replies to is
        expired."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        node_id = create_dummy_node(state)
        # Create and store a message
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        state.store_message_ins(message=msg)

        msg_to_reply_to = state.get_message_ins(node_id=node_id, limit=2)[0]
        reply_msg = Message(RecordDict(), reply_to=msg_to_reply_to)

        # Execute
        # This patch respresents a very slow communication/ClientApp execution
        # that triggers TTL
        future_dt = now() + timedelta(seconds=msg.metadata.ttl)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt
            result = state.store_message_res(reply_msg)

        # Assert
        assert result is None
        assert state.num_message_ins() == 1
        assert state.num_message_res() == 0

    # pylint: disable=W0212
    def test_store_message_res_limit_ttl(self) -> None:
        """Test store_message_res regarding the TTL in reply Message."""
        current_time = now().timestamp()

        test_cases = [
            (
                current_time - 5,
                10,
                current_time - 2,
                6,
                True,
            ),  # Message within allowed TTL
            (
                current_time - 5,
                10,
                current_time - 2,
                15,
                False,
            ),  # Message TTL exceeds max allowed TTL
        ]

        for (
            msg_ins_created_at,
            msg_ins_ttl,
            msg_res_created_at,
            msg_res_ttl,
            expected_store_result,
        ) in test_cases:

            # Prepare
            state: LinkState = self.state_factory()
            run_id = create_dummy_run(state)
            node_id = create_dummy_node(state)

            # Create message, tweak created_at and store
            msg = message_from_proto(
                create_ins_message(
                    src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
                )
            )

            msg.metadata.created_at = msg_ins_created_at
            msg.metadata.ttl = msg_ins_ttl
            state.store_message_ins(message=msg)

            reply_msg = Message(RecordDict(), reply_to=msg)
            reply_msg.metadata._message_id = str(uuid4())  # type: ignore
            reply_msg.metadata.created_at = msg_res_created_at
            reply_msg.metadata.ttl = msg_res_ttl

            # Execute
            res = state.store_message_res(reply_msg)

            # Assert
            if expected_store_result:
                assert res is not None
            else:
                assert res is None

    def test_store_message_res_node_removed_from_federation(self) -> None:
        """Test that store_message_res returns None if destination node is removed from
        federation after message was stored."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        # Store message for node and retrieve
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        state.store_message_ins(message=msg)
        assert state.get_message_ins(node_id=node_id, limit=None)

        # Mock removal of node from federation
        state.federation_manager.filter_nodes = Mock(return_value=set())  # type: ignore

        # Create reply message
        reply_msg = Message(RecordDict(), reply_to=msg)
        reply_msg.metadata.__dict__["_message_id"] = reply_msg.object_id

        # Execute
        result = state.store_message_res(reply_msg)

        # Assert
        # Should return None since node is no longer in federation
        assert result is None
        assert state.num_message_res() == 0
        assert state.num_message_ins() == 0

    def test_get_message_ins_not_return_expired(self) -> None:
        """Test get_message_ins not to return expired Messages."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)
        # Create message, tweak created_at, ttl and store
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        msg.metadata.created_at = now().timestamp() - 5
        msg.metadata.ttl = 5.1

        # Execute
        state.store_message_ins(message=msg)

        # Assert
        future_dt = now() + timedelta(seconds=1.1)  # over TTL limit by 1 second
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt
            message_list = state.get_message_ins(node_id=2, limit=None)
            assert len(message_list) == 0

    def test_get_message_ins_node_removed_from_federation(self) -> None:
        """Test that get_message_ins returns nothing if node is removed from federation
        after message was stored."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        # Store message for node
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        assert state.store_message_ins(message=msg)

        # Mock removal of node from federation
        state.federation_manager.filter_nodes = Mock(return_value=set())  # type: ignore

        # Execute
        message_ins_list = state.get_message_ins(node_id=node_id, limit=None)

        # Assert
        # Should return empty list since node is no longer in federation
        assert len(message_ins_list) == 0
        # Message should still be deleted
        assert state.num_message_ins() == 0

    def test_get_message_res_expired_message_ins(self) -> None:
        """Test get_message_res to return error Message if the inquired message has
        expired."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        # A message that will expire before it gets pulled
        msg1 = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        ins_msg1_id = state.store_message_ins(msg1)
        assert ins_msg1_id
        assert state.num_message_ins() == 1

        future_dt = now() + timedelta(seconds=msg1.metadata.ttl + 0.1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt  # over TTL limit
            res_msg = state.get_message_res({ins_msg1_id})[0]
            assert res_msg.has_error()
            assert res_msg.error.code == ErrorCode.MESSAGE_UNAVAILABLE

    def test_get_message_res_reply_not_ready(self) -> None:
        """Test get_message_res to return nothing since reply Message isn't present."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        ins_msg_id = state.store_message_ins(msg)
        assert ins_msg_id

        reply = state.get_message_res({ins_msg_id})
        assert len(reply) == 0
        # Check message contains error informing reply message hasn't arrived
        assert state.num_message_ins() == 1
        assert state.num_message_res() == 0

    def test_get_message_res_returns_empty_for_missing_message_ins(self) -> None:
        """Test that get_message_res returns an empty result when the corresponding
        Message does not exist."""
        # Prepare
        state = self.state_factory()
        message_ins_id = "5b0a3fc2-edba-4525-a89a-04b83420b7c8"
        # Execute
        message_res_list = state.get_message_res(message_ids={message_ins_id})
        print(message_res_list)

        # Assert
        assert len(message_res_list) == 1
        assert message_res_list[0].has_error()
        assert message_res_list[0].error.code == ErrorCode.MESSAGE_UNAVAILABLE

    def test_get_message_res_node_removed_from_federation(self) -> None:
        """Test that when node is removed from federation after storing message_ins and
        message_res, both are deleted and get_message_res returns error."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        # Store message_ins and retrieve
        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        assert state.store_message_ins(msg)
        state.get_message_ins(node_id=node_id, limit=None)

        # Store message_res
        res_msg = Message(RecordDict(), reply_to=msg)
        res_msg.metadata.__dict__["_message_id"] = res_msg.object_id
        assert state.store_message_res(res_msg)

        # Mock removal of node from federation
        state.federation_manager.filter_nodes = Mock(return_value=set())  # type: ignore

        # Execute
        message_res_list = state.get_message_res(message_ids={msg.object_id})

        # Assert
        # Should return error message since node is no longer in federation
        assert len(message_res_list) == 1
        assert message_res_list[0].has_error()
        assert message_res_list[0].error.code == ErrorCode.MESSAGE_UNAVAILABLE
        # Both message_ins and message_res should be deleted
        assert state.num_message_ins() == 0
        assert state.num_message_res() == 0

    def test_get_message_res_return_successful(self) -> None:
        """Test get_message_res returns correct Message."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        ins_msg_id = state.store_message_ins(msg)
        assert state.num_message_ins() == 1
        assert ins_msg_id
        # Fetch ins message
        ins_msg = state.get_message_ins(node_id=node_id, limit=1)[0]
        # Create reply and insert
        res_msg = Message(RecordDict(), reply_to=ins_msg)
        res_msg.metadata._message_id = str(uuid4())  # type: ignore
        state.store_message_res(res_msg)
        assert state.num_message_res() == 1

        # Fetch reply
        reply_msg = state.get_message_res({ins_msg_id})

        # Assert
        assert reply_msg[0].metadata.dst_node_id == msg.metadata.src_node_id

        # We haven't called deletion of messages
        assert state.num_message_ins() == 1
        assert state.num_message_res() == 1

    def test_store_message_res_fail_if_dst_src_node_id_mismatch(self) -> None:
        """Test store_message_res to fail if there is a mismatch between the dst_node_id
        of orginal Message and the src_node_id of the reply Message."""
        # Prepare
        state = self.state_factory()
        node_id = create_dummy_node(state)
        run_id = create_dummy_run(state)

        msg = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
        )
        state.store_message_ins(msg)
        assert state.num_message_ins() == 1

        # Fetch ins message
        ins_msg = state.get_message_ins(node_id=node_id, limit=1)[0]
        assert state.num_message_ins() == 1

        # Create reply, modify src_node_id and insert
        res_msg = Message(RecordDict(), reply_to=ins_msg)
        # pylint: disable=W0212
        res_msg.metadata._src_node_id = node_id + 1  # type: ignore
        msg_res_id = state.store_message_res(res_msg)

        # Assert
        assert msg_res_id is None
        assert state.num_message_ins() == 1
        assert state.num_message_res() == 0

    def test_get_set_serverapp_context(self) -> None:
        """Test get and set serverapp context."""
        # Prepare
        state: LinkState = self.state_factory()
        context = Context(
            run_id=1,
            node_id=SUPERLINK_NODE_ID,
            node_config={"mock": "mock"},
            state=RecordDict(),
            run_config={"test": "test"},
        )
        run_id = create_dummy_run(state)

        # Execute
        init = state.get_serverapp_context(run_id)
        state.set_serverapp_context(run_id, context)
        retrieved_context = state.get_serverapp_context(run_id)

        # Assert
        assert init is None
        assert retrieved_context == context

    def test_set_context_invalid_run_id(self) -> None:
        """Test set_serverapp_context with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        context = Context(
            run_id=1,
            node_id=1234,
            node_config={"mock": "mock"},
            state=RecordDict(),
            run_config={"test": "test"},
        )

        # Execute and assert
        with self.assertRaises(ValueError):
            state.set_serverapp_context(61016, context)  # Invalid run_id

    def test_add_serverapp_log_invalid_run_id(self) -> None:
        """Test adding serverapp log with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        invalid_run_id = 99999
        log_entry = "Invalid log entry"

        # Execute and assert
        with self.assertRaises(ValueError):
            state.add_serverapp_log(invalid_run_id, log_entry)

    def test_get_serverapp_log_invalid_run_id(self) -> None:
        """Test retrieving serverapp log with invalid run_id."""
        # Prepare
        state: LinkState = self.state_factory()
        invalid_run_id = 99999

        # Execute and assert
        with self.assertRaises(ValueError):
            state.get_serverapp_log(invalid_run_id, after_timestamp=None)

    def test_add_and_get_serverapp_log(self) -> None:
        """Test adding and retrieving serverapp logs."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        log_entry_1 = "Log entry 1"
        log_entry_2 = "Log entry 2"
        timestamp = now().timestamp()

        # Execute
        state.add_serverapp_log(run_id, log_entry_1)
        state.add_serverapp_log(run_id, log_entry_2)
        retrieved_logs, latest = state.get_serverapp_log(
            run_id, after_timestamp=timestamp
        )

        # Assert
        assert latest > timestamp
        assert log_entry_1 + log_entry_2 == retrieved_logs

    def test_get_serverapp_log_after_timestamp(self) -> None:
        """Test retrieving serverapp logs after a specific timestamp."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        log_entry_1 = "Log entry 1"
        log_entry_2 = "Log entry 2"
        state.add_serverapp_log(run_id, log_entry_1)
        # Add trivial delays to avoid random failure due to same timestamp
        time.sleep(1e-6)
        timestamp = now().timestamp()
        time.sleep(1e-6)
        state.add_serverapp_log(run_id, log_entry_2)

        # Execute
        retrieved_logs, latest = state.get_serverapp_log(
            run_id, after_timestamp=timestamp
        )

        # Assert
        assert latest > timestamp
        assert log_entry_1 not in retrieved_logs
        assert log_entry_2 == retrieved_logs

    def test_get_serverapp_log_after_timestamp_no_logs(self) -> None:
        """Test retrieving serverapp logs after a specific timestamp but no logs are
        found."""
        # Prepare
        state: LinkState = self.state_factory()
        run_id = create_dummy_run(state)
        log_entry = "Log entry"
        state.add_serverapp_log(run_id, log_entry)
        timestamp = now().timestamp() + 0.001  # Ensure timestamp is after the log entry

        # Execute
        retrieved_logs, latest = state.get_serverapp_log(
            run_id, after_timestamp=timestamp
        )

        # Assert
        assert latest == 0
        assert retrieved_logs == ""

    def test_create_run_with_and_without_federation_config(self) -> None:
        """Test that run federation config is stored on the run."""
        # Prepare
        state = self.state_factory()
        federation_config = SimulationConfig(num_supernodes=3, backend="ray")
        run_id = create_dummy_run(
            state,
            federation_config=federation_config,
            run_type=RunType.SIMULATION,
        )
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        second_run_id = create_dummy_run(state)

        # Execute
        run_info = state.get_run_info(run_ids=[run_id])[0]
        second_run_info = state.get_run_info(run_ids=[second_run_id])[0]

        # Assert
        assert run_info.run_type == RunType.SIMULATION
        assert state.get_federation_config(run_id) == federation_config
        assert second_run_info.run_type == RunType.SERVER_APP
        assert state.get_federation_config(second_run_id) is None

    def test_set_linkstate_of_federation_manager(self) -> None:
        """Test that setting the LinkState of the FederationManager works."""
        state: LinkState = self.state_factory()
        assert state.federation_manager.linkstate is state

    def test_store_traffic_basic(self) -> None:
        """Test basic traffic storage functionality."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        transition_run_status(state, run_id, 2)  # Transition to RUNNING

        # Execute
        state.store_traffic(run_id, bytes_sent=1000, bytes_recv=2000)
        run = state.get_run_info(run_ids=[run_id])[0]

        # Assert
        assert run.bytes_sent == 1000
        assert run.bytes_recv == 2000

    def test_store_traffic_accumulation(self) -> None:
        """Test that traffic accumulates correctly over multiple calls."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        transition_run_status(state, run_id, 2)  # Transition to RUNNING

        # Execute
        state.store_traffic(run_id, bytes_sent=1000, bytes_recv=500)
        state.store_traffic(run_id, bytes_sent=2000, bytes_recv=1500)
        state.store_traffic(run_id, bytes_sent=500, bytes_recv=1000)
        run = state.get_run_info(run_ids=[run_id])[0]

        # Assert
        assert run.bytes_sent == 3500
        assert run.bytes_recv == 3000

    @parameterized.expand(
        [
            (-1000, 2000),  # negative bytes_sent
            (1000, -2000),  # negative bytes_recv
            (-500, -1000),  # both negative
        ]
    )  # type: ignore
    def test_store_traffic_negative_values(
        self, bytes_sent: int, bytes_recv: int
    ) -> None:
        """Test that negative traffic values raise ValueError."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)

        # Set initial traffic
        state.store_traffic(run_id, bytes_sent=1000, bytes_recv=2000)

        # Execute & Assert
        with self.assertRaises(ValueError):
            state.store_traffic(run_id, bytes_sent=bytes_sent, bytes_recv=bytes_recv)

        # Verify traffic was not updated
        run = state.get_run_info(run_ids=[run_id])[0]
        assert run.bytes_sent == 1000
        assert run.bytes_recv == 2000

    def test_store_traffic_invalid_run_id(self) -> None:
        """Test that invalid run_id raises ValueError."""
        # Prepare
        state = self.state_factory()
        invalid_run_id = 98889  # Run ID that doesn't exist

        # Execute & Assert
        with self.assertRaises(ValueError):
            state.store_traffic(invalid_run_id, bytes_sent=1000, bytes_recv=2000)

    def test_store_traffic_both_zero(self) -> None:
        """Test that both bytes_sent and bytes_recv being zero raises ValueError."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)

        # Execute & Assert
        with self.assertRaises(ValueError) as context:
            state.store_traffic(run_id, bytes_sent=0, bytes_recv=0)

        assert "cannot be zero" in str(context.exception)
        run = state.get_run_info(run_ids=[run_id])[0]
        assert run.bytes_sent == 0
        assert run.bytes_recv == 0

    def test_add_clientapp_runtime_invalid_run_id(self) -> None:
        """Test that invalid run_id raises ValueError for add_clientapp_runtime."""
        # Prepare
        state = self.state_factory()
        invalid_run_id = 57775  # Run ID that doesn't exist

        # Execute & Assert
        with self.assertRaises(ValueError) as context:
            state.add_clientapp_runtime(invalid_run_id, runtime=10.5)

        assert f"Run {invalid_run_id} not found" in str(context.exception)


def create_ins_message(
    src_node_id: int,
    dst_node_id: int,
    run_id: int,
) -> ProtoMessage:
    """Create an instruction Message proto for testing."""
    proto = ProtoMessage(
        metadata=ProtoMetadata(
            run_id=run_id,
            message_id="",
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            group_id="",
            ttl=DEFAULT_TTL,
            message_type="query",
            created_at=now().timestamp(),
        ),
        content=ProtoRecordDict(),
    )
    proto.metadata.message_id = message_from_proto(proto).object_id
    return proto


def create_res_message(
    src_node_id: int,
    dst_node_id: int,
    run_id: int,
    error: Error | None = None,
) -> ProtoMessage:
    """Create a (reply) Message proto for testing."""
    in_msg_proto = create_ins_message(
        src_node_id=dst_node_id, dst_node_id=src_node_id, run_id=run_id
    )
    in_msg = message_from_proto(in_msg_proto)

    if error:
        out_msg = Message(error, reply_to=in_msg)
    else:
        out_msg = Message(RecordDict(), reply_to=in_msg)
    out_msg.metadata.__dict__["_message_id"] = out_msg.object_id
    return message_to_proto(out_msg)


def create_ins_message_obj(src_node_id: int, dst_node_id: int, run_id: int) -> Message:
    """Create an instruction Message object for testing."""
    proto = create_ins_message(src_node_id, dst_node_id, run_id)
    return message_from_proto(proto)


def create_res_message_obj(
    src_node_id: int,
    dst_node_id: int,
    run_id: int,
    error: Error | None = None,
) -> Message:
    """Create a (reply) Message object for testing."""
    proto = create_res_message(src_node_id, dst_node_id, run_id, error)
    return message_from_proto(proto)


def transition_run_status(state: LinkState, run_id: int, num_transitions: int) -> None:
    """Transition run status from PENDING."""
    if num_transitions > 0:
        state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
    if num_transitions > 1:
        state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
    if num_transitions > 2:
        state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.COMPLETED, "")
        )


def create_dummy_node(
    state: LinkState,
    heartbeat_interval: int = 1000,
    owner_aid: str = "mock_flwr_aid",
    owner_name: str = "mock_flwr_name",
    activate: bool = True,
) -> int:
    """Create a dummy node."""
    node_id = state.create_node(
        owner_aid, owner_name, secrets.token_bytes(32), heartbeat_interval
    )
    if activate:
        state.acknowledge_node_heartbeat(node_id, heartbeat_interval)
    return node_id


def create_dummy_run(  # pylint: disable=too-many-positional-arguments
    state: LinkState,
    fab_id: str | None = "mock_fab_id",
    fab_version: str | None = "mock_fab_version",
    fab_hash: str | None = "mock_fab_hash",
    override_config: UserConfig | None = None,
    federation: str = NOOP_FEDERATION,
    federation_config: SimulationConfig | None = None,
    flwr_aid: str | None = "mock_flwr_aid",
    run_type: str = RunType.SERVER_APP,
) -> int:
    """Create a dummy run."""
    return state.create_run(
        fab_id=fab_id,
        fab_version=fab_version,
        fab_hash=fab_hash,
        override_config=override_config or {},
        federation=federation,
        federation_config=federation_config,
        flwr_aid=flwr_aid,
        run_type=run_type,
    )


def _claim_running_in_separate_process(
    database_path: str,
    run_id: int,
    start_event: Any,
    result_queue: Any,
    timeout: float,
) -> None:
    """Try to claim STARTING -> RUNNING in a dedicated process."""
    state = SqlLinkState(
        database_path=database_path,
        federation_manager=NoOpFederationManager(),
        object_store=ObjectStoreFactory().store(),
    )
    state.initialize()
    if not start_event.wait(timeout=timeout):
        result_queue.put((False, "start-event-timeout"))
        return
    try:
        result = state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        result_queue.put((result, None))
    except Exception as ex:  # pylint: disable=broad-exception-caught
        result_queue.put((False, repr(ex)))


class InMemoryStateTest(StateTest):
    """Test InMemoryState implementation."""

    __test__ = True

    def state_factory(self) -> InMemoryLinkState:
        """Return InMemoryState."""
        return InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())

    def test_owner_aid_index(self) -> None:
        """Test that the owner_aid index works correctly."""
        # Prepare
        state = self.state_factory()
        node_id1 = state.create_node("aid1", "owner1", b"key1", 10)
        node_id2 = state.create_node("aid1", "owner2", b"key2", 10)
        node_id3 = state.create_node("aid2", "owner3", b"key3", 10)

        # Assert
        self.assertSetEqual(state.owner_to_node_ids["aid1"], {node_id1, node_id2})
        self.assertSetEqual(state.owner_to_node_ids["aid2"], {node_id3})


class SqlInMemoryStateTest(StateTest, unittest.TestCase):
    """Test SqlLinkState implementation with in-memory database."""

    __test__ = True

    def state_factory(self) -> SqlLinkState:
        """Return SqlLinkState with in-memory database."""
        state = SqlLinkState(
            database_path=":memory:",
            federation_manager=NoOpFederationManager(),
            object_store=ObjectStoreFactory().store(),
        )
        state.initialize()
        return state

    def test_token_expiry_does_not_overwrite_finished_completed_run(self) -> None:
        """Ensure token cleanup doesn't mutate terminal COMPLETED status."""
        # Prepare
        state = self.state_factory()
        run_id = create_dummy_run(state)
        assert state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))
        assert state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
        assert state.update_run_status(
            run_id, RunStatus(Status.FINISHED, SubStatus.COMPLETED, "done")
        )
        assert state.create_token(run_id) is not None

        # Execute: force token expiry and trigger cleanup
        patched_dt = now() + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL + 1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = patched_dt
            status = state.get_run_status({run_id})[run_id]

        # Assert
        assert status.status == Status.FINISHED
        assert status.sub_status == SubStatus.COMPLETED
        assert status.details == "done"


class SqlFileBasedTest(StateTest, unittest.TestCase):
    """Test SqlLinkState implementation with file-based database."""

    __test__ = True
    _CONCURRENT_TEST_TIMEOUT = 10.0
    states: list[SqlLinkState] = []

    def state_factory(self) -> SqlLinkState:
        """Return SqlLinkState with file-based database."""
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        state = SqlLinkState(
            database_path=os.path.join(tmp_dir, "state.db"),
            federation_manager=NoOpFederationManager(),
            object_store=ObjectStoreFactory().store(),
        )
        state.initialize()
        return state

    def _create_shared_sql_states(
        self, database_path: str, num_replicas: int = 2
    ) -> list[SqlLinkState]:
        """Create two SqlLinkState replicas sharing the same SQLite file."""
        self.states = []
        for _ in range(num_replicas):
            state = SqlLinkState(
                database_path=database_path,
                federation_manager=NoOpFederationManager(),
                object_store=ObjectStoreFactory().store(),
            )
            state.initialize()
            self.states.append(state)
        return self.states

    def _query_states_in_parallel(
        self,
        fn: Callable[[SqlLinkState], Any],
        timeout: float = _CONCURRENT_TEST_TIMEOUT,
    ) -> list[Any]:
        """Query SqlLinkState concurrently and return their results."""
        n_threads = len(self.states)
        barrier = threading.Barrier(n_threads + 1)
        results: list[Any] = [None] * n_threads
        exceptions: list[Exception] = []

        def _run(idx: int) -> None:
            try:
                barrier.wait(timeout=timeout)
                results[idx] = fn(self.states[idx])
            except Exception as ex:  # pylint: disable=broad-exception-caught
                exceptions.append(ex)

        threads = [
            threading.Thread(target=_run, args=(idx,)) for idx in range(n_threads)
        ]
        for thread in threads:
            thread.start()
        try:
            barrier.wait(timeout=timeout)
        except threading.BrokenBarrierError as ex:
            exceptions.append(ex)
        for thread in threads:
            thread.join(timeout=timeout)
        alive_threads = [thread for thread in threads if thread.is_alive()]
        if alive_threads:
            alive_count = len(alive_threads)
            raise AssertionError(
                f"Concurrent test timed out; {alive_count} thread(s) still alive "
                f"after {timeout} seconds."
            )

        if exceptions:
            raise exceptions[0]
        return results

    # pylint: disable-next=too-many-locals
    def test_get_message_ins_claim_is_unique_across_replicas(self) -> None:
        """Ensure concurrent replicas cannot both claim the same instruction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare
            db_path = os.path.join(tmpdir, "shared.db")
            state = self._create_shared_sql_states(db_path)[0]
            node_id = create_dummy_node(state)
            run_id = create_dummy_run(state)
            msg = create_ins_message_obj(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
            assert state.store_message_ins(message=msg)

            # Execute
            results = self._query_states_in_parallel(
                lambda state: state.get_message_ins(node_id=node_id, limit=1)
            )
            claimed = [msgs for msgs in results if msgs]

            # Assert
            assert len(claimed) == 1
            assert len(claimed[0]) == 1

    def test_get_message_res_claim_is_unique_across_replicas(self) -> None:
        """Ensure concurrent replicas cannot both claim the same reply Message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare
            db_path = os.path.join(tmpdir, "shared.db")
            state = self._create_shared_sql_states(db_path)[0]

            node_id = create_dummy_node(state)
            assert state.store_message_ins(
                create_ins_message_obj(
                    src_node_id=SUPERLINK_NODE_ID,
                    dst_node_id=node_id,
                    run_id=create_dummy_run(state),
                )
            )
            pulled_ins = state.get_message_ins(node_id=node_id, limit=1)[0]

            msg_res = Message(RecordDict(), reply_to=pulled_ins)
            msg_res.metadata.__dict__["_message_id"] = str(uuid4())
            assert state.store_message_res(msg_res)

            # Execute
            msg_id = pulled_ins.metadata.message_id
            results = self._query_states_in_parallel(
                lambda state: state.get_message_res({msg_id}),
            )

            # Assert
            assert sum(len(res) for res in results if res is not None) == 1

    # pylint: disable-next=too-many-locals
    def test_get_message_ins_distributes_available_work_under_contention(self) -> None:
        """Ensure two replicas can each claim work when two Messages are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare
            db_path = os.path.join(tmpdir, "shared.db")
            state = self._create_shared_sql_states(db_path)[0]

            node_id = create_dummy_node(state)
            run_id = create_dummy_run(state)
            msg_0 = create_ins_message_obj(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
            msg_1 = create_ins_message_obj(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
            )
            assert state.store_message_ins(message=msg_0)
            assert state.store_message_ins(message=msg_1)

            # Execute
            results = self._query_states_in_parallel(
                lambda state: state.get_message_ins(node_id=node_id, limit=1)
            )
            claimed_messages = [msgs for msgs in results if msgs]

            # Assert
            assert len(claimed_messages) == 2
            assert all(len(msgs) == 1 for msgs in claimed_messages)
            assert (
                claimed_messages[0][0].metadata.message_id
                != claimed_messages[1][0].metadata.message_id
            )

    # pylint: disable-next=too-many-locals
    def test_update_run_status_running_claim_is_atomic_across_replicas(self) -> None:
        """Ensure only one replica can claim STARTING -> RUNNING transition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Prepare
            db_path = os.path.join(tmpdir, "shared.db")
            state = self._create_shared_sql_states(db_path)[0]
            run_id = create_dummy_run(state)
            assert state.update_run_status(run_id, RunStatus(Status.STARTING, "", ""))

            ctx = multiprocessing.get_context("spawn")
            start_event = ctx.Event()
            result_queue = ctx.Queue()
            timeout = self._CONCURRENT_TEST_TIMEOUT

            # Execute
            processes = [
                ctx.Process(
                    target=_claim_running_in_separate_process,
                    args=(db_path, run_id, start_event, result_queue, timeout),
                ),
                ctx.Process(
                    target=_claim_running_in_separate_process,
                    args=(db_path, run_id, start_event, result_queue, timeout),
                ),
            ]
            for proc in processes:
                proc.start()
            # Release both processes to claim at (roughly) the same time.
            start_event.set()
            for proc in processes:
                proc.join(timeout=timeout)

            alive_processes = [proc for proc in processes if proc.is_alive()]
            if alive_processes:
                for proc in alive_processes:
                    proc.terminate()
                for proc in alive_processes:
                    proc.join(timeout=1.0)
                self.fail(
                    f"Concurrent run-claim test timed out; {len(alive_processes)} "
                    f"process(es) still alive after {timeout} seconds."
                )
            for proc in processes:
                assert proc.exitcode == 0

            results: list[bool] = []
            errors: list[str] = []
            for _ in processes:
                result, error = result_queue.get(timeout=timeout)
                results.append(result)
                if error is not None:
                    errors.append(error)
            if errors:
                self.fail(f"Concurrent run-claim process failed: {errors[0]}")

            # Assert
            assert results.count(True) == 1
            assert results.count(False) == 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
