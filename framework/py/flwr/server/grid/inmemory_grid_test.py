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
"""Tests for in-memory grid."""


import secrets
import time
import unittest
from collections.abc import Iterable
from unittest.mock import MagicMock, patch
from uuid import uuid4

from flwr.common import Message, RecordDict, now
from flwr.common.constant import (
    HEARTBEAT_INTERVAL_INF,
    NODE_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    Status,
)
from flwr.common.serde import message_from_proto
from flwr.common.typing import Run, RunStatus
from flwr.server.superlink.linkstate import (
    InMemoryLinkState,
    LinkStateFactory,
    SqlLinkState,
)
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.server.superlink.linkstate.utils import generate_rand_int_from_bytes
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, RunType
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from .inmemory_grid import InMemoryGrid


def push_messages(grid: InMemoryGrid, num_nodes: int) -> tuple[Iterable[str], int]:
    """Help push messages to state."""
    for _ in range(num_nodes):
        node_id = grid.state.create_node(
            "mock_owner",
            "mock_account",
            secrets.token_bytes(32),
            heartbeat_interval=0,  # This field has no effect
        )
        grid.state.acknowledge_node_heartbeat(node_id, HEARTBEAT_INTERVAL_INF)
    num_messages = 3
    msgs = [Message(RecordDict(), node_id, "query") for _ in range(num_messages)]

    # Execute: push messages
    return grid.push_messages(msgs), node_id


def get_replies(grid: InMemoryGrid, msg_ids: Iterable[str], node_id: int) -> list[str]:
    """Help create message replies and pull them from state."""
    messages = grid.state.get_message_ins(node_id, limit=len(list(msg_ids)))
    for msg in messages:
        reply_msg = Message(RecordDict(), reply_to=msg)
        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = str(uuid4())  # type: ignore
        grid.state.store_message_res(message=reply_msg)

    # Execute: Pull messages
    pulled_msgs = grid.pull_messages(msg_ids)
    return [msg.metadata.reply_to_message_id for msg in pulled_msgs]


class TestInMemoryGrid(unittest.TestCase):
    """Tests for `InMemoryGrid` class."""

    def setUp(self) -> None:
        """Initialize State and Grid instance before each test.

        Grid uses the default StateFactory (i.e. SQLite)
        """
        # Create grid
        self.num_nodes = 42
        self.state = MagicMock()
        self.state.get_nodes.return_value = [
            generate_rand_int_from_bytes(NODE_ID_NUM_BYTES)
            for _ in range(self.num_nodes)
        ]
        self.mock_run = Run(
            run_id=61016,
            fab_id="mock/mock",
            fab_version="v1.0.0",
            fab_hash="9f86d08",
            override_config={"test_key": "test_value"},
            pending_at=now().isoformat(),
            starting_at="",
            running_at="",
            finished_at="",
            status=RunStatus(status=Status.PENDING, sub_status="", details=""),
            flwr_aid="user123",
            federation="mock-fed",
            primary_task_id=None,
            bytes_sent=0,
            bytes_recv=0,
            clientapp_runtime=0.0,
        )
        self.state.get_run_info.return_value = [self.mock_run]
        state_factory = MagicMock(state=lambda: self.state)
        self.grid = InMemoryGrid(state_factory=state_factory)
        self.grid.set_run(self.mock_run)
        self.grid.state = self.state

    def test_get_run(self) -> None:
        """Test the InMemoryGrid starting with a `Run` object."""
        # Assert
        self.assertEqual(self.grid.run.run_id, 61016)
        self.assertEqual(self.grid.run.fab_id, "mock/mock")
        self.assertEqual(self.grid.run.fab_version, "v1.0.0")
        self.assertEqual(self.grid.run.fab_hash, "9f86d08")
        self.assertEqual(self.grid.run.override_config["test_key"], "test_value")

    def test_set_run_rejects_non_run_type(self) -> None:
        """Test `set_run` rejects invalid input types."""
        with self.assertRaises(TypeError):
            self.grid.set_run(61016)  # type: ignore[arg-type]

    def test_set_run_rejects_missing_run(self) -> None:
        """Test `set_run` rejects runs that are not registered in state."""
        self.state.get_run_info.return_value = []
        with self.assertRaises(RuntimeError):
            self.grid.set_run(self.mock_run)

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Execute
        node_ids = list(self.grid.get_node_ids())

        # Assert
        self.assertEqual(len(node_ids), self.num_nodes)

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        num_messages = 2
        msgs = [Message(RecordDict(), 1, "query") for _ in range(num_messages)]

        msg_ids = [uuid4() for _ in range(num_messages)]
        self.state.store_message_ins.side_effect = msg_ids

        # Execute
        msg_res_ids = list(self.grid.push_messages(msgs))

        # Assert
        self.assertEqual(len(msg_res_ids), 2)
        self.assertEqual(msg_res_ids, [str(ids) for ids in msg_ids])

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        message_res_list = create_message_replies_for_specific_ids(msg_ids)
        self.state.get_message_res.return_value = message_res_list

        # Execute
        pulled_msgs = list(self.grid.pull_messages(msg_ids))
        reply_tos = [msg.metadata.reply_to_message_id for msg in pulled_msgs]

        # Assert
        self.assertEqual(len(pulled_msgs), 2)
        self.assertEqual(reply_tos, msg_ids)
        # Ensure messages are deleted
        self.state.delete_messages.assert_called_once_with(message_ins_ids=set(msg_ids))

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare
        msgs = [Message(RecordDict(), 0, "query")]
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        message_res_list = create_message_replies_for_specific_ids(msg_ids)
        self.state.get_message_res.return_value = message_res_list
        self.state.store_message_ins.side_effect = msg_ids

        # Execute
        ret_msgs = list(self.grid.send_and_receive(msgs))
        reply_tos = [msg.metadata.reply_to_message_id for msg in ret_msgs]
        # Assert
        self.assertEqual(len(ret_msgs), 2)
        self.assertEqual(reply_tos, msg_ids)

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        msgs = [Message(RecordDict(), 0, "query")]
        # Prepare
        msg_ids = [str(uuid4()) for _ in range(2)]
        message_res_list = create_message_replies_for_specific_ids(msg_ids)
        self.state.get_message_res.return_value = message_res_list
        self.state.store_message_ins.side_effect = msg_ids

        # Execute
        with patch("time.sleep", side_effect=lambda t: time.sleep(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.grid.send_and_receive(msgs, timeout=-1))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

    def test_message_store_consistency_after_push_pull_sqlitestate(self) -> None:
        """Test messages are deleted in sqlite state once messages are pulled."""
        # Prepare
        state = LinkStateFactory(
            "", NoOpFederationManager(), ObjectStoreFactory()
        ).state()
        run_id = state.create_run(
            "", "", "", {}, NOOP_FEDERATION, None, "", RunType.SERVER_APP
        )
        self.grid = InMemoryGrid(MagicMock(state=lambda: state))
        runs = state.get_run_info(run_ids=[run_id])
        self.assertEqual(len(runs), 1)
        self.grid.set_run(runs[0])
        msg_ids, node_id = push_messages(self.grid, self.num_nodes)
        assert isinstance(state, SqlLinkState)

        # Check recorded
        num_msg_ins = len(state.query("SELECT * FROM message_ins;"))
        self.assertEqual(num_msg_ins, len(list(msg_ids)))

        # Prepare: create replies
        reply_tos = get_replies(self.grid, msg_ids, node_id)

        # Query number of Messages and reply Messages in State
        num_msg_res = len(state.query("SELECT * FROM message_res;"))
        num_msg_ins = len(state.query("SELECT * FROM message_ins;"))

        # Assert
        self.assertEqual(reply_tos, msg_ids)
        self.assertEqual(num_msg_res, 0)
        self.assertEqual(num_msg_ins, 0)

    def test_message_store_consistency_after_push_pull_inmemory_state(self) -> None:
        """Test messages are deleted in in-memory state once messages are pulled."""
        # Prepare
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
        )
        state = state_factory.state()
        run_id = state.create_run(
            "", "", "", {}, NOOP_FEDERATION, None, "", RunType.SERVER_APP
        )
        self.grid = InMemoryGrid(state_factory)
        runs = state.get_run_info(run_ids=[run_id])
        self.assertEqual(len(runs), 1)
        self.grid.set_run(runs[0])
        msg_ids, node_id = push_messages(self.grid, self.num_nodes)
        assert isinstance(state, InMemoryLinkState)

        # Check recorded
        self.assertEqual(len(state.message_ins_store), len(list(msg_ids)))

        # Prepare: create replies
        reply_tos = get_replies(self.grid, msg_ids, node_id)

        # Assert
        self.assertEqual(set(reply_tos), set(msg_ids))
        self.assertEqual(len(state.message_res_store), 0)
        self.assertEqual(len(state.message_ins_store), 0)


def create_message_replies_for_specific_ids(message_ids: list[str]) -> list[Message]:
    """Create reply Messages for a set of message IDs."""
    message_replies = []

    for msg_id in message_ids:

        message = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=123, run_id=456
            )
        )
        # pylint: disable=W0212
        message.metadata._message_id = msg_id  # type: ignore

        # Append reply
        message_replies.append(Message(RecordDict(), reply_to=message))

    return message_replies
