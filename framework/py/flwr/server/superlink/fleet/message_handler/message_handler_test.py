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
"""Fleet API message handler tests."""


from unittest.mock import MagicMock

from flwr.app import Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common.serde import message_to_proto
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    FleetPushTaskEventsRequest,
    PullMessagesRequest,
    PushMessagesRequest,
)
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.date import now
from flwr.supercore.run import RunStatus
from flwr.supercore.utils import strict_json_loads

from .message_handler import pull_messages, push_messages, push_task_events


def test_pull_messages() -> None:
    """Test pull_messages."""
    # Prepare
    request = PullMessagesRequest(node=Node(node_id=1234))
    state = MagicMock()
    store = MagicMock()

    # Execute
    pull_messages(request=request, state=state, store=store)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_called_once()
    state.store_message_res.assert_not_called()
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_not_called()


def test_pull_messages_records_traffic_when_messages_found() -> None:
    """Test pull_messages records traffic when messages are successfully retrieved."""
    # Prepare
    msg = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=234,
            message_id="msg-234",
            group_id="",
            src_node_id=0,
            dst_node_id=1234,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=123,
            message_type="query",
        ),
    )
    request = PullMessagesRequest(node=Node(node_id=2345))
    state = MagicMock()
    state.get_message_ins.return_value = [msg]
    store = MagicMock()
    store.get_object_tree.return_value = {}

    # Execute
    pull_messages(request=request, state=state, store=store)

    # Assert
    state.get_message_ins.assert_called_once()
    store.get_object_tree.assert_called_once_with("msg-234")
    state.store_traffic.assert_called_once()
    # Verify store_traffic was called with run_id=123, bytes_sent > 0, bytes_recv=0
    call_args = state.store_traffic.call_args
    assert call_args[0][0] == 234  # run_id
    assert call_args[1]["bytes_sent"] > 0
    assert call_args[1]["bytes_recv"] > 0


def test_push_messages() -> None:
    """Test push_messages."""
    # Prepare
    msg = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=123,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=123,
            message_type="query",
        ),
    )

    object_tree = ObjectTree(object_id="object-id")
    request = PushMessagesRequest(
        messages_list=[message_to_proto(msg)],
        message_object_trees=[object_tree],
    )
    state = MagicMock()
    state.start_session.return_value = "session-id"
    state.store_message_and_object_tree.return_value = (True, ["object-id"])

    # Execute
    response = push_messages(request=request, state=state)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_not_called()
    state.store_message_res.assert_not_called()
    state.start_session.assert_called_once_with(123)
    state.store_message_and_object_tree.assert_called_once()
    assert state.store_message_and_object_tree.call_args.args[1] == object_tree
    assert state.store_message_and_object_tree.call_args.args[2] == "session-id"
    assert response.session_id == "session-id"
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_called_once()


def test_push_task_events_stamps_authenticated_node_and_primary_task() -> None:
    """Lifecycle events become visible through the run's primary task stream."""
    # Prepare
    run_id = 123
    node_id = 456
    primary_task_id = 789
    event = TaskEvent(
        run_id=999,
        task_id=888,
        event="fl.node.fit.started",
        data='{"type":"fl.node.fit.started","node_id":1}',
    )
    request = FleetPushTaskEventsRequest(
        node=Node(node_id=node_id), run_id=run_id, events=[event]
    )
    state = MagicMock()
    state.get_run_info.return_value = [
        MagicMock(federation_id="federation", primary_task_id=primary_task_id)
    ]
    state.federation_manager.has_node.return_value = True
    state.get_run_status.return_value = {
        run_id: RunStatus(status="running", sub_status="", details="")
    }

    # Execute
    response = push_task_events(request=request, state=state)

    # Assert
    assert response.ByteSize() == 0
    state.federation_manager.has_node.assert_called_once_with(node_id, "federation")
    state.store_task_events.assert_called_once()
    stored_event = state.store_task_events.call_args.args[0][0]
    assert stored_event.run_id == run_id
    assert stored_event.task_id == primary_task_id
    assert strict_json_loads(stored_event.data) == {
        "type": "fl.node.fit.started",
        "node_id": node_id,
    }
