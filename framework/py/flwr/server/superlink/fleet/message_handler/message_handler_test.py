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
from flwr.common.constant import Status
from flwr.common.serde import message_to_proto
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    PullMessagesRequest,
    PushMessagesRequest,
)
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.supercore.date import now
from flwr.supercore.run import RunStatus

from .message_handler import pull_messages, push_messages


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

    request = PushMessagesRequest(messages_list=[message_to_proto(msg)])
    state = MagicMock()
    store = MagicMock()

    # Execute
    push_messages(request=request, state=state, store=store)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_not_called()
    state.store_message_res.assert_called_once()
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_called_once()


def test_push_messages_cleans_up_failed_message_objects() -> None:
    """Test push_messages cleanup preregistered objects on message store failure."""
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
    state.store_message_res.return_value = None
    state.get_run_status.side_effect = [
        {123: RunStatus(status=Status.RUNNING, sub_status="", details="")},
        {123: RunStatus(status=Status.FINISHED, sub_status="", details="")},
    ]
    store = MagicMock()
    store.preregister.return_value = ["object-id"]

    response = push_messages(request=request, state=state, store=store)

    store.delete.assert_called_once_with("object-id")
    store.delete_objects_in_run.assert_not_called()
    assert not response.objects_to_push
