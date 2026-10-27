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

import pytest

from flwr.app import Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common.serde import message_to_proto
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    PullMessagesRequest,
    PushMessagesRequest,
    RegisterNodeFleetRequest,
)
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.supercore.corestate.utils_test import create_task_message
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError

from .fleet_handlers import activate_node, pull_messages, push_messages, register_node


def test_register_node_auth_error_is_transport_independent() -> None:
    """The handler rejects registration before touching state when auth is enabled."""
    state = MagicMock()

    with pytest.raises(FlowerError) as exc_info:
        register_node(
            RegisterNodeFleetRequest(public_key=b"node"),
            state,
            enable_supernode_auth=True,
        )

    assert exc_info.value.code == ApiErrorCode.FLEET_SUPERNODE_REGISTRATION_DISABLED
    state.create_node.assert_not_called()


def test_activate_node_maps_interval_error_without_grpc() -> None:
    """The handler raises the same Flower error without a gRPC servicer."""
    state = MagicMock()
    state.get_node_id_by_public_key.return_value = 123

    with pytest.raises(FlowerError) as exc_info:
        activate_node(
            ActivateNodeRequest(public_key=b"node", heartbeat_interval=1), state
        )

    assert exc_info.value.code == ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL
    state.activate_node.assert_not_called()


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
    state.get_messages.assert_called_once()
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
    state.get_messages.return_value = [msg]
    store = MagicMock()
    store.get_object_tree.return_value = {}

    # Execute
    pull_messages(request=request, state=state, store=store)

    # Assert
    state.get_messages.assert_called_once()
    store.get_object_tree.assert_called_once_with("msg-234")
    state.store_traffic.assert_called_once()
    # Verify store_traffic was called with run_id=123, bytes_sent > 0, bytes_recv=0
    call_args = state.store_traffic.call_args
    assert call_args[0][0] == 234  # run_id
    assert call_args[1]["bytes_sent"] > 0
    assert call_args[1]["bytes_recv"] > 0


@pytest.mark.parametrize("stored", [False, True])
def test_push_messages(stored: bool) -> None:
    """Test push_messages."""
    # Prepare
    msg = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=123,
            message_id="",
            group_id="",
            src_node_id=123,
            dst_node_id=1,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=123,
            message_type="query",
        ),
    )

    msg.metadata.__dict__["_message_id"] = msg.object_id
    object_tree = ObjectTree(object_id="object-id")
    request = PushMessagesRequest(
        node=Node(node_id=123),
        messages_list=[message_to_proto(msg)],
        message_object_trees=[object_tree],
    )
    state = MagicMock()
    state.start_session.return_value = "session-id"
    state.store_message_and_object_tree.return_value = (
        stored,
        ["object-id"] if stored else [],
    )

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
    assert not response.results
    assert list(response.objects_to_push) == (["object-id"] if stored else [])
    assert state.store_traffic.call_count == int(stored)


def test_push_messages_binds_authenticated_source() -> None:
    """A SuperNode cannot submit another node's Messages."""
    message = create_task_message(node_id=42)
    state = MagicMock()
    request = PushMessagesRequest(
        node=Node(node_id=43),
        messages_list=[message_to_proto(message)],
        message_object_trees=[ObjectTree(object_id=message.metadata.message_id)],
    )
    with pytest.raises(FlowerError) as error:
        push_messages(request, state)
    assert error.value.code == ApiErrorCode.FLEET_MESSAGE_SOURCE_NODE_ID_MISMATCH
    state.store_message_and_object_tree.assert_not_called()
