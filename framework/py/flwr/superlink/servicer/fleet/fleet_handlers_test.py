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


import hashlib
from unittest.mock import MagicMock, Mock

import pytest

from flwr.app import ConfigRecord, Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common.constant import (
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    SUPERLINK_NODE_ID,
    SubStatus,
)
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.fab_pb2 import GetFabRequest  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    DeactivateNodeRequest,
    PullMessagesRequest,
    PushMessagesRequest,
    RegisterNodeFleetRequest,
    UnregisterNodeFleetRequest,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ObjectTree,
    PullObjectRequest,
    PushObjectRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    NodeStatus,
    TaskType,
)
from flwr.supercore.corestate.utils_test import create_task_message
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from .fleet_handlers import (
    activate_node,
    confirm_message_received,
    deactivate_node,
    get_fab,
    get_run,
    pull_messages,
    pull_object,
    push_messages,
    push_object,
    register_node,
    unregister_node,
)


@pytest.fixture(name="state")
def fixture_state() -> LinkState:
    """Create isolated in-memory Fleet state."""
    return LinkStateFactory(
        FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), ObjectStoreFactory()
    ).state()


def _create_run(state: LinkState, transitions: int = 2, fab_hash: str = "") -> int:
    """Create a run and advance its primary task through the requested transitions."""
    run_id = state.create_run(
        fab_id="",
        fab_version="",
        fab_hash=fab_hash,
        override_config={},
        federation_id=NOOP_FEDERATION_ID,
        federation_config=None,
        flwr_aid="",
        primary_task_type=TaskType.SERVER_APP,
    )
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.primary_task_id is not None
    task_id = run.primary_task_id
    if transitions > 0:
        assert state.claim_task(task_id) is not None
    if transitions > 1:
        assert state.activate_task(task_id)
    if transitions > 2:
        assert state.finish_task(task_id, SubStatus.COMPLETED, "")
    return run_id


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
    state.get_messages.assert_called_once()
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
    state.start_session.assert_called_once_with(123)
    state.store_message_and_object_tree.assert_called_once()
    assert state.store_message_and_object_tree.call_args.args[1] == object_tree
    assert state.store_message_and_object_tree.call_args.args[2] == "session-id"
    assert response.session_id == "session-id"
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


def test_node_deactivation_and_unregistration(state: LinkState) -> None:
    """Fleet node lifecycle operations update state and reject invalid requests."""
    public_key = b"node-key"
    node_id = register_node(
        RegisterNodeFleetRequest(public_key=public_key), state, False
    ).node_id
    activate_node(
        ActivateNodeRequest(public_key=public_key, heartbeat_interval=30), state
    )
    deactivate_node(DeactivateNodeRequest(node_id=node_id), state)
    assert state.get_node_info(node_ids=[node_id])[0].status == NodeStatus.OFFLINE

    with pytest.raises(FlowerError) as error:
        deactivate_node(DeactivateNodeRequest(node_id=node_id + 1), state)
    assert error.value.code == ApiErrorCode.FLEET_NODE_DEACTIVATION_FAILED

    with pytest.raises(FlowerError) as error:
        unregister_node(UnregisterNodeFleetRequest(node_id=node_id), state, True)
    assert error.value.code == ApiErrorCode.FLEET_SUPERNODE_UNREGISTRATION_DISABLED

    unregister_node(UnregisterNodeFleetRequest(node_id=node_id), state, False)
    assert state.get_node_id_by_public_key(public_key) is None


@pytest.mark.parametrize("transitions", [0, 1, 2, 3])
def test_get_run_and_fab_require_running_status(
    state: LinkState, transitions: int
) -> None:
    """Run and FAB retrieval obey the run status policy."""
    content = b"fab-content"
    fab_hash = state.store_fab(Fab(hashlib.sha256(content).hexdigest(), content, {}))
    run_id = _create_run(state, transitions, fab_hash)
    node = Node(node_id=1)

    if transitions == 2:
        assert (
            get_run(GetRunRequest(run_id=run_id, node=node), state).run.run_id == run_id
        )
        assert (
            get_fab(
                GetFabRequest(run_id=run_id, node=node, hash_str=fab_hash), state
            ).fab.content
            == content
        )
    else:
        with pytest.raises(FlowerError) as error:
            get_run(GetRunRequest(run_id=run_id, node=node), state)
        assert error.value.code == ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED
        with pytest.raises(FlowerError) as error:
            get_fab(GetFabRequest(run_id=run_id, node=node, hash_str=fab_hash), state)
        assert error.value.code == ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED


def test_get_run_and_fab_permissions(
    state: LinkState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run membership and the run's FAB hash are checked before returning data."""
    content = b"fab-content"
    fab_hash = state.store_fab(Fab(hashlib.sha256(content).hexdigest(), content, {}))
    run_id = _create_run(state, fab_hash=fab_hash)
    node = Node(node_id=1)

    with pytest.raises(FlowerError) as error:
        get_fab(GetFabRequest(run_id=run_id, node=node, hash_str="wrong"), state)
    assert error.value.code == ApiErrorCode.FLEET_GET_FAB_FAILED

    monkeypatch.setattr(state.federation_manager, "has_node", Mock(return_value=False))
    with pytest.raises(FlowerError) as error:
        get_run(GetRunRequest(run_id=run_id, node=node), state)
    assert error.value.code == ApiErrorCode.FLEET_GET_RUN_FAILED
    with pytest.raises(FlowerError) as error:
        get_fab(GetFabRequest(run_id=run_id, node=node, hash_str=fab_hash), state)
    assert error.value.code == ApiErrorCode.FLEET_GET_FAB_FAILED


def test_object_transfer_and_traffic(state: LinkState) -> None:
    """Transferred objects are available and their bytes are accounted for."""
    run_id = _create_run(state)
    node = Node(node_id=1)
    obj = ConfigRecord({"value": 123})
    content = obj.deflate()
    session_id = state.start_session(run_id)
    state.preregister_object_tree(get_object_tree(obj), session_id)

    request = PullObjectRequest(node=node, run_id=run_id, object_id=obj.object_id)
    pending = pull_object(request, state)
    assert pending.object_found and not pending.object_available

    assert push_object(
        PushObjectRequest(
            node=node,
            run_id=run_id,
            object_id=obj.object_id,
            object_content=content,
            session_id=session_id,
        ),
        state,
    ).stored
    fetched = pull_object(request, state)
    assert fetched.object_found and fetched.object_available
    assert fetched.object_content == content
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.bytes_recv == len(content)
    assert run.bytes_sent == len(content)


def test_object_operations_reject_non_running_run(state: LinkState) -> None:
    """Object transfer and message confirmation require a running run."""
    run_id = _create_run(state, transitions=0)
    node = Node(node_id=1)
    with pytest.raises(FlowerError) as error:
        push_object(PushObjectRequest(node=node, run_id=run_id), state)
    assert error.value.code == ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED
    with pytest.raises(FlowerError) as error:
        pull_object(PullObjectRequest(node=node, run_id=run_id), state)
    assert error.value.code == ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED
    with pytest.raises(FlowerError) as error:
        confirm_message_received(
            ConfirmMessageReceivedRequest(node=node, run_id=run_id), state
        )
    assert error.value.code == ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED


@pytest.mark.parametrize("transitions", [0, 1, 2, 3])
def test_push_messages_run_status(state: LinkState, transitions: int) -> None:
    """Finished runs reject new message headers while active runs accept them."""
    node_id = state.create_node(NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, b"node", 30)
    run_id = _create_run(state, transitions)
    message = create_ins_message(
        src_node_id=node_id, dst_node_id=SUPERLINK_NODE_ID, run_id=run_id
    )
    response = push_messages(
        PushMessagesRequest(
            node=Node(node_id=node_id),
            messages_list=[message],
            message_object_trees=[get_object_tree(message_from_proto(message))],
        ),
        state,
    )
    assert bool(response.objects_to_push) == (transitions != 3)


@pytest.mark.parametrize("objects_available", [False, True])
def test_pull_messages_requires_objects(
    state: LinkState, objects_available: bool
) -> None:
    """Only messages with registered object trees are delivered."""
    node_id = state.create_node(NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, b"node", 30)
    run_id = _create_run(state)
    message = message_from_proto(
        create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )
    )
    message.metadata.__dict__["_message_id"] = message.object_id
    assert state.store_messages([message]) == [True]
    if objects_available:
        state.object_store.preregister(run_id, get_object_tree(message))

    response = pull_messages(
        PullMessagesRequest(node=Node(node_id=node_id)), state, state.object_store
    )
    assert len(response.messages_list) == int(objects_available)
    assert len(response.message_object_trees) == int(objects_available)


def test_confirm_message_received_releases_objects(state: LinkState) -> None:
    """Acknowledging a delivered message releases its stored objects."""
    node_id = state.create_node(NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, b"node", 30)
    run_id = _create_run(state)
    message = message_from_proto(
        create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=node_id, run_id=run_id
        )
    )
    message.content["config"] = ConfigRecord({"value": 123})  # pylint: disable=E1137
    message.metadata.__dict__["_message_id"] = message.object_id
    assert state.store_messages([message]) == [True]
    state.object_store.preregister(run_id, get_object_tree(message))
    for object_id, obj in get_all_nested_objects(message).items():
        state.object_store.put(object_id=object_id, object_content=obj.deflate())
    assert len(state.object_store) > 0

    confirm_message_received(
        ConfirmMessageReceivedRequest(
            node=Node(node_id=node_id),
            run_id=run_id,
            message_object_id=message.object_id,
        ),
        state,
    )
    assert len(state.object_store) == 0
