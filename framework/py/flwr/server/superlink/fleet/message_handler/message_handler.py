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
"""Fleet API message handlers."""

from logging import ERROR

from flwr.common import Message, log
from flwr.common.constant import (
    HEARTBEAT_MAX_INTERVAL,
    HEARTBEAT_MIN_INTERVAL,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    Status,
)
from flwr.common.serde import (
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.common.typing import Fab, InvalidRunStatusException, Run
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    ActivateNodeResponse,
    DeactivateNodeRequest,
    DeactivateNodeResponse,
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
    Reconnect,
    RegisterNodeFleetRequest,
    RegisterNodeFleetResponse,
    UnregisterNodeFleetRequest,
    UnregisterNodeFleetResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.utils import check_abort
from flwr.supercore.ffs import Ffs
from flwr.supercore.inflatable.inflatable_object import UnexpectedObjectContentError
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStore


class InvalidHeartbeatIntervalError(Exception):
    """Invalid heartbeat interval exception."""


def register_node(
    request: RegisterNodeFleetRequest,
    state: LinkState,
) -> RegisterNodeFleetResponse:
    """Register a node (Fleet API only)."""
    node_id = state.create_node(NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, request.public_key, 0)
    return RegisterNodeFleetResponse(node_id=node_id)


def activate_node(
    request: ActivateNodeRequest,
    state: LinkState,
) -> ActivateNodeResponse:
    """Activate a node."""
    node_id = state.get_node_id_by_public_key(request.public_key)
    if node_id is None:
        raise ValueError("No SuperNode found with the given public key.")
    _validate_heartbeat_interval(request.heartbeat_interval)
    if not state.activate_node(node_id, request.heartbeat_interval):
        raise ValueError(f"SuperNode with node ID {node_id} could not be activated.")
    return ActivateNodeResponse(node_id=node_id)


def deactivate_node(
    request: DeactivateNodeRequest,
    state: LinkState,
) -> DeactivateNodeResponse:
    """Deactivate a node."""
    if not state.deactivate_node(request.node_id):
        raise ValueError(
            f"SuperNode with node ID {request.node_id} could not be deactivated."
        )
    return DeactivateNodeResponse()


def unregister_node(
    request: UnregisterNodeFleetRequest,
    state: LinkState,
) -> UnregisterNodeFleetResponse:
    """Unregister a node (Fleet API only)."""
    state.delete_node(NOOP_FLWR_AID, request.node_id)
    return UnregisterNodeFleetResponse()


def send_node_heartbeat(
    request: SendNodeHeartbeatRequest,  # pylint: disable=unused-argument
    state: LinkState,  # pylint: disable=unused-argument
) -> SendNodeHeartbeatResponse:
    """."""
    _validate_heartbeat_interval(request.heartbeat_interval)
    res = state.acknowledge_node_heartbeat(
        request.node.node_id, request.heartbeat_interval
    )
    return SendNodeHeartbeatResponse(success=res)


def pull_messages(  # pylint: disable=too-many-locals
    request: PullMessagesRequest,
    state: LinkState,
    store: ObjectStore,
) -> PullMessagesResponse:
    """Pull Messages handler."""
    # Get node_id if client node is not anonymous
    node = request.node  # pylint: disable=no-member
    node_id: int = node.node_id

    # Retrieve Message from State
    message_list: list[Message] = state.get_message_ins(node_id=node_id, limit=1)

    # Convert to Messages
    msg_proto = []
    trees = []
    run_id_to_record: int | None = None

    for msg in message_list:
        try:
            # Retrieve Message object tree from ObjectStore
            msg_object_id = msg.metadata.message_id
            obj_tree = store.get_object_tree(msg_object_id)

            # Add Message and its object tree to the response
            msg_proto.append(message_to_proto(msg))
            trees.append(obj_tree)

            # Track run_id for traffic recording
            run_id_to_record = msg.metadata.run_id

        except NoObjectInStoreError as e:
            log(ERROR, e.message)
            # Delete message ins from state
            state.delete_messages(message_ins_ids={msg_object_id})

    response = PullMessagesResponse(messages_list=msg_proto, message_object_trees=trees)

    # Record incoming traffic size
    bytes_recv = request.ByteSize()

    # Record traffic only if message was successfully processed
    # All messages in this request are assumed to belong to the same run
    if run_id_to_record is not None:
        # Record outgoing traffic size
        bytes_sent = response.ByteSize()
        state.store_traffic(
            run_id_to_record, bytes_sent=bytes_sent, bytes_recv=bytes_recv
        )

    return response


def push_messages(
    request: PushMessagesRequest,
    state: LinkState,
    store: ObjectStore,
) -> PushMessagesResponse:
    """Push Messages handler."""
    # Convert Message from proto
    msg = message_from_proto(message_proto=request.messages_list[0])
    run_id = msg.metadata.run_id

    # Record incoming traffic size
    bytes_recv = request.ByteSize()

    # Abort if the run is not running
    abort_msg = check_abort(
        run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Store Message object to descendants mapping and preregister objects
    objects_to_push: set[str] = set()
    for object_tree in request.message_object_trees:
        objects_to_push |= set(store.preregister(run_id, object_tree))
    # Store Message in State
    message_id: str | None = state.store_message_res(message=msg)

    # Build response
    response = PushMessagesResponse(
        reconnect=Reconnect(reconnect=5),
        results={str(message_id): 0},
        objects_to_push=objects_to_push,
    )

    # Record outgoing traffic size
    bytes_sent = response.ByteSize()

    # Record traffic only if message was successfully processed
    # Only one message is processed per request
    state.store_traffic(run_id, bytes_sent=bytes_sent, bytes_recv=bytes_recv)
    if request.clientapp_runtime_list:
        state.add_clientapp_runtime(run_id, request.clientapp_runtime_list[0])

    return response


def get_run(
    request: GetRunRequest, state: LinkState, store: ObjectStore
) -> GetRunResponse:
    """Get run information."""
    # Validate that the requesting SuperNode is part of the federation
    run = _validate_node_in_federation(state, request.node.node_id, request.run_id)

    # Abort if the run is not running
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    return GetRunResponse(run=run_to_proto(run))


def get_fab(
    request: GetFabRequest, ffs: Ffs, state: LinkState, store: ObjectStore
) -> GetFabResponse:
    """Get FAB."""
    # Validate that the requesting SuperNode is part of the federation
    _validate_node_in_federation(state, request.node.node_id, request.run_id)

    # Abort if the run is not running
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    if result := ffs.get(request.hash_str):
        fab = Fab(request.hash_str, result[0], result[1])
        return GetFabResponse(fab=fab_to_proto(fab))

    raise ValueError(f"Found no FAB with hash: {request.hash_str}")


def push_object(
    request: PushObjectRequest, state: LinkState, store: ObjectStore
) -> PushObjectResponse:
    """Push Object."""
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    stored = False
    try:
        store.put(request.object_id, request.object_content)
        stored = True
        # Record bytes traffic pushed from SuperNode
        state.store_traffic(
            request.run_id, bytes_sent=0, bytes_recv=len(request.object_content)
        )
    except (NoObjectInStoreError, ValueError) as e:
        log(ERROR, str(e))
    except UnexpectedObjectContentError as e:
        # Object content is not valid
        log(ERROR, str(e))
        raise
    return PushObjectResponse(stored=stored)


def pull_object(
    request: PullObjectRequest, state: LinkState, store: ObjectStore
) -> PullObjectResponse:
    """Pull Object."""
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Fetch from store
    content = store.get(request.object_id)
    if content is not None:
        object_available = content != b""
        # Record bytes traffic pulled by SuperNode
        if object_available:
            state.store_traffic(request.run_id, bytes_sent=len(content), bytes_recv=0)
        return PullObjectResponse(
            object_found=True,
            object_available=object_available,
            object_content=content,
        )
    return PullObjectResponse(object_found=False, object_available=False)


def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
    state: LinkState,
    store: ObjectStore,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received handler."""
    abort_msg = check_abort(
        request.run_id,
        [Status.PENDING, Status.STARTING, Status.FINISHED],
        state,
        store,
    )
    if abort_msg:
        raise InvalidRunStatusException(abort_msg)

    # Delete the message object
    store.delete(request.message_object_id)

    return ConfirmMessageReceivedResponse()


def _validate_heartbeat_interval(interval: float) -> None:
    """Raise if heartbeat interval is out of bounds."""
    if not HEARTBEAT_MIN_INTERVAL <= interval <= HEARTBEAT_MAX_INTERVAL:
        raise InvalidHeartbeatIntervalError(
            f"Heartbeat interval {interval} is out of bounds "
            f"[{HEARTBEAT_MIN_INTERVAL}, {HEARTBEAT_MAX_INTERVAL}]."
        )


def _validate_node_in_federation(
    state: LinkState,
    node_id: int,
    run_id: int,
) -> Run:
    """Raise if the requesting SuperNode is not part of the federation the run belongs
    to."""
    if not (runs := state.get_run_info(run_ids=[run_id])):
        raise ValueError(f"Run ID not found: {run_id}")

    run = runs[0]
    if not state.federation_manager.has_node(node_id, run.federation):
        raise ValueError(f"SuperNode is not part of the federation '{run.federation}'.")
    return run
