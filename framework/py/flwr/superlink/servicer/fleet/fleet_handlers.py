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
"""Fleet API handlers."""

from logging import DEBUG, ERROR, INFO

from google.protobuf.json_format import MessageToDict

from flwr.app import Message
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
from flwr.supercore import log
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStore
from flwr.supercore.run import InvalidRunStatusException, Run


class InvalidHeartbeatIntervalError(Exception):
    """Invalid heartbeat interval exception."""


def register_node(
    request: RegisterNodeFleetRequest,
    state: LinkState,
    enable_supernode_auth: bool,
) -> RegisterNodeFleetResponse:
    """Register a node (Fleet API only)."""
    error_context = (
        f"Attempted to register SuperNode with public key: {request.public_key!r}"
    )
    if enable_supernode_auth:
        raise FlowerError(
            ApiErrorCode.FLEET_SUPERNODE_REGISTRATION_DISABLED, error_context
        )
    try:
        node_id = state.create_node(
            NOOP_FLWR_AID, NOOP_ACCOUNT_NAME, request.public_key, 0
        )
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.PUBLIC_KEY_ALREADY_IN_USE, error_context
        ) from exc
    log(DEBUG, "[Fleet.RegisterNode] Registered node_id=%s", node_id)
    return RegisterNodeFleetResponse(node_id=node_id)


def activate_node(
    request: ActivateNodeRequest,
    state: LinkState,
) -> ActivateNodeResponse:
    """Activate a node."""
    error_context = (
        f"Attempted to register SuperNode with public key: {request.public_key!r}"
    )
    try:
        node_id = state.get_node_id_by_public_key(request.public_key)
        if node_id is None:
            raise ValueError("No SuperNode found with the given public key.")
        _validate_heartbeat_interval(request.heartbeat_interval)
        if not state.activate_node(node_id, request.heartbeat_interval):
            raise ValueError(
                f"SuperNode with node ID {node_id} could not be activated."
            )
    except InvalidHeartbeatIntervalError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL,
            f"{error_context}, exception: {exc}",
        ) from exc
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_NODE_ACTIVATION_FAILED,
            f"{error_context}, exception: {exc}",
        ) from exc
    log(INFO, "[Fleet.ActivateNode] Activated node_id=%s", node_id)
    return ActivateNodeResponse(node_id=node_id)


def deactivate_node(
    request: DeactivateNodeRequest,
    state: LinkState,
) -> DeactivateNodeResponse:
    """Deactivate a node."""
    try:
        if not state.deactivate_node(request.node_id):
            raise ValueError(
                f"SuperNode with node ID {request.node_id} could not be deactivated."
            )
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_NODE_DEACTIVATION_FAILED,
            f"SuperNode {request.node_id}, exception: {exc}",
        ) from exc
    log(INFO, "[Fleet.DeactivateNode] Deactivated node_id=%s", request.node_id)
    return DeactivateNodeResponse()


def unregister_node(
    request: UnregisterNodeFleetRequest,
    state: LinkState,
    enable_supernode_auth: bool,
) -> UnregisterNodeFleetResponse:
    """Unregister a node (Fleet API only)."""
    error_context = f"node_id={request.node_id}"
    if enable_supernode_auth:
        raise FlowerError(
            ApiErrorCode.FLEET_SUPERNODE_UNREGISTRATION_DISABLED,
            f"{error_context}, SuperNode unregistration is disabled through Fleet API.",
        )
    try:
        state.delete_node(NOOP_FLWR_AID, request.node_id)
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_NODE_UNREGISTRATION_FAILED,
            f"{error_context}, exception: {exc}",
        ) from exc
    log(DEBUG, "[Fleet.UnregisterNode] Unregistered node_id=%s", request.node_id)
    return UnregisterNodeFleetResponse()


def send_node_heartbeat(
    request: SendNodeHeartbeatRequest,
    state: LinkState,
) -> SendNodeHeartbeatResponse:
    """."""
    log(DEBUG, "[Fleet.SendNodeHeartbeat] Request: %s", MessageToDict(request))
    try:
        _validate_heartbeat_interval(request.heartbeat_interval)
    except InvalidHeartbeatIntervalError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL, str(exc)
        ) from exc
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
    log(DEBUG, "[Fleet.PullMessages] node_id=%s", request.node.node_id)
    log(DEBUG, "[Fleet.PullMessages] Request: %s", MessageToDict(request))
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
) -> PushMessagesResponse:
    """Push Messages handler."""
    try:
        if request.messages_list:
            log(
                INFO,
                "[Fleet.PushMessages] Push replies from node_id=%s",
                request.messages_list[0].metadata.src_node_id,
            )
        else:
            log(INFO, "[Fleet.PushMessages] No replies to push")
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
        )
        if abort_msg:
            raise InvalidRunStatusException(abort_msg)

        # Store Message in State and preregister its objects.
        session_id = state.start_session(run_id)
        _, objects_to_push = state.store_message_and_object_tree(
            msg, request.message_object_trees[0], session_id
        )

        # Build response
        response = PushMessagesResponse(
            reconnect=Reconnect(reconnect=5),
            results={msg.metadata.message_id: 0},
            objects_to_push=objects_to_push,
            session_id=session_id,
        )

        # Record outgoing traffic size
        bytes_sent = response.ByteSize()

        # Record traffic only if message was successfully processed
        # Only one message is processed per request
        state.store_traffic(run_id, bytes_sent=bytes_sent, bytes_recv=bytes_recv)
        if request.clientapp_runtime_list:
            state.add_clientapp_runtime(run_id, request.clientapp_runtime_list[0])

        return response
    except InvalidRunStatusException as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
            f"SuperNode {request.node.node_id}, exception: {exc.message}",
        ) from exc


def get_run(request: GetRunRequest, state: LinkState) -> GetRunResponse:
    """Get run information."""
    log(INFO, "[Fleet.GetRun] Requesting `Run` for run_id=%s", request.run_id)
    error_context = f"SuperNode {request.node.node_id}, run_id={request.run_id}"
    try:
        # Validate that the requesting SuperNode is part of the federation
        run = _validate_node_in_federation(state, request.node.node_id, request.run_id)

        # Abort if the run is not running
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            raise InvalidRunStatusException(abort_msg)

        return GetRunResponse(run=run_to_proto(run))
    except InvalidRunStatusException as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
            f"{error_context}, exception: {exc.message}",
        ) from exc
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_GET_RUN_FAILED,
            f"{error_context}, exception: {exc}",
        ) from exc


def get_fab(request: GetFabRequest, state: LinkState) -> GetFabResponse:
    """Get FAB."""
    log(INFO, "[Fleet.GetFab] Requesting FAB for fab_hash=%s", request.hash_str)
    error_context = (
        f"SuperNode {request.node.node_id}, run_id={request.run_id}, "
        f"fab_hash={request.hash_str}"
    )
    try:
        # Validate that the requesting SuperNode is part of the federation
        run = _validate_node_in_federation(state, request.node.node_id, request.run_id)

        # Abort if the run is not running
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            raise InvalidRunStatusException(abort_msg)

        if request.hash_str != run.fab_hash:
            raise ValueError(
                f"Requested FAB hash {request.hash_str} "
                f"does not match run FAB hash {run.fab_hash}.",
            )

        if fab := state.get_fab(request.hash_str):
            return GetFabResponse(fab=fab_to_proto(fab))

        raise ValueError(f"Found no FAB with hash: {request.hash_str}")
    except InvalidRunStatusException as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
            f"{error_context}, exception: {exc.message}",
        ) from exc
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_GET_FAB_FAILED,
            f"{error_context}, exception: {exc}",
        ) from exc


def push_object(request: PushObjectRequest, state: LinkState) -> PushObjectResponse:
    """Push Object."""
    log(DEBUG, "[Fleet.PushObject] Push Object with object_id=%s", request.object_id)
    error_context = (
        f"SuperNode {request.node.node_id}, run_id={request.run_id}, "
        f"object_id={request.object_id}"
    )
    try:
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            raise InvalidRunStatusException(abort_msg)

        stored = state.store_object(
            request.run_id,
            request.session_id,
            request.object_id,
            request.object_content,
        )
        # Record bytes traffic pushed from SuperNode
        if stored:
            state.store_traffic(
                request.run_id, bytes_sent=0, bytes_recv=len(request.object_content)
            )
        return PushObjectResponse(stored=stored)
    except InvalidRunStatusException as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
            f"{error_context}, exception: {exc.message}",
            public_details=f"Object_id: {request.object_id}",
        ) from exc


def pull_object(request: PullObjectRequest, state: LinkState) -> PullObjectResponse:
    """Pull Object."""
    log(DEBUG, "[Fleet.PullObject] Pull Object with object_id=%s", request.object_id)
    error_context = (
        f"SuperNode {request.node.node_id}, run_id={request.run_id}, "
        f"object_id={request.object_id}"
    )
    try:
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            raise InvalidRunStatusException(abort_msg)

        # Fetch from state
        content = state.get_object(request.run_id, request.object_id)
        if content is not None:
            object_available = content != b""
            # Record bytes traffic pulled by SuperNode
            if object_available:
                state.store_traffic(
                    request.run_id, bytes_sent=len(content), bytes_recv=0
                )
            return PullObjectResponse(
                object_found=True,
                object_available=object_available,
                object_content=content,
            )
        return PullObjectResponse(object_found=False, object_available=False)
    except InvalidRunStatusException as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
            f"{error_context}, exception: {exc.message}",
            public_details=f"Object_id: {request.object_id}",
        ) from exc


def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
    state: LinkState,
    store: ObjectStore,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received handler."""
    try:
        log(
            DEBUG,
            "[Fleet.ConfirmMessageReceived] Message with ID '%s' has been received",
            request.message_object_id,
        )
        abort_msg = check_abort(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
        )
        if abort_msg:
            raise InvalidRunStatusException(abort_msg)

        # Delete the message object
        store.delete(request.message_object_id)

        return ConfirmMessageReceivedResponse()
    except InvalidRunStatusException as exc:
        raise FlowerError(
            ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED,
            f"SuperNode {request.node.node_id}, exception: {exc.message}",
        ) from exc


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
    if not state.federation_manager.has_node(node_id, run.federation_id):
        raise ValueError(
            f"SuperNode is not part of the federation '{run.federation_id}'."
        )
    return run
