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
"""ServerAppIo API servicer."""


from logging import DEBUG, ERROR, INFO
from typing import cast

import grpc

from flwr.common import Message
from flwr.common.constant import SUPERLINK_NODE_ID, Status
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_status_from_proto,
    run_to_proto,
)
from flwr.common.typing import RunStatus
from flwr.proto import serverappio_pb2_grpc  # pylint: disable=E0611
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendAppHeartbeatRequest,
    SendAppHeartbeatResponse,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import (  # pylint: disable=E0611
    GetFederationOptionsRequest,
    GetFederationOptionsResponse,
    GetRunRequest,
    GetRunResponse,
    UpdateRunStatusRequest,
    UpdateRunStatusResponse,
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.utils import abort_if
from flwr.server.utils.validator import validate_message
from flwr.supercore.constant import (
    TASK_TYPES_REQUIRING_CONNECTOR_REF,
    TASK_TYPES_REQUIRING_FAB_HASH,
    TASK_TYPES_REQUIRING_MODEL_REF,
    TaskType,
)
from flwr.supercore.inflatable.inflatable_object import (
    UnexpectedObjectContentError,
    get_all_nested_objects,
    get_object_tree,
    no_object_id_recompute,
)
from flwr.supercore.interceptors import get_authenticated_task
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStoreFactory
from flwr.supercore.servicers import AppIoServicer


class ServerAppIoServicer(AppIoServicer, serverappio_pb2_grpc.ServerAppIoServicer):
    """ServerAppIo API servicer."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.objectstore_factory = objectstore_factory

    def state(self) -> LinkState:
        """Return the LinkState instance."""
        return self.state_factory.state()

    def ListAppsToLaunch(
        self,
        request: ListAppsToLaunchRequest,
        context: grpc.ServicerContext,
    ) -> ListAppsToLaunchResponse:
        """Get run IDs with pending messages."""
        log(DEBUG, "ServerAppIoServicer.ListAppsToLaunch")

        # Initialize state connection
        state = self.state_factory.state()

        # Get IDs of runs in pending status
        pending_run_ids = [
            run.run_id for run in state.get_run_info(statuses=[Status.PENDING])
        ]

        # Return run IDs
        return ListAppsToLaunchResponse(run_ids=pending_run_ids)

    def RequestToken(
        self, request: RequestTokenRequest, context: grpc.ServicerContext
    ) -> RequestTokenResponse:
        """Request token."""
        log(DEBUG, "ServerAppIoServicer.RequestToken")

        # Initialize state connection
        state = self.state_factory.state()

        # Attempt to create a token for the provided run ID
        run = state.get_run_info(run_ids=[request.run_id])[0]
        token = state.claim_task(cast(int, run.primary_task_id))

        if not token:
            return RequestTokenResponse(token="")

        # Keep run status working
        state.update_run_status(request.run_id, RunStatus(Status.STARTING, "", ""))

        # Return the token
        return RequestTokenResponse(token=token)

    def ClaimTask(
        self, request: ClaimTaskRequest, context: grpc.ServicerContext
    ) -> ClaimTaskResponse:
        """Claim a pending task."""
        res = super().ClaimTask(request, context)

        # Keep run status working
        if res.HasField("token"):
            state = self.state_factory.state()
            task = state.get_tasks(task_ids=[request.task_id])[0]
            state.update_run_status(task.run_id, RunStatus(Status.STARTING, "", ""))
        return res

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "ServerAppIoServicer.GetNodes")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        all_ids: set[int] = state.get_nodes(request.run_id)
        nodes: list[Node] = [Node(node_id=node_id) for node_id in all_ids]
        return GetNodesResponse(nodes=nodes)

    def CreateTask(
        self, request: CreateTaskRequest, context: grpc.ServicerContext
    ) -> CreateTaskResponse:
        """Create a task."""
        log(DEBUG, "ServerAppIoServicer.CreateTask")

        state = self.state_factory.state()
        _validate_create_task_request(request, context)

        task_id = state.create_task(
            task_type=request.type,
            run_id=request.run_id,
            fab_hash=request.fab_hash if request.HasField("fab_hash") else None,
            model_ref=request.model_ref if request.HasField("model_ref") else None,
            connector_ref=(
                request.connector_ref if request.HasField("connector_ref") else None
            ),
        )
        if task_id is None:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to create task")
            raise RuntimeError("This line should never be reached.")

        return CreateTaskResponse(task_id=task_id)

    def PushMessages(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PushMessages")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        # Validate request and insert in State
        _raise_if(
            validation_error=len(request.messages_list) == 0,
            request_name="PushMessages",
            detail="`messages_list` must not be empty",
        )
        message_ids: list[str | None] = []
        objects_to_push: set[str] = set()
        for message_proto, object_tree in zip(
            request.messages_list, request.message_object_trees, strict=True
        ):
            message = message_from_proto(message_proto=message_proto)
            validation_errors = validate_message(message, is_reply_message=False)
            _raise_if(
                validation_error=bool(validation_errors),
                request_name="PushMessages",
                detail=", ".join(validation_errors),
            )
            _raise_if(
                validation_error=request.run_id != message.metadata.run_id,
                request_name="PushMessages",
                detail="`Message.metadata` has mismatched `run_id`",
            )
            # Store objects
            objects_to_push |= set(store.preregister(request.run_id, object_tree))
            # Store message
            message_id: str | None = state.store_message_ins(message=message)
            message_ids.append(message_id)

        return PushAppMessagesResponse(
            message_ids=[
                str(message_id) if message_id else "" for message_id in message_ids
            ],
            objects_to_push=objects_to_push,
        )

    def PullMessages(  # pylint: disable=R0914
        self, request: PullAppMessagesRequest, context: grpc.ServicerContext
    ) -> PullAppMessagesResponse:
        """Pull a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PullMessages")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        # Read from state
        messages_res: list[Message] = state.get_message_res(
            message_ids=set(request.message_ids)
        )

        # Register messages generated by LinkState in the Store for consistency
        for msg_res in messages_res:
            if msg_res.metadata.src_node_id == SUPERLINK_NODE_ID:
                with no_object_id_recompute():
                    all_objects = get_all_nested_objects(msg_res)
                    # Preregister
                    store.preregister(request.run_id, get_object_tree(msg_res))
                    # Store objects
                    for obj_id, obj in all_objects.items():
                        store.put(obj_id, obj.deflate())

        # Delete the instruction Messages and their replies if found
        message_ins_ids_to_delete = {
            msg_res.metadata.reply_to_message_id for msg_res in messages_res
        }

        state.delete_messages(message_ins_ids=message_ins_ids_to_delete)

        # Convert Messages to proto
        messages_list = []
        trees = []
        while messages_res:
            msg = messages_res.pop(0)

            # Skip `run_id` check for SuperLink generated replies
            if msg.metadata.src_node_id != SUPERLINK_NODE_ID:
                _raise_if(
                    validation_error=request.run_id != msg.metadata.run_id,
                    request_name="PullMessages",
                    detail="`message.metadata` has mismatched `run_id`",
                )

            try:
                msg_object_id = msg.metadata.message_id
                obj_tree = store.get_object_tree(msg_object_id)
                # Add message and object tree to the response
                messages_list.append(message_to_proto(msg))
                trees.append(obj_tree)
            except NoObjectInStoreError as e:
                log(ERROR, e.message)
                # Delete message ins from state
                state.delete_messages(message_ins_ids={msg_object_id})

        return PullAppMessagesResponse(
            messages_list=messages_list, message_object_trees=trees
        )

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "ServerAppIoServicer.GetRun")

        # Init state
        state: LinkState = self.state_factory.state()

        # Retrieve run information
        runs = state.get_run_info(run_ids=[request.run_id])

        if not runs:
            return GetRunResponse()

        return GetRunResponse(run=run_to_proto(runs[0]))

    def PullAppInputs(
        self, request: PullAppInputsRequest, context: grpc.ServicerContext
    ) -> PullAppInputsResponse:
        """Pull ServerApp process inputs."""
        log(DEBUG, "ServerAppIoServicer.PullAppInputs")
        # Init access to LinkState
        state = self.state_factory.state()

        # Get the authenticated task and associated run ID
        task = get_authenticated_task()
        run_id = task.run_id

        # Retrieve Context, Run and Fab for the run_id
        serverapp_ctxt = state.get_serverapp_context(run_id)
        runs = state.get_run_info(run_ids=[run_id])
        run = runs[0] if runs else None
        fab = state.get_fab(run.fab_hash) if run and run.fab_hash else None
        if run and fab and serverapp_ctxt:
            # Update run status to RUNNING
            if state.activate_task(task.task_id):
                log(INFO, "Started task %d of run %d", task.task_id, run_id)
                # Keep run status working
                state.update_run_status(run_id, RunStatus(Status.RUNNING, "", ""))
                return PullAppInputsResponse(
                    context=context_to_proto(serverapp_ctxt),
                    run=run_to_proto(run),
                    fab=fab_to_proto(fab),
                    federation_config=state.get_federation_config(run_id),
                    task_id=task.task_id,
                )

        # Raise an exception if the Run or Fab is not found,
        # or if the status cannot be updated to RUNNING
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Failed to start task {task.task_id} of run {run_id}",
        )
        raise RuntimeError("Unreachable code")  # for mypy

    def PushAppOutputs(
        self, request: PushAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushAppOutputsResponse:
        """Push ServerApp process outputs."""
        log(DEBUG, "ServerAppIoServicer.PushAppOutputs")

        # Get the authenticated task and associated run ID
        task = get_authenticated_task()
        run_id = task.run_id

        # Init state and store
        state = self.state_factory.state()

        # Finish the task
        if state.finish_task(
            task.task_id, sub_status=request.sub_status, details=request.details
        ):
            log(INFO, "Finished task %d of run %d", task.task_id, run_id)
            # Keep run status working
            state.update_run_status(
                run_id, RunStatus(Status.FINISHED, request.sub_status, request.details)
            )
            if request.HasField("context"):
                state.set_serverapp_context(run_id, context_from_proto(request.context))
        else:
            log(ERROR, "Failed to finish task %d of run %s", task.task_id, run_id)
        return PushAppOutputsResponse()

    def UpdateRunStatus(
        self, request: UpdateRunStatusRequest, context: grpc.ServicerContext
    ) -> UpdateRunStatusResponse:
        """Update the status of a run."""
        log(DEBUG, "ServerAppIoServicer.UpdateRunStatus")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is finished
        abort_if(request.run_id, [Status.FINISHED], state, store, context)

        # Update the run status
        state.update_run_status(
            run_id=request.run_id, new_status=run_status_from_proto(request.run_status)
        )

        # If the run is finished, delete the run from ObjectStore
        if request.run_status.status == Status.FINISHED:
            # Remove the token once the run completes.
            state.delete_token(request.run_id)
            # Delete all objects related to the run
            store.delete_objects_in_run(request.run_id)

        return UpdateRunStatusResponse()

    def PushLogs(
        self, request: PushLogsRequest, context: grpc.ServicerContext
    ) -> PushLogsResponse:
        """Push logs."""
        log(DEBUG, "ServerAppIoServicer.PushLogs")
        state = self.state_factory.state()

        # Add logs to LinkState
        merged_logs = "".join(request.logs)
        state.add_serverapp_log(request.run_id, merged_logs)
        return PushLogsResponse()

    def GetFederationOptions(
        self, request: GetFederationOptionsRequest, context: grpc.ServicerContext
    ) -> GetFederationOptionsResponse:
        """Get Federation Options associated with a run."""
        log(DEBUG, "ServerAppIoServicer.GetFederationOptions")
        raise NotImplementedError("To be removed")

    def SendAppHeartbeat(
        self, request: SendAppHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendAppHeartbeatResponse:
        """Handle a heartbeat from an app process."""
        log(DEBUG, "ServerAppIoServicer.SendAppHeartbeat")

        # Get the authenticated task
        task = get_authenticated_task()

        # Init state
        state = self.state_factory.state()

        # Acknowledge the heartbeat
        success = state.acknowledge_task_heartbeat(task.task_id)
        return SendAppHeartbeatResponse(success=success)

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(DEBUG, "ServerAppIoServicer.PushObject")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        if request.node.node_id != SUPERLINK_NODE_ID:
            # Cancel insertion in ObjectStore
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Unexpected node ID.")

        # Insert in store
        stored = False
        try:
            store.put(request.object_id, request.object_content)
            stored = True
        except (NoObjectInStoreError, ValueError) as e:
            log(ERROR, str(e))
        except UnexpectedObjectContentError as e:
            # Object content is not valid
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))

        return PushObjectResponse(stored=stored)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(DEBUG, "ServerAppIoServicer.PullObject")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        if request.node.node_id != SUPERLINK_NODE_ID:
            # Cancel insertion in ObjectStore
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Unexpected node ID.")

        # Fetch from store
        content = store.get(request.object_id)
        if content is not None:
            object_available = content != b""
            return PullObjectResponse(
                object_found=True,
                object_available=object_available,
                object_content=content,
            )
        return PullObjectResponse(object_found=False, object_available=False)

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        log(DEBUG, "ServerAppIoServicer.ConfirmMessageReceived")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Abort if the run is not running
        abort_if(
            request.run_id,
            [Status.PENDING, Status.STARTING, Status.FINISHED],
            state,
            store,
            context,
        )

        # Delete the message object
        store.delete(request.message_object_id)

        return ConfirmMessageReceivedResponse()


def _raise_if(validation_error: bool, request_name: str, detail: str) -> None:
    """Raise a `ValueError` with a detailed message if a validation error occurs."""
    if validation_error:
        raise ValueError(f"Malformed {request_name}: {detail}")


def _validate_create_task_request(
    request: CreateTaskRequest, context: grpc.ServicerContext
) -> None:
    """Validate the task creation request."""
    try:
        task_type = TaskType(request.type)
    except ValueError:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Invalid task type: {request.type}",
        )

    if task_type in TASK_TYPES_REQUIRING_FAB_HASH and not request.fab_hash:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Task type '{request.type}' requires fab_hash.",
        )

    if task_type in TASK_TYPES_REQUIRING_MODEL_REF and not request.model_ref:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Task type '{request.type}' requires model_ref.",
        )

    if task_type in TASK_TYPES_REQUIRING_CONNECTOR_REF and not request.connector_ref:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Task type '{request.type}' requires connector_ref.",
        )
