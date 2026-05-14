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

import grpc

from flwr.common import Message
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.proto import serverappio_pb2_grpc  # pylint: disable=E0611
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
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
)
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.utils.validator import validate_message
from flwr.supercore.constant import TaskType
from flwr.supercore.inflatable.inflatable_object import (
    UnexpectedObjectContentError,
    get_all_nested_objects,
    get_object_tree,
    no_object_id_recompute,
)
from flwr.supercore.interceptors import get_authenticated_task
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStoreFactory
from flwr.supercore.servicers import AppIoServicer

SERVERAPPIO_ENDPOINT_UNAVAILABLE_MESSAGE = (
    "Some ServerAppIo API endpoints are only available for Deployment Runtime runs."
)


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

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "ServerAppIoServicer.GetNodes")

        # Init state
        state = self.state_factory.state()

        run_id = _get_authenticated_serverapp_run_id(context)

        all_ids: set[int] = state.get_nodes(run_id)
        nodes: list[Node] = [Node(node_id=node_id) for node_id in all_ids]
        return GetNodesResponse(nodes=nodes)

    def PushMessages(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push a set of Messages."""
        log(DEBUG, "ServerAppIoServicer.PushMessages")

        # Init state and store
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        run_id = _get_authenticated_serverapp_run_id(context)

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
                validation_error=run_id != message.metadata.run_id,
                request_name="PushMessages",
                detail="`Message.metadata` has mismatched `run_id`",
            )
            # Store objects
            objects_to_push |= set(store.preregister(run_id, object_tree))
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

        run_id = _get_authenticated_serverapp_run_id(context)

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
                    store.preregister(run_id, get_object_tree(msg_res))
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
                    validation_error=run_id != msg.metadata.run_id,
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

    def PullTaskInput(
        self, request: PullTaskInputRequest, context: grpc.ServicerContext
    ) -> PullTaskInputResponse:
        """Pull ServerApp process inputs."""
        log(DEBUG, "ServerAppIoServicer.PullTaskInput")
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
            if state.activate_task(task.task_id):
                log(INFO, "Started task %d of run %d", task.task_id, run_id)
                return PullTaskInputResponse(
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

    def PushTaskOutput(
        self, request: PushTaskOutputRequest, context: grpc.ServicerContext
    ) -> PushTaskOutputResponse:
        """Push ServerApp process outputs."""
        log(DEBUG, "ServerAppIoServicer.PushTaskOutput")

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
            if request.HasField("context"):
                state.set_serverapp_context(run_id, context_from_proto(request.context))
        else:
            log(ERROR, "Failed to finish task %d of run %s", task.task_id, run_id)
        return PushTaskOutputResponse()

    def GetFederationOptions(
        self, request: GetFederationOptionsRequest, context: grpc.ServicerContext
    ) -> GetFederationOptionsResponse:
        """Get Federation Options associated with a run."""
        log(DEBUG, "ServerAppIoServicer.GetFederationOptions")
        raise NotImplementedError("To be removed")

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(DEBUG, "ServerAppIoServicer.PushObject")

        # Init store
        store = self.objectstore_factory.store()

        _ = _get_authenticated_serverapp_run_id(context)

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

        # Init store
        store = self.objectstore_factory.store()

        _ = _get_authenticated_serverapp_run_id(context)

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

        # Init store
        store = self.objectstore_factory.store()

        _ = _get_authenticated_serverapp_run_id(context)

        # Delete the message object
        store.delete(request.message_object_id)

        return ConfirmMessageReceivedResponse()


def _get_authenticated_serverapp_run_id(context: grpc.ServicerContext) -> int:
    """Return the authenticated run ID if it can use ServerAppIo endpoints."""
    task = get_authenticated_task()
    if task.type != TaskType.SERVER_APP:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            SERVERAPPIO_ENDPOINT_UNAVAILABLE_MESSAGE,
        )
    return task.run_id


def _raise_if(validation_error: bool, request_name: str, detail: str) -> None:
    """Raise a `ValueError` with a detailed message if a validation error occurs."""
    if validation_error:
        raise ValueError(f"Malformed {request_name}: {detail}")
