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
"""ClientAppIo API servicer."""


from logging import DEBUG, ERROR

import grpc

from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)

# pylint: disable=E0611
from flwr.proto import clientappio_pb2_grpc
from flwr.proto.appio_pb2 import (
    GetNodesRequest,
    GetNodesResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.proto.message_pb2 import (
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse
from flwr.supercore.interceptors import get_authenticated_task
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.servicer.appio import AppIoServicer
from flwr.supernode.nodestate import NodeState, NodeStateFactory


# pylint: disable=C0103,W0613,W0201
class ClientAppIoServicer(AppIoServicer, clientappio_pb2_grpc.ClientAppIoServicer):
    """ClientAppIo API servicer."""

    def __init__(
        self,
        state_factory: NodeStateFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.objectstore_factory = objectstore_factory

    def state(self) -> NodeState:
        """Return the NodeState instance."""
        return self.state_factory.state()

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "ClientAppIo.GetRun")

        # Initialize state connection
        state = self.state_factory.state()

        # Retrieve run information
        run = state.get_run(request.run_id)

        if run is None:
            return GetRunResponse()

        return GetRunResponse(run=run_to_proto(run))

    def PullTaskInput(
        self, request: PullTaskInputRequest, context: grpc.ServicerContext
    ) -> PullTaskInputResponse:
        """Pull Message, Context, and Run."""
        log(DEBUG, "ClientAppIo.PullTaskInput")

        # Get the authenticated task and associated run ID
        task = get_authenticated_task()
        run_id = task.run_id

        # Initialize state connection
        state = self.state_factory.state()

        # Retrieve run, context, and FAB for this run
        run = state.get_run(run_id)
        if run is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Run {run_id} not found in NodeState.",
            )
            raise RuntimeError("This line should never be reached.")
        series_context = state.get_run_series_context(run.series_id)
        if series_context is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"Context for RunSeries {run.series_id} not found in NodeState.",
            )
            raise RuntimeError("This line should never be reached.")

        # Retrieve FAB from NodeState
        if fab := state.get_fab(run.fab_hash):
            log(
                DEBUG,
                "Retrieved FAB: hash=%s, content_len=%d, verifications=%s",
                run.fab_hash,
                len(fab.content),
                fab.verifications,
            )
        else:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"FAB with hash {run.fab_hash} not found in NodeState.",
            )
            raise RuntimeError("This line should never be reached.")

        # Activate task
        if state.activate_task(task_id=task.task_id):
            log(DEBUG, "Started task %d of run %s", task.task_id, run_id)
            return PullTaskInputResponse(
                context=context_to_proto(series_context),
                run=run_to_proto(run),
                fab=fab_to_proto(fab),
            )

        log(ERROR, "Failed to start task %d of run %s", task.task_id, run_id)
        context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Failed to start task.")
        raise RuntimeError("Unreachable code")  # for mypy

    def PushTaskOutput(
        self, request: PushTaskOutputRequest, context: grpc.ServicerContext
    ) -> PushTaskOutputResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushTaskOutput")

        # Get the authenticated task and associated run ID
        task = get_authenticated_task()
        run_id = task.run_id

        # Initialize state connection
        state = self.state_factory.state()

        # Flag task as finished
        if state.finish_task(
            task_id=task.task_id,
            sub_status=request.sub_status,
            details=request.details,
        ):
            log(DEBUG, "Finished task %d of run %s", task.task_id, run_id)
            # Save the context to the state
            if request.HasField("context"):
                run = state.get_run(run_id)
                if run is not None:
                    state.set_run_series_context(
                        run.series_id,
                        context_from_proto(request.context),
                    )
        else:
            log(ERROR, "Failed to finish task %d of run %s", task.task_id, run_id)

        return PushTaskOutputResponse()

    def PullMessages(
        self, request: PullAppMessagesRequest, context: grpc.ServicerContext
    ) -> PullAppMessagesResponse:
        """Pull messages for ClientApp; currently returns exactly one message."""
        log(DEBUG, "ClientAppIo.PullMessages")

        # Get the authenticated task and associated run ID
        task = get_authenticated_task()
        run_id = task.run_id

        # Initialize state and store connection
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Retrieve message for this run
        messages = state.get_messages(run_ids=[run_id], is_reply=False)
        if not messages:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"No message found for run {run_id} in NodeState.",
            )
            raise RuntimeError("Unreachable code")  # for mypy
        message = messages[0]

        # Record message processing start time
        state.record_message_processing_start(message_id=message.metadata.message_id)

        # Retrieve the object tree for the message
        object_tree = store.get_object_tree(message.metadata.message_id)

        return PullAppMessagesResponse(
            messages_list=[message_to_proto(message)],
            message_object_trees=[object_tree],
        )

    def PushMessages(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push messages for ClientApp; currently accepts exactly one message."""
        log(DEBUG, "ClientAppIo.PushMessages")

        # Get the authenticated task and associated run ID
        task = get_authenticated_task()
        run_id = task.run_id

        # Initialize state connection
        state = self.state_factory.state()

        if len(request.messages_list) != 1 or len(request.message_object_trees) != 1:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "ClientAppIo.PushMessages expects exactly one message and "
                "one object tree.",
            )
            raise RuntimeError("Unreachable code")  # for mypy

        if run_id != request.messages_list[0].metadata.run_id:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Run ID in message does not match authenticated task's run ID.",
            )
            raise RuntimeError("Unreachable code")  # for mypy

        # Record message processing end time
        message = message_from_proto(request.messages_list[0])
        state.record_message_processing_end(
            message_id=message.metadata.reply_to_message_id
        )

        # Save the message to the state and preregister its objects
        session_id = state.start_session(run_id)
        _, objects_to_push = state.store_message_and_object_tree(
            message, request.message_object_trees[0], session_id
        )

        return PushAppMessagesResponse(
            objects_to_push=objects_to_push, session_id=session_id
        )

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        log(DEBUG, "ClientAppIo.GetNodes")
        context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "GetNodes is not available on ClientAppIo.",
        )
        raise RuntimeError("Unreachable code")  # for mypy

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(DEBUG, "ClientAppIoServicer.PushObject")

        # Init state
        state = self.state_factory.state()
        run_id = get_authenticated_task().run_id

        # Insert in state
        stored = state.store_object(
            run_id,
            request.session_id,
            request.object_id,
            request.object_content,
        )

        return PushObjectResponse(stored=stored)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(DEBUG, "ClientAppIoServicer.PullObject")

        # Init state
        state = self.state_factory.state()
        run_id = get_authenticated_task().run_id

        # Fetch from state
        content = state.get_object(run_id, request.object_id)
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
        log(DEBUG, "ClientAppIoServicer.ConfirmMessageReceived")

        # Init state and store
        store = self.objectstore_factory.store()

        # Delete the message object
        store.delete(request.message_object_id)

        return ConfirmMessageReceivedResponse()
