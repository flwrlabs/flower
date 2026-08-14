# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""SuperNode Runtime API functions."""

# pylint: disable=unused-argument

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
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
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
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
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
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supernode.nodestate import NodeState


def get_run(request: GetRunRequest, state: NodeState) -> GetRunResponse:
    """Get run information."""
    log(DEBUG, "Runtime.GetRun")

    # Retrieve run information
    run = state.get_run(request.run_id)

    if run is None:
        return GetRunResponse()

    return GetRunResponse(run=run_to_proto(run))


def pull_task_input(
    request: PullTaskInputRequest,
    state: NodeState,
    task: Task,
    context: grpc.ServicerContext,
) -> PullTaskInputResponse:
    """Pull Message, Context, and Run."""
    log(DEBUG, "Runtime.PullTaskInput")

    run_id = task.run_id

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


def push_task_output(
    request: PushTaskOutputRequest,
    state: NodeState,
    task: Task,
) -> PushTaskOutputResponse:
    """Push Message and Context."""
    log(DEBUG, "Runtime.PushTaskOutput")

    run_id = task.run_id

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


def pull_messages(
    request: PullAppMessagesRequest,
    state: NodeState,
    task: Task,
) -> PullAppMessagesResponse:
    """Pull messages for ClientApp; currently returns exactly one message."""
    log(DEBUG, "Runtime.PullMessages")

    run_id = task.run_id

    # Retrieve message for this run
    messages = state.get_messages(run_ids=[run_id], is_reply=False, limit=1)
    if not messages:
        return PullAppMessagesResponse()
    message = messages[0]

    # Record message processing start time
    state.record_message_processing_start(message_id=message.metadata.message_id)

    # Retrieve the object tree for the message
    object_tree = state.object_store.get_object_tree(message.metadata.message_id)

    return PullAppMessagesResponse(
        messages_list=[message_to_proto(message)],
        message_object_trees=[object_tree],
    )


def push_messages(
    request: PushAppMessagesRequest,
    state: NodeState,
    task: Task,
    context: grpc.ServicerContext,
) -> PushAppMessagesResponse:
    """Push messages for ClientApp; currently accepts exactly one message."""
    log(DEBUG, "Runtime.PushMessages")

    run_id = task.run_id

    if len(request.messages_list) != 1 or len(request.message_object_trees) != 1:
        context.abort(
            grpc.StatusCode.INVALID_ARGUMENT,
            "Runtime.PushMessages expects exactly one message and one object tree.",
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
    state.record_message_processing_end(message_id=message.metadata.reply_to_message_id)

    # Save the message to the state and preregister its objects
    session_id = state.start_session(run_id)
    _, objects_to_push = state.store_message_and_object_tree(
        message, request.message_object_trees[0], session_id
    )

    return PushAppMessagesResponse(
        objects_to_push=objects_to_push, session_id=session_id
    )


def get_nodes(
    request: GetNodesRequest, context: grpc.ServicerContext
) -> GetNodesResponse:
    """Get available nodes."""
    log(DEBUG, "Runtime.GetNodes")
    context.abort(
        grpc.StatusCode.PERMISSION_DENIED,
        "This endpoint is only available to ServerApp tasks.",
    )
    raise RuntimeError("Unreachable code")  # for mypy


def start_automation(
    request: StartAutomationRequest,
    context: grpc.ServicerContext,
) -> StartAutomationResponse:
    """Reject automation requests from ClientApp tasks."""
    log(DEBUG, "Runtime.StartAutomation")
    context.abort(
        grpc.StatusCode.PERMISSION_DENIED,
        "Only AgentApp and ServerApp tasks can create automations.",
    )
    raise RuntimeError("Unreachable code")  # for mypy


def get_connector(
    request: GetConnectorRequest,
    context: grpc.ServicerContext,
) -> GetConnectorResponse:
    """Reject connector credential requests from ClientApp tasks."""
    log(DEBUG, "Runtime.GetConnector")
    context.abort(
        grpc.StatusCode.PERMISSION_DENIED,
        "Connector credentials are not available to this task.",
    )
    raise RuntimeError("Unreachable code")  # for mypy


def push_object(
    request: PushObjectRequest,
    state: NodeState,
    task: Task,
) -> PushObjectResponse:
    """Push an object to the ObjectStore."""
    log(DEBUG, "Runtime.PushObject")

    # Insert in state
    stored = state.store_object(
        task.run_id,
        request.session_id,
        request.object_id,
        request.object_content,
    )

    return PushObjectResponse(stored=stored)


def pull_object(
    request: PullObjectRequest,
    state: NodeState,
    task: Task,
) -> PullObjectResponse:
    """Pull an object from the ObjectStore."""
    log(DEBUG, "Runtime.PullObject")

    # Fetch from state
    content = state.get_object(task.run_id, request.object_id)
    if content is not None:
        object_available = content != b""
        return PullObjectResponse(
            object_found=True,
            object_available=object_available,
            object_content=content,
        )
    return PullObjectResponse(object_found=False, object_available=False)


def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
    state: NodeState,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received."""
    log(DEBUG, "Runtime.ConfirmMessageReceived")

    # Delete the message object
    state.object_store.delete(request.message_object_id)

    return ConfirmMessageReceivedResponse()
