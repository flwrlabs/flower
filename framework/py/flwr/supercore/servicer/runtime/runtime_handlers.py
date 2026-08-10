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
"""Shared Runtime API functions."""

# pylint: disable=unused-argument

from collections.abc import Callable
from logging import DEBUG, ERROR

import grpc

from flwr.common.constant import Status
from flwr.common.logger import log
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushTaskEventsRequest,
    PushTaskEventsResponse,
    PushTaskMessageRequest,
    PushTaskMessageResponse,
    RecordTaskUsageRequest,
    RecordTaskUsageResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import (
    TASK_TYPES_ALLOWED_TO_CREATE_TASKS,
    TASK_TYPES_REQUIRING_CONNECTOR_REF,
    TASK_TYPES_REQUIRING_FAB_HASH,
    TASK_TYPES_REQUIRING_MODEL_REF,
    TaskType,
)
from flwr.supercore.corestate import CoreState
from flwr.supercore.interceptors import get_authenticated_task
from flwr.supercore.task_process.connector import registry as connector_registry

StateProvider = Callable[[], CoreState]


def pull_pending_tasks(
    request: PullPendingTasksRequest, state_provider: StateProvider
) -> PullPendingTasksResponse:
    """Pull pending tasks."""
    log(DEBUG, "Runtime.PullPendingTasks")

    tasks = state_provider().get_tasks(
        statuses=[Status.PENDING], order_by="pending_at", ascending=True
    )
    return PullPendingTasksResponse(tasks=tasks)


def claim_task(
    request: ClaimTaskRequest, state_provider: StateProvider
) -> ClaimTaskResponse:
    """Claim a pending task."""
    log(DEBUG, "Runtime.ClaimTask")

    token = state_provider().claim_task(request.task_id)
    return ClaimTaskResponse(token=token)


def send_task_heartbeat(
    request: SendTaskHeartbeatRequest, state_provider: StateProvider
) -> SendTaskHeartbeatResponse:
    """Handle a heartbeat for a claimed task."""
    log(DEBUG, "Runtime.SendTaskHeartbeat")

    task = get_authenticated_task()
    success = state_provider().acknowledge_task_heartbeat(task.task_id)
    return SendTaskHeartbeatResponse(success=success)


def create_task(
    request: CreateTaskRequest,
    state_provider: StateProvider,
    context: grpc.ServicerContext,
) -> CreateTaskResponse:
    """Create a task."""
    log(DEBUG, "Runtime.CreateTask")

    # Get authenticated task and associated run ID
    task = get_authenticated_task()
    run_id = task.run_id

    state = state_provider()
    connector_ref = request.connector_ref or None

    _validate_create_task_request(request, task, connector_ref, state, context)
    created_task_id = state.create_task(
        task_type=request.type,
        run_id=run_id,
        fab_hash=request.fab_hash if request.HasField("fab_hash") else None,
        model_ref=request.model_ref if request.HasField("model_ref") else None,
        connector_ref=connector_ref,
        requesting_task_id=task.task_id,
    )
    if created_task_id is None:
        context.abort(grpc.StatusCode.INTERNAL, "Failed to create task")
        raise RuntimeError("This line should never be reached.")

    return CreateTaskResponse(task_id=created_task_id)


def push_task_message(
    request: PushTaskMessageRequest,
    state_provider: StateProvider,
    context: grpc.ServicerContext,
) -> PushTaskMessageResponse:
    """Push a task message."""
    log(DEBUG, "Runtime.PushTaskMessage")

    task = get_authenticated_task()

    if request.message.metadata.src_task_id != task.task_id:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "`Message.metadata.src_task_id` does not match the authenticated task.",
        )

    message = message_from_proto(request.message)

    stored = state_provider().store_task_message(message)
    if not stored:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Task message could not be stored.",
        )

    return PushTaskMessageResponse(message_id=message.metadata.message_id)


def push_task_events(
    request: PushTaskEventsRequest, state_provider: StateProvider
) -> PushTaskEventsResponse:
    """Push task events."""
    log(DEBUG, "Runtime.PushTaskEvents")

    task = get_authenticated_task()
    if not request.events:
        return PushTaskEventsResponse()

    for event in request.events:
        event.run_id = task.run_id
        event.task_id = task.task_id

    if not state_provider().store_task_events(request.events):
        log(
            ERROR,
            "Task events could not be stored for task %d of run %d.",
            task.task_id,
            task.run_id,
        )

    return PushTaskEventsResponse()


def record_task_usage(
    request: RecordTaskUsageRequest, state_provider: StateProvider
) -> RecordTaskUsageResponse:
    """Record task usage."""
    log(DEBUG, "Runtime.RecordTaskUsage")

    task = get_authenticated_task()
    state_provider().add_task_usage(task.task_id, request.task_usage)
    return RecordTaskUsageResponse()


def pull_task_message(
    request: PullTaskMessageRequest, state_provider: StateProvider
) -> PullTaskMessageResponse:
    """Pull task messages."""
    log(DEBUG, "Runtime.PullTaskMessage")

    task = get_authenticated_task()
    limit = request.limit if request.HasField("limit") else None
    messages = state_provider().get_task_message(
        dst_task_ids=[task.task_id],
        limit=limit,
        order_by="created_at",
    )
    return PullTaskMessageResponse(
        messages=[message_to_proto(message) for message in messages]
    )


def push_logs(
    request: PushLogsRequest, state_provider: StateProvider
) -> PushLogsResponse:
    """Push logs."""
    log(DEBUG, "Runtime.PushLogs")
    state = state_provider()

    task = get_authenticated_task()

    # Add logs to LinkState
    merged_logs = "".join(request.logs)
    state.add_task_log(task.task_id, merged_logs)
    return PushLogsResponse()


def _validate_create_task_request(
    request: CreateTaskRequest,
    requesting_task: Task,
    connector_ref: str | None,
    state: CoreState,
    context: grpc.ServicerContext,
) -> None:
    """Validate the task creation request."""
    if requesting_task.type not in TASK_TYPES_ALLOWED_TO_CREATE_TASKS:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            f"Task type '{requesting_task.type}' is not allowed to create tasks.",
        )

    if request.type not in set(TaskType):
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Invalid task type: {request.type}",
        )

    if request.type in TASK_TYPES_REQUIRING_FAB_HASH and not request.fab_hash:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Task type '{request.type}' requires fab_hash.",
        )

    if request.type in TASK_TYPES_REQUIRING_MODEL_REF and not request.model_ref:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Task type '{request.type}' requires model_ref.",
        )

    if request.type in TASK_TYPES_REQUIRING_CONNECTOR_REF and not connector_ref:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Task type '{request.type}' requires connector_ref.",
        )

    # Check if the connector ref is valid
    if request.type == TaskType.CONNECTOR and connector_ref:

        if connector_registry.has_builtin_connector(connector_ref):
            return

        try:
            connector_registry.get_oauth_connector_provider(connector_ref)
        except ValueError as err:
            context.abort(grpc.StatusCode.NOT_FOUND, str(err))

        available_refs = state.get_run_connector_refs(run_id=requesting_task.run_id)
        if connector_ref not in available_refs:
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Connector is not available to this run.",
            )
