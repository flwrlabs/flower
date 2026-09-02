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

from logging import DEBUG, ERROR

from flwr.common.constant import Status
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
from flwr.supercore import log
from flwr.supercore.constant import (
    TASK_TYPES_ALLOWED_TO_CREATE_TASKS,
    TASK_TYPES_REQUIRING_CONNECTOR_REF,
    TASK_TYPES_REQUIRING_FAB_HASH,
    TASK_TYPES_REQUIRING_MODEL_REF,
    TaskType,
)
from flwr.supercore.corestate import CoreState
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.runtime_timing import (
    emit_runtime_timing,
    get_runtime_task_lineage,
    is_runtime_timing_logging_enabled,
    is_runtime_timing_task,
    mark_runtime_task_first_event_persisted,
    register_runtime_task_lineage,
)
from flwr.supercore.task_process.connector import registry as connector_registry

_FINAL_RUN_EVENT_TYPES = frozenset(
    {"response.completed", "response.failed", "response.incomplete"}
)


def pull_pending_tasks(
    request: PullPendingTasksRequest, state: CoreState
) -> PullPendingTasksResponse:
    """Pull pending tasks."""
    log(DEBUG, "Runtime.PullPendingTasks")

    tasks = state.get_tasks(
        statuses=[Status.PENDING], order_by="pending_at", ascending=True
    )
    return PullPendingTasksResponse(tasks=tasks)


def claim_task(request: ClaimTaskRequest, state: CoreState) -> ClaimTaskResponse:
    """Claim a pending task."""
    log(DEBUG, "Runtime.ClaimTask")

    token = state.claim_task(request.task_id)
    if token and is_runtime_timing_logging_enabled():
        try:
            tasks = state.get_tasks(task_ids=[request.task_id])
        except Exception:  # pylint: disable=broad-exception-caught
            # Timing must not change the outcome of an already-successful claim.
            tasks = []
        if tasks and is_runtime_timing_task(tasks[0].type):
            task = tasks[0]
            lineage = get_runtime_task_lineage(run_id=task.run_id, task_id=task.task_id)
            if lineage is None:
                try:
                    lineage = state.get_task_lineage(task.task_id)
                except Exception:  # pylint: disable=broad-exception-caught
                    lineage = None
                if lineage is not None:
                    register_runtime_task_lineage(
                        run_id=task.run_id,
                        task_id=task.task_id,
                        parent_task_id=lineage[0],
                        root_task_id=lineage[1],
                    )
            parent_task_id, root_task_id = (
                lineage
                if lineage is not None
                else (None, task.task_id if task.type == TaskType.AGENT_APP else None)
            )
            emit_runtime_timing(
                "runtime.task.claimed",
                component="superlink",
                run_id=task.run_id,
                task_id=task.task_id,
                parent_task_id=parent_task_id,
                root_task_id=root_task_id,
                task_type=task.type,
            )
    return ClaimTaskResponse(token=token)


def send_task_heartbeat(
    request: SendTaskHeartbeatRequest,
    state: CoreState,
    task: Task,
) -> SendTaskHeartbeatResponse:
    """Handle a heartbeat for a claimed task."""
    log(DEBUG, "Runtime.SendTaskHeartbeat")

    success = state.acknowledge_task_heartbeat(task.task_id)
    return SendTaskHeartbeatResponse(success=success)


def create_task(
    request: CreateTaskRequest,
    state: CoreState,
    task: Task,
) -> CreateTaskResponse:
    """Create a task."""
    log(DEBUG, "Runtime.CreateTask")

    run_id = task.run_id

    connector_ref = request.connector_ref or None

    _validate_create_task_request(request, task, connector_ref, state)
    lineage = (
        (task.task_id, task.task_id)
        if (
            is_runtime_timing_logging_enabled()
            and task.type == TaskType.AGENT_APP
            and request.type == TaskType.MODEL
        )
        else None
    )
    if lineage is not None:
        created_task_id = state.create_task(
            task_type=request.type,
            run_id=run_id,
            fab_hash=request.fab_hash if request.HasField("fab_hash") else None,
            model_ref=request.model_ref if request.HasField("model_ref") else None,
            connector_ref=connector_ref,
            requesting_task_id=task.task_id,
            parent_task_id=lineage[0],
            root_task_id=lineage[1],
        )
    else:
        created_task_id = state.create_task(
            task_type=request.type,
            run_id=run_id,
            fab_hash=request.fab_hash if request.HasField("fab_hash") else None,
            model_ref=request.model_ref if request.HasField("model_ref") else None,
            connector_ref=connector_ref,
            requesting_task_id=task.task_id,
        )
    if created_task_id is None:
        raise FlowerError(
            ApiErrorCode.RUNTIME_TASK_CREATION_FAILED, "Failed to create task"
        )

    if task.type == TaskType.AGENT_APP and request.type == TaskType.MODEL:
        if lineage is not None:
            register_runtime_task_lineage(
                run_id=run_id,
                task_id=created_task_id,
                parent_task_id=lineage[0],
                root_task_id=lineage[1],
            )
        emit_runtime_timing(
            "runtime.task.queued",
            component="superlink",
            run_id=run_id,
            task_id=created_task_id,
            parent_task_id=lineage[0] if lineage is not None else None,
            root_task_id=lineage[1] if lineage is not None else None,
            task_type=request.type,
        )

    return CreateTaskResponse(task_id=created_task_id)


def push_task_message(
    request: PushTaskMessageRequest,
    state: CoreState,
    task: Task,
) -> PushTaskMessageResponse:
    """Push a task message."""
    log(DEBUG, "Runtime.PushTaskMessage")

    if request.message.metadata.src_task_id != task.task_id:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_MESSAGE,
            "`Message.metadata.src_task_id` does not match the authenticated task.",
        )

    message = message_from_proto(request.message)

    stored = state.store_task_message(message)
    if not stored:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_MESSAGE,
            "Task message could not be stored.",
        )

    if is_runtime_timing_logging_enabled():
        try:
            destination_tasks = state.get_tasks(
                task_ids=(
                    [message.metadata.dst_task_id]
                    if message.metadata.dst_task_id is not None
                    else []
                )
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # The message is already stored. Skip optional lineage on lookup failure.
            destination_tasks = []
        if (
            task.type == TaskType.AGENT_APP
            and destination_tasks
            and destination_tasks[0].type == TaskType.MODEL
        ):
            register_runtime_task_lineage(
                run_id=task.run_id,
                task_id=destination_tasks[0].task_id,
                parent_task_id=task.task_id,
                root_task_id=task.task_id,
            )
            emit_runtime_timing(
                "runtime.agent.model.dispatch.accepted",
                component="superlink",
                run_id=task.run_id,
                task_id=destination_tasks[0].task_id,
                parent_task_id=task.task_id,
                root_task_id=task.task_id,
                task_type=TaskType.MODEL,
            )

    return PushTaskMessageResponse(message_id=message.metadata.message_id)


def push_task_events(
    request: PushTaskEventsRequest,
    state: CoreState,
    task: Task,
) -> PushTaskEventsResponse:
    """Push task events."""
    log(DEBUG, "Runtime.PushTaskEvents")

    if not request.events:
        return PushTaskEventsResponse()

    for event in request.events:
        event.run_id = task.run_id
        event.task_id = task.task_id

    timing_enabled = is_runtime_timing_logging_enabled() and is_runtime_timing_task(
        task.type
    )
    has_persisted_events = True
    if timing_enabled:
        try:
            has_persisted_events = state.has_task_events(task_id=task.task_id)
        except Exception:  # pylint: disable=broad-exception-caught
            # Timing must not change event persistence semantics.
            pass

    try:
        stored = state.store_task_events(request.events)
    except Exception:  # pylint: disable=broad-exception-caught
        if timing_enabled:
            lineage = _get_runtime_task_lineage(state, task)
            parent_task_id, root_task_id = (
                lineage
                if lineage is not None
                else (None, task.task_id if task.type == TaskType.AGENT_APP else None)
            )
            emit_runtime_timing(
                "runtime.events.publish.failed",
                component="superlink",
                run_id=task.run_id,
                task_id=task.task_id,
                parent_task_id=parent_task_id,
                root_task_id=root_task_id,
                task_type=task.type,
                outcome="error",
                error_kind="state",
            )
        raise

    if not stored:
        log(
            ERROR,
            "Task events could not be stored for task %d of run %d.",
            task.task_id,
            task.run_id,
        )
        if timing_enabled:
            lineage = _get_runtime_task_lineage(state, task)
            parent_task_id, root_task_id = (
                lineage
                if lineage is not None
                else (None, task.task_id if task.type == TaskType.AGENT_APP else None)
            )
            emit_runtime_timing(
                "runtime.events.publish.failed",
                component="superlink",
                run_id=task.run_id,
                task_id=task.task_id,
                parent_task_id=parent_task_id,
                root_task_id=root_task_id,
                task_type=task.type,
                outcome="error",
                error_kind="state",
            )
        return PushTaskEventsResponse()

    if timing_enabled:
        lineage = _get_runtime_task_lineage(state, task)
        parent_task_id, root_task_id = (
            lineage
            if lineage is not None
            else (None, task.task_id if task.type == TaskType.AGENT_APP else None)
        )
        if not has_persisted_events and mark_runtime_task_first_event_persisted(
            run_id=task.run_id,
            task_id=task.task_id,
        ):
            emit_runtime_timing(
                "runtime.events.first.persisted",
                component="superlink",
                run_id=task.run_id,
                task_id=task.task_id,
                parent_task_id=parent_task_id,
                root_task_id=root_task_id,
                task_type=task.type,
            )
            if task.type == TaskType.MODEL and lineage is not None:
                # AgentApp does not consume Model stream events directly. This is the
                # nearest content-free relay boundary for its first Model event.
                emit_runtime_timing(
                    "runtime.agent.model.first_event.received",
                    component="superlink",
                    run_id=task.run_id,
                    task_id=task.task_id,
                    parent_task_id=parent_task_id,
                    root_task_id=root_task_id,
                    task_type=task.type,
                )
        if any(event.event in _FINAL_RUN_EVENT_TYPES for event in request.events):
            emit_runtime_timing(
                "runtime.events.final.persisted",
                component="superlink",
                run_id=task.run_id,
                task_id=task.task_id,
                parent_task_id=parent_task_id,
                root_task_id=root_task_id,
                task_type=task.type,
            )

    return PushTaskEventsResponse()


def record_task_usage(
    request: RecordTaskUsageRequest,
    state: CoreState,
    task: Task,
) -> RecordTaskUsageResponse:
    """Record task usage."""
    log(DEBUG, "Runtime.RecordTaskUsage")

    state.add_task_usage(task.task_id, request.task_usage)
    return RecordTaskUsageResponse()


def pull_task_message(
    request: PullTaskMessageRequest,
    state: CoreState,
    task: Task,
) -> PullTaskMessageResponse:
    """Pull task messages."""
    log(DEBUG, "Runtime.PullTaskMessage")

    limit = request.limit if request.HasField("limit") else None
    src_task_ids = [request.src_task_id] if request.HasField("src_task_id") else None
    messages = state.get_task_message(
        dst_task_ids=[task.task_id],
        src_task_ids=src_task_ids,
        limit=limit,
        order_by="created_at",
    )
    return PullTaskMessageResponse(
        messages=[message_to_proto(message) for message in messages]
    )


def push_logs(
    request: PushLogsRequest,
    state: CoreState,
    task: Task,
) -> PushLogsResponse:
    """Push logs."""
    log(DEBUG, "Runtime.PushLogs")
    # Add logs to LinkState
    merged_logs = "".join(request.logs)
    state.add_task_log(task.task_id, merged_logs)
    return PushLogsResponse()


def _get_runtime_task_lineage(state: CoreState, task: Task) -> tuple[int, int] | None:
    """Return cached or durable lineage without affecting runtime behavior."""
    lineage = get_runtime_task_lineage(run_id=task.run_id, task_id=task.task_id)
    if lineage is not None:
        return lineage
    try:
        lineage = state.get_task_lineage(task.task_id)
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    if lineage is not None:
        register_runtime_task_lineage(
            run_id=task.run_id,
            task_id=task.task_id,
            parent_task_id=lineage[0],
            root_task_id=lineage[1],
        )
    return lineage


def _validate_create_task_request(
    request: CreateTaskRequest,
    requesting_task: Task,
    connector_ref: str | None,
    state: CoreState,
) -> None:
    """Validate the task creation request."""
    if requesting_task.type not in TASK_TYPES_ALLOWED_TO_CREATE_TASKS:
        raise FlowerError(
            ApiErrorCode.RUNTIME_TASK_CREATION_NOT_ALLOWED,
            f"Task type '{requesting_task.type}' is not allowed to create tasks.",
        )

    if request.type not in set(TaskType):
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Invalid task type: {request.type}",
        )

    if request.type in TASK_TYPES_REQUIRING_FAB_HASH and not request.fab_hash:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Task type '{request.type}' requires fab_hash.",
        )

    if request.type in TASK_TYPES_REQUIRING_MODEL_REF and not request.model_ref:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Task type '{request.type}' requires model_ref.",
        )

    if request.type in TASK_TYPES_REQUIRING_CONNECTOR_REF and not connector_ref:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Task type '{request.type}' requires connector_ref.",
        )

    # Check if the connector ref is valid
    if request.type == TaskType.CONNECTOR and connector_ref:

        if connector_registry.has_builtin_connector(connector_ref):
            return

        try:
            connector_registry.get_oauth_flow(connector_ref)
        except ValueError as err:
            raise FlowerError(ApiErrorCode.CONNECTOR_NOT_FOUND, str(err)) from err

        available_refs = state.get_run_connector_refs(run_id=requesting_task.run_id)
        if connector_ref not in available_refs:
            raise FlowerError(
                ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE,
                "Connector is not available to this run.",
            )
