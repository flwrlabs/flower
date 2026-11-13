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
from uuid import UUID

from flwr.common.constant import Status
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    AcquireTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    DeleteFileRequest,
    DeleteFileResponse,
    GetCompactedContextsRequest,
    GetCompactedContextsResponse,
    GetFileRequest,
    GetFileResponse,
    GetSpaceEventsRequest,
    GetSpaceEventsResponse,
    ListFilesRequest,
    ListFilesResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushTaskEventsRequest,
    PushTaskEventsResponse,
    PushTaskMessageRequest,
    PushTaskMessageResponse,
    PutCompactedContextRequest,
    PutCompactedContextResponse,
    PutFileRequest,
    PutFileResponse,
    RecordTaskUsageRequest,
    RecordTaskUsageResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    GetSpaceMessagesRequest,
    GetSpaceMessagesResponse,
    SendSpaceMessageRequest,
    SendSpaceMessageResponse,
    SpaceParticipant,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.constant import (
    TASK_TYPE_TO_COMMAND,
    TASK_TYPES_ALLOWED_TO_CREATE_TASKS,
    TASK_TYPES_REQUIRING_CONNECTOR_REF,
    TASK_TYPES_REQUIRING_FAB_HASH,
    TaskType,
)
from flwr.supercore.corestate import CoreState
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.run import Run
from flwr.supercore.task_process.connector import registry as connector_registry


def get_space_events(
    request: GetSpaceEventsRequest, state: CoreState, space_id: UUID
) -> GetSpaceEventsResponse:
    """Get locally stored events for a Space."""
    log(DEBUG, "Runtime.GetSpaceEvents")
    events = state.get_task_events(
        space_id=space_id,
        event_types=request.event_types or None,
        after_task_event_id=(
            request.after_event_id if request.HasField("after_event_id") else None
        ),
        before_task_event_id=(
            request.before_event_id if request.HasField("before_event_id") else None
        ),
        last_n=request.last_n or None,
    )
    return GetSpaceEventsResponse(events=events)


def require_agent_id(run: Run | None, run_id: int) -> int:
    """Return the agent ID of a Run, or reject a missing or unbound Run."""
    if run is None:
        raise FlowerError(ApiErrorCode.RUN_ID_NOT_FOUND, f"Run {run_id} not found.")
    if run.agent_id is None:
        raise FlowerError(
            ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE,
            "This endpoint requires a Run bound to an agent.",
        )
    return run.agent_id


def authenticate_space_agent(task: Task, run: Run) -> tuple[UUID, SpaceParticipant]:
    """Require the primary AgentApp task of a Run bound to a Space and agent."""
    if run.space_id is None or run.agent_id is None:
        raise FlowerError(
            ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE,
            "This endpoint requires a Run bound to a Space and an agent.",
        )
    if task.type != TaskType.AGENT_APP or task.task_id != run.primary_task_id:
        raise FlowerError(
            ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE,
            "Only the primary AgentApp task can access this endpoint.",
        )
    return run.space_id, SpaceParticipant(agent_id=run.agent_id)


def get_space_messages(
    request: GetSpaceMessagesRequest, state: CoreState, task: Task, run: Run
) -> GetSpaceMessagesResponse:
    """Read messages within the authenticated task's Space grant."""
    space_id, participant = authenticate_space_agent(task, run)
    limit = request.limit if request.HasField("limit") else 100
    if limit > 100:
        raise FlowerError(
            ApiErrorCode.INVALID_SPACE_REQUEST, "limit must be at most 100."
        )
    try:
        messages = state.get_space_messages(
            space_id,
            participant,
            limit=limit,
            before_id=request.before_id if request.HasField("before_id") else None,
            after_id=request.after_id if request.HasField("after_id") else None,
            unread_only=request.unread_only,
        )
    except ValueError as exc:
        raise FlowerError(ApiErrorCode.INVALID_SPACE_REQUEST, str(exc)) from exc
    if messages is None:
        raise FlowerError(ApiErrorCode.SPACE_NOT_FOUND, "Space not found.")
    return GetSpaceMessagesResponse(messages=messages)


def send_space_message(
    request: SendSpaceMessageRequest, state: CoreState, task: Task, run: Run
) -> SendSpaceMessageResponse:
    """Publish with the primary AgentApp's server-bound agent identity."""
    space_id, participant = authenticate_space_agent(task, run)
    try:
        message = state.create_space_message(space_id, participant, request.content)
    except ValueError as exc:
        raise FlowerError(ApiErrorCode.INVALID_SPACE_REQUEST, str(exc)) from exc
    if message is None:
        raise FlowerError(ApiErrorCode.SPACE_NOT_FOUND, "Space participant not found.")
    return SendSpaceMessageResponse(message=message)


def get_compacted_contexts(
    request: GetCompactedContextsRequest, state: CoreState, agent_id: int
) -> GetCompactedContextsResponse:
    """Get compacted contexts for the agent."""
    limit = request.limit if request.HasField("limit") else 1
    if limit == 0:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_COMPACTED_CONTEXT_REQUEST,
            "Limit must be positive.",
        )
    contexts = state.get_compacted_contexts(
        agent_id,
        limit=limit,
        before_context_id=(
            request.before_context_id if request.HasField("before_context_id") else None
        ),
    )
    return GetCompactedContextsResponse(contexts=contexts)


def put_compacted_context(
    request: PutCompactedContextRequest, state: CoreState, agent_id: int
) -> PutCompactedContextResponse:
    """Store a compacted context for the agent."""
    if not request.HasField("context") or not request.context.text:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_COMPACTED_CONTEXT_REQUEST,
            "Compacted context text is required.",
        )
    context = request.context
    if context.agent_id and context.agent_id != agent_id:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_COMPACTED_CONTEXT_REQUEST,
            "Compacted context agent does not match the authenticated task.",
        )
    try:
        state.store_compacted_context(
            agent_id,
            context.through_event_id,
            context.text,
            context.token_count,
        )
    except ValueError as exc:
        raise FlowerError(
            ApiErrorCode.RUNTIME_COMPACTED_CONTEXT_ALREADY_EXISTS,
            f"Compacted context already exists for agent {agent_id} "
            f"and event {context.through_event_id}.",
        ) from exc
    return PutCompactedContextResponse()


def acquire_task(request: AcquireTaskRequest, state: CoreState) -> AcquireTaskResponse:
    """Claim the oldest pending task matching the executor's available capacity."""
    log(DEBUG, "Runtime.AcquireTask")
    supported_types = set(request.supported_task_types)
    agentapp_fab_hashes = set(request.agentapp_fab_hashes)
    if not supported_types and not agentapp_fab_hashes:
        return AcquireTaskResponse()

    tasks = state.get_tasks(
        statuses=[Status.PENDING], order_by="pending_at", ascending=True
    )
    for task in tasks:
        eligible = task.type in supported_types or task.fab_hash in agentapp_fab_hashes
        if eligible and (token := state.claim_task(task.task_id)):
            return AcquireTaskResponse(task=task, token=token)
    return AcquireTaskResponse()


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

    _validate_create_task_request(request, task, connector_ref)
    connector_id = None
    if request.type == TaskType.CONNECTOR and connector_ref:
        connector_id = _resolve_connector_id(task, connector_ref, state)
    created_task_id = state.create_task(
        task_type=request.type,
        run_id=run_id,
        fab_hash=request.fab_hash if request.HasField("fab_hash") else None,
        model_ref=request.model_ref if request.HasField("model_ref") else None,
        connector_ref=connector_ref,
        connector_id=connector_id,
        requesting_task_id=task.task_id,
    )
    if created_task_id is None:
        raise FlowerError(
            ApiErrorCode.RUNTIME_TASK_CREATION_FAILED, "Failed to create task"
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

    if not state.store_task_events(request.events):
        log(
            ERROR,
            "Task events could not be stored for task %d of run %d.",
            task.task_id,
            task.run_id,
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


def put_file(
    request: PutFileRequest, state: CoreState, agent_id: int
) -> PutFileResponse:
    """Store a file in the authenticated task's agent scope."""
    log(DEBUG, "Runtime.PutFile")
    if not request.key.strip():
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_FILE_KEY,
            "File key must not be empty or whitespace-only.",
        )
    revision = state.put_file(
        scope=_file_scope(agent_id),
        key=request.key,
        data=request.data,
        expected_revision=(
            request.expected_revision if request.HasField("expected_revision") else None
        ),
    )
    return PutFileResponse(revision=revision)


def get_file(
    request: GetFileRequest, state: CoreState, agent_id: int
) -> GetFileResponse:
    """Get a file from the authenticated task's agent scope."""
    log(DEBUG, "Runtime.GetFile")
    return GetFileResponse(
        file=state.get_file(scope=_file_scope(agent_id), key=request.key)
    )


def list_files(
    request: ListFilesRequest, state: CoreState, agent_id: int
) -> ListFilesResponse:
    """List file metadata in the authenticated task's agent scope.

    When a limit is set, fetch one extra entry to check for another page.
    Return the last included key as the cursor only if more files remain.
    """
    log(DEBUG, "Runtime.ListFiles")
    limit = request.limit if request.HasField("limit") else None
    files = state.list_files(
        scope=_file_scope(agent_id),
        prefix=request.prefix,
        start_after=request.start_after,
        limit=limit + 1 if limit else limit,
    )
    return ListFilesResponse(
        files=files[:limit],
        next_start_after=files[limit - 1].key if limit and len(files) > limit else "",
    )


def delete_file(
    request: DeleteFileRequest, state: CoreState, agent_id: int
) -> DeleteFileResponse:
    """Delete a file in the authenticated task's agent scope."""
    log(DEBUG, "Runtime.DeleteFile")
    state.delete_file(
        scope=_file_scope(agent_id),
        key=request.key,
        expected_revision=request.expected_revision,
    )
    return DeleteFileResponse()


def _file_scope(agent_id: int) -> str:
    """Return the file namespace for an agent."""
    return f"agent-{agent_id}"


def _validate_create_task_request(
    request: CreateTaskRequest,
    requesting_task: Task,
    connector_ref: str | None,
) -> None:
    """Validate the task creation request."""
    if requesting_task.type not in TASK_TYPES_ALLOWED_TO_CREATE_TASKS:
        raise FlowerError(
            ApiErrorCode.RUNTIME_TASK_CREATION_NOT_ALLOWED,
            f"Task type '{requesting_task.type}' is not allowed to create tasks.",
        )

    if request.type not in TASK_TYPE_TO_COMMAND:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Invalid task type: {request.type}",
        )

    if request.type in TASK_TYPES_REQUIRING_FAB_HASH and not request.fab_hash:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Task type '{request.type}' requires fab_hash.",
        )

    if request.type in TASK_TYPES_REQUIRING_CONNECTOR_REF and not connector_ref:
        raise FlowerError(
            ApiErrorCode.RUNTIME_INVALID_TASK_CREATION_REQUEST,
            f"Task type '{request.type}' requires connector_ref.",
        )

    # Check if the connector ref is valid.
    if request.type == TaskType.CONNECTOR and connector_ref:
        if connector_registry.has_builtin_connector(connector_ref):
            return

        try:
            connector_registry.get_connector_tools(connector_ref)
        except ValueError as err:
            raise FlowerError(ApiErrorCode.CONNECTOR_NOT_FOUND, str(err)) from err


def _resolve_connector_id(
    requesting_task: Task,
    connector_ref: str,
    state: CoreState,
) -> int | None:
    """Resolve the connector ID for a connector task."""
    if connector_registry.has_builtin_connector(connector_ref):
        return None

    matching_ids = [
        connector_id
        for connector_id in state.get_run_connector_ids(requesting_task.run_id)
        if (connector := state.get_connector_by_id(connector_id)) is not None
        and connector.connector_ref == connector_ref
    ]
    if len(matching_ids) != 1:
        raise FlowerError(
            ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE,
            "Connector is not available to this run.",
        )
    return matching_ids[0]
