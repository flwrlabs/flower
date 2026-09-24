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
"""Shared Runtime API router."""

import asyncio
from collections.abc import Callable
from time import monotonic
from typing import Annotated, TypeVar

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
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
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetRunSeriesEventsRequest,
    GetRunSeriesEventsResponse,
    PullAndClaimTaskRequest,
    PullAndClaimTaskResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskEventsRequest,
    PushTaskEventsResponse,
    PushTaskMessageRequest,
    PushTaskMessageResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
    RecordTaskUsageRequest,
    RecordTaskUsageResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.supercore.dependencies.runtime import (
    RuntimeHandlersDependency,
    RuntimeStateDependency,
    SuperExecAuthDependency,
    TaskDependency,
)
from flwr.supercore.protobuf.routing import ProtobufRoute
from flwr.supercore.protobuf.translation import PROTOBUF_REQUEST_DEPENDENCY
from flwr.supercore.servicer.runtime import runtime_handlers as core_runtime_handlers
from flwr.supercore.task_notification import subscribe_to_task_notifications

router = APIRouter(
    prefix="/v1/runtime",
    tags=["Runtime"],
    route_class=ProtobufRoute,
)

PullPendingTasksAuthDependency = Annotated[
    None,
    Depends(SuperExecAuthDependency("/flwr.proto.Runtime/PullPendingTasks")),
]
PullAndClaimTaskAuthDependency = Annotated[
    None,
    Depends(SuperExecAuthDependency("/flwr.proto.Runtime/PullAndClaimTask")),
]
ClaimTaskAuthDependency = Annotated[
    None,
    Depends(SuperExecAuthDependency("/flwr.proto.Runtime/ClaimTask")),
]

_MAX_TASK_WAIT_MS = 5_000
_TASK_RECHECK_SECONDS = 0.2
ResponseT = TypeVar("ResponseT")


async def _wait_for_task(
    wait_timeout_ms: int,
    pull: Callable[[], ResponseT],
    has_task: Callable[[ResponseT], bool],
) -> ResponseT:
    """Recheck shared state until work appears or the bounded wait expires.

    Local notifications wake the request promptly. Periodic reads also see tasks
    created by other processes and dispatch due SuperLink automations. No database
    transaction is held between reads.
    """
    deadline = monotonic() + min(wait_timeout_ms, _MAX_TASK_WAIT_MS) / 1_000
    with subscribe_to_task_notifications() as task_event:
        while True:
            task_event.clear()
            response = await run_in_threadpool(pull)
            remaining = deadline - monotonic()
            if has_task(response) or remaining <= 0:
                return response
            try:
                await asyncio.wait_for(
                    task_event.wait(), min(_TASK_RECHECK_SECONDS, remaining)
                )
            except TimeoutError:
                pass


@router.post("/pull-pending-tasks")
async def pull_pending_tasks(
    request: Annotated[PullPendingTasksRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    _auth: PullPendingTasksAuthDependency,
) -> PullPendingTasksResponse:
    """Pull pending tasks."""
    return await _wait_for_task(
        request.wait_timeout_ms,
        lambda: handlers.pull_pending_tasks(request, state),
        lambda response: bool(response.tasks),
    )


@router.post("/pull-and-claim-task")
async def pull_and_claim_task(
    request: Annotated[PullAndClaimTaskRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    _auth: PullAndClaimTaskAuthDependency,
) -> PullAndClaimTaskResponse:
    """Pull and claim the oldest supported pending task."""
    return await _wait_for_task(
        request.wait_timeout_ms if request.supported_task_types else 0,
        lambda: handlers.pull_and_claim_task(request, state),
        lambda response: response.HasField("task") and bool(response.token),
    )


@router.post("/claim-task")
def claim_task(
    request: Annotated[ClaimTaskRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    _auth: ClaimTaskAuthDependency,
) -> ClaimTaskResponse:
    """Claim a pending task."""
    return core_runtime_handlers.claim_task(request, state)


@router.post("/send-task-heartbeat")
def send_task_heartbeat(
    request: Annotated[SendTaskHeartbeatRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> SendTaskHeartbeatResponse:
    """Handle a heartbeat for a claimed task."""
    return core_runtime_handlers.send_task_heartbeat(request, state, task)


@router.post("/pull-task-input")
def pull_task_input(
    request: Annotated[PullTaskInputRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PullTaskInputResponse:
    """Pull app process inputs."""
    return handlers.pull_task_input(request, state, task)


@router.post("/push-task-output")
def push_task_output(
    request: Annotated[PushTaskOutputRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PushTaskOutputResponse:
    """Push app process outputs."""
    return handlers.push_task_output(request, state, task)


@router.post("/push-object")
def push_object(
    request: Annotated[PushObjectRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PushObjectResponse:
    """Push an object to the ObjectStore."""
    return handlers.push_object(request, state, task)


@router.post("/pull-object")
def pull_object(
    request: Annotated[PullObjectRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PullObjectResponse:
    """Pull an object from the ObjectStore."""
    return handlers.pull_object(request, state, task)


@router.post("/confirm-message-received")
def confirm_message_received(
    request: Annotated[ConfirmMessageReceivedRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> ConfirmMessageReceivedResponse:
    """Confirm message receipt."""
    return handlers.confirm_message_received(request, state, task)


@router.post("/create-task")
def create_task(
    request: Annotated[CreateTaskRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> CreateTaskResponse:
    """Create a task."""
    return core_runtime_handlers.create_task(request, state, task)


@router.post("/start-automation")
def runtime_start_automation(
    request: Annotated[StartAutomationRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> StartAutomationResponse:
    """Start an automation from a Runtime task."""
    return handlers.start_automation(request, state, task)


@router.post("/push-task-message")
def push_task_message(
    request: Annotated[PushTaskMessageRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> PushTaskMessageResponse:
    """Push a task message."""
    return core_runtime_handlers.push_task_message(request, state, task)


@router.post("/push-task-events")
def push_task_events(
    request: Annotated[PushTaskEventsRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> PushTaskEventsResponse:
    """Push task events."""
    return core_runtime_handlers.push_task_events(request, state, task)


@router.post("/get-run-series-events")
def get_run_series_events(
    request: Annotated[GetRunSeriesEventsRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> GetRunSeriesEventsResponse:
    """Get events from the authenticated task's run series."""
    return handlers.get_run_series_events(request, state, task)


@router.post("/pull-task-message")
def pull_task_message(
    request: Annotated[PullTaskMessageRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> PullTaskMessageResponse:
    """Pull task messages."""
    return core_runtime_handlers.pull_task_message(request, state, task)


@router.post("/record-task-usage")
def record_task_usage(
    request: Annotated[RecordTaskUsageRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> RecordTaskUsageResponse:
    """Record task usage."""
    return core_runtime_handlers.record_task_usage(request, state, task)


@router.post("/get-connector")
def get_connector(
    request: Annotated[GetConnectorRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> GetConnectorResponse:
    """Get connector credentials."""
    return handlers.get_connector(request, state, task)


@router.post("/push-logs")
def push_logs(
    request: Annotated[PushLogsRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    task: TaskDependency,
) -> PushLogsResponse:
    """Push task logs."""
    return core_runtime_handlers.push_logs(request, state, task)


@router.post("/push-messages")
def push_messages(
    request: Annotated[PushAppMessagesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PushAppMessagesResponse:
    """Push app messages."""
    return handlers.push_messages(request, state, task)


@router.post("/pull-messages")
def pull_messages(
    request: Annotated[PullAppMessagesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PullAppMessagesResponse:
    """Pull app messages."""
    return handlers.pull_messages(request, state, task)


@router.post("/get-nodes")
def get_nodes(
    request: Annotated[GetNodesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> GetNodesResponse:
    """Get available nodes."""
    return handlers.get_nodes(request, state, task)
