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

from typing import Annotated, cast

from fastapi import APIRouter, Depends

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

router = APIRouter(
    prefix="/v1/runtime",
    tags=["Runtime"],
    route_class=ProtobufRoute,
)

PullPendingTasksAuthDependency = Annotated[
    None,
    Depends(SuperExecAuthDependency("/flwr.proto.Runtime/PullPendingTasks")),
]
ClaimTaskAuthDependency = Annotated[
    None,
    Depends(SuperExecAuthDependency("/flwr.proto.Runtime/ClaimTask")),
]


@router.post("/pull-pending-tasks")
def pull_pending_tasks(
    request: Annotated[PullPendingTasksRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    _auth: PullPendingTasksAuthDependency,
) -> PullPendingTasksResponse:
    """Pull pending tasks."""
    return cast(PullPendingTasksResponse, handlers.pull_pending_tasks(request, state))


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
    return cast(PullTaskInputResponse, handlers.pull_task_input(request, state, task))


@router.post("/push-task-output")
def push_task_output(
    request: Annotated[PushTaskOutputRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PushTaskOutputResponse:
    """Push app process outputs."""
    return cast(PushTaskOutputResponse, handlers.push_task_output(request, state, task))


@router.post("/push-object")
def push_object(
    request: Annotated[PushObjectRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PushObjectResponse:
    """Push an object to the ObjectStore."""
    return cast(PushObjectResponse, handlers.push_object(request, state, task))


@router.post("/pull-object")
def pull_object(
    request: Annotated[PullObjectRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PullObjectResponse:
    """Pull an object from the ObjectStore."""
    return cast(PullObjectResponse, handlers.pull_object(request, state, task))


@router.post("/confirm-message-received")
def confirm_message_received(
    request: Annotated[ConfirmMessageReceivedRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> ConfirmMessageReceivedResponse:
    """Confirm message receipt."""
    return cast(
        ConfirmMessageReceivedResponse,
        handlers.confirm_message_received(request, state, task),
    )


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
    return cast(
        StartAutomationResponse, handlers.start_automation(request, state, task)
    )


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
    return cast(GetConnectorResponse, handlers.get_connector(request, state, task))


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
    return cast(PushAppMessagesResponse, handlers.push_messages(request, state, task))


@router.post("/pull-messages")
def pull_messages(
    request: Annotated[PullAppMessagesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> PullAppMessagesResponse:
    """Pull app messages."""
    return cast(PullAppMessagesResponse, handlers.pull_messages(request, state, task))


@router.post("/get-nodes")
def get_nodes(
    request: Annotated[GetNodesRequest, PROTOBUF_REQUEST_DEPENDENCY],
    state: RuntimeStateDependency,
    handlers: RuntimeHandlersDependency,
    task: TaskDependency,
) -> GetNodesResponse:
    """Get available nodes."""
    return cast(GetNodesResponse, handlers.get_nodes(request, state, task))
