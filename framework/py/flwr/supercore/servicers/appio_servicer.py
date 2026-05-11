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
"""Shared AppIo API servicer."""


from abc import ABC, abstractmethod
from logging import DEBUG

import grpc

from flwr.common.constant import Status
from flwr.common.logger import log
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    CreateTaskRequest,
    CreateTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.supercore.constant import (
    TASK_TYPES_REQUIRING_CONNECTOR_REF,
    TASK_TYPES_REQUIRING_FAB_HASH,
    TASK_TYPES_REQUIRING_MODEL_REF,
    TaskType,
)
from flwr.supercore.interceptors import get_authenticated_task

from ..corestate import CoreState


# pylint: disable=invalid-name, unused-argument
class AppIoServicer(ABC):
    """Shared scaffolding for task-based AppIo RPCs."""

    @abstractmethod
    def state(self) -> CoreState:
        """Return the CoreState instance."""

    def PullPendingTasks(
        self, request: PullPendingTasksRequest, context: grpc.ServicerContext
    ) -> PullPendingTasksResponse:
        """Pull pending tasks."""
        log(DEBUG, "AppIoServicer.PullPendingTasks")

        tasks = self.state().get_tasks(
            statuses=[Status.PENDING], order_by="pending_at", ascending=True
        )
        return PullPendingTasksResponse(tasks=tasks)

    def ClaimTask(
        self, request: ClaimTaskRequest, context: grpc.ServicerContext
    ) -> ClaimTaskResponse:
        """Claim a pending task."""
        log(DEBUG, "AppIoServicer.ClaimTask")

        token = self.state().claim_task(request.task_id)
        return ClaimTaskResponse(token=token)

    def SendTaskHeartbeat(
        self, request: SendTaskHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendTaskHeartbeatResponse:
        """Handle a heartbeat for a claimed task."""
        log(DEBUG, "AppIoServicer.SendTaskHeartbeat")

        task = get_authenticated_task()
        success = self.state().acknowledge_task_heartbeat(task.task_id)
        return SendTaskHeartbeatResponse(success=success)

    def CreateTask(
        self, request: CreateTaskRequest, context: grpc.ServicerContext
    ) -> CreateTaskResponse:
        """Create a task."""
        log(DEBUG, "AppIoServicer.CreateTask")

        run_id = get_authenticated_task().run_id

        _validate_create_task_request(request, context)

        state = self.state()
        created_task_id = state.create_task(
            task_type=request.type,
            run_id=run_id,
            fab_hash=request.fab_hash if request.HasField("fab_hash") else None,
            model_ref=request.model_ref if request.HasField("model_ref") else None,
            connector_ref=(
                request.connector_ref if request.HasField("connector_ref") else None
            ),
        )
        if created_task_id is None:
            context.abort(grpc.StatusCode.INTERNAL, "Failed to create task")
            raise RuntimeError("This line should never be reached.")

        return CreateTaskResponse(task_id=created_task_id)

    def PushLogs(
        self, request: PushLogsRequest, context: grpc.ServicerContext
    ) -> PushLogsResponse:
        """Push logs."""
        log(DEBUG, "AppIoServicer.PushLogs")
        state = self.state()

        task = get_authenticated_task()

        # Add logs to LinkState
        merged_logs = "".join(request.logs)
        state.add_task_log(task.task_id, merged_logs)
        return PushLogsResponse()


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
