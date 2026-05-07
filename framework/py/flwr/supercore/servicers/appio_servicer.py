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
    PullPendingTasksRequest,
    PullPendingTasksResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
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
