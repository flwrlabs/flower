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
"""Shared Runtime API servicer implementation."""

from abc import ABC, abstractmethod

import grpc

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
from flwr.supercore.corestate import CoreState
from flwr.supercore.interceptors import get_authenticated_task

from . import runtime_handlers


# pylint: disable=invalid-name, unused-argument
class RuntimeServicer(ABC):
    """Shared scaffolding for task-based Runtime RPCs."""

    @abstractmethod
    def state(self) -> CoreState:
        """Return the CoreState instance."""

    def PullPendingTasks(
        self, request: PullPendingTasksRequest, context: grpc.ServicerContext
    ) -> PullPendingTasksResponse:
        """Pull pending tasks."""
        return runtime_handlers.pull_pending_tasks(request, self.state())

    def ClaimTask(
        self, request: ClaimTaskRequest, context: grpc.ServicerContext
    ) -> ClaimTaskResponse:
        """Claim a pending task."""
        return runtime_handlers.claim_task(request, self.state())

    def SendTaskHeartbeat(
        self, request: SendTaskHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendTaskHeartbeatResponse:
        """Handle a heartbeat for a claimed task."""
        task = get_authenticated_task()
        return runtime_handlers.send_task_heartbeat(request, self.state(), task)

    def CreateTask(
        self, request: CreateTaskRequest, context: grpc.ServicerContext
    ) -> CreateTaskResponse:
        """Create a task."""
        task = get_authenticated_task()
        return runtime_handlers.create_task(request, self.state(), task, context)

    def PushTaskMessage(
        self, request: PushTaskMessageRequest, context: grpc.ServicerContext
    ) -> PushTaskMessageResponse:
        """Push a task message."""
        task = get_authenticated_task()
        return runtime_handlers.push_task_message(request, self.state(), task, context)

    def PushTaskEvents(
        self, request: PushTaskEventsRequest, context: grpc.ServicerContext
    ) -> PushTaskEventsResponse:
        """Push task events."""
        task = get_authenticated_task()
        return runtime_handlers.push_task_events(request, self.state(), task)

    def RecordTaskUsage(
        self, request: RecordTaskUsageRequest, context: grpc.ServicerContext
    ) -> RecordTaskUsageResponse:
        """Record task usage."""
        task = get_authenticated_task()
        return runtime_handlers.record_task_usage(request, self.state(), task)

    def PullTaskMessage(
        self, request: PullTaskMessageRequest, context: grpc.ServicerContext
    ) -> PullTaskMessageResponse:
        """Pull task messages."""
        task = get_authenticated_task()
        return runtime_handlers.pull_task_message(request, self.state(), task)

    def PushLogs(
        self, request: PushLogsRequest, context: grpc.ServicerContext
    ) -> PushLogsResponse:
        """Push logs."""
        state = self.state()
        task = get_authenticated_task()
        return runtime_handlers.push_logs(request, state, task)
