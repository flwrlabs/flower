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
"""HTTP stub for the SuperLink Runtime API."""

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushTaskEventsRequest,
    PushTaskEventsResponse,
    PushTaskMessageRequest,
    PushTaskMessageResponse,
    RecordTaskUsageRequest,
    RecordTaskUsageResponse,
)
from flwr.supercore.runtime import RuntimeHttpStub as CoreRuntimeHttpStub


# Match the method names exposed by the generated gRPC RuntimeStub.
# pylint: disable=invalid-name
class RuntimeHttpStub(CoreRuntimeHttpStub):
    """Protobuf-over-HTTP client for SuperLink Runtime API methods."""

    def PushLogs(self, request: PushLogsRequest) -> PushLogsResponse:
        """Push task logs to the Runtime API."""
        return self._unary_unary(
            path="/v1/runtime/push-logs",
            rpc_method="/flwr.proto.Runtime/PushLogs",
            request=request,
            response_type=PushLogsResponse,
        )

    def GetNodes(self, request: GetNodesRequest) -> GetNodesResponse:
        """Get nodes available to the ServerApp."""
        return self._unary_unary(
            path="/v1/runtime/get-nodes",
            rpc_method="/flwr.proto.Runtime/GetNodes",
            request=request,
            response_type=GetNodesResponse,
        )

    def CreateTask(self, request: CreateTaskRequest) -> CreateTaskResponse:
        """Create a child task."""
        return self._unary_unary(
            path="/v1/runtime/create-task",
            rpc_method="/flwr.proto.Runtime/CreateTask",
            request=request,
            response_type=CreateTaskResponse,
        )

    def StartAutomation(
        self, request: StartAutomationRequest
    ) -> StartAutomationResponse:
        """Start an automation from a Runtime task."""
        return self._unary_unary(
            path="/v1/runtime/start-automation",
            rpc_method="/flwr.proto.Runtime/StartAutomation",
            request=request,
            response_type=StartAutomationResponse,
        )

    def PushTaskMessage(
        self, request: PushTaskMessageRequest
    ) -> PushTaskMessageResponse:
        """Push a task message."""
        return self._unary_unary(
            path="/v1/runtime/push-task-message",
            rpc_method="/flwr.proto.Runtime/PushTaskMessage",
            request=request,
            response_type=PushTaskMessageResponse,
        )

    def PushTaskEvents(self, request: PushTaskEventsRequest) -> PushTaskEventsResponse:
        """Push task events."""
        return self._unary_unary(
            path="/v1/runtime/push-task-events",
            rpc_method="/flwr.proto.Runtime/PushTaskEvents",
            request=request,
            response_type=PushTaskEventsResponse,
        )

    def PullTaskMessage(
        self, request: PullTaskMessageRequest
    ) -> PullTaskMessageResponse:
        """Pull a task message."""
        return self._unary_unary(
            path="/v1/runtime/pull-task-message",
            rpc_method="/flwr.proto.Runtime/PullTaskMessage",
            request=request,
            response_type=PullTaskMessageResponse,
        )

    def RecordTaskUsage(
        self, request: RecordTaskUsageRequest
    ) -> RecordTaskUsageResponse:
        """Record task usage."""
        return self._unary_unary(
            path="/v1/runtime/record-task-usage",
            rpc_method="/flwr.proto.Runtime/RecordTaskUsage",
            request=request,
            response_type=RecordTaskUsageResponse,
        )

    def GetConnector(self, request: GetConnectorRequest) -> GetConnectorResponse:
        """Get connector credentials."""
        return self._unary_unary(
            path="/v1/runtime/get-connector",
            rpc_method="/flwr.proto.Runtime/GetConnector",
            request=request,
            response_type=GetConnectorResponse,
        )
