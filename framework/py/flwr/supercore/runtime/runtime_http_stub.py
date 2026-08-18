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
"""HTTP stub for the Runtime API."""

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
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
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
from flwr.supercore.protobuf.client import ProtobufClient


# Match the method names exposed by the generated gRPC RuntimeStub.
# pylint: disable=invalid-name
class RuntimeHttpStub(ProtobufClient):
    """Protobuf-over-HTTP client for the Runtime API."""

    def PullPendingTasks(
        self, request: PullPendingTasksRequest
    ) -> PullPendingTasksResponse:
        """Pull tasks waiting to be executed."""
        return self._unary_unary(
            path="/v1/runtime/pull-pending-tasks",
            rpc_method="/flwr.proto.Runtime/PullPendingTasks",
            request=request,
            response_type=PullPendingTasksResponse,
        )

    def ClaimTask(self, request: ClaimTaskRequest) -> ClaimTaskResponse:
        """Claim a task for execution."""
        return self._unary_unary(
            path="/v1/runtime/claim-task",
            rpc_method="/flwr.proto.Runtime/ClaimTask",
            request=request,
            response_type=ClaimTaskResponse,
        )

    def GetRun(self, request: GetRunRequest) -> GetRunResponse:
        """Get the run associated with a task."""
        return self._unary_unary(
            path="/v1/runtime/get-run",
            rpc_method="/flwr.proto.Runtime/GetRun",
            request=request,
            response_type=GetRunResponse,
        )

    def SendTaskHeartbeat(
        self, request: SendTaskHeartbeatRequest
    ) -> SendTaskHeartbeatResponse:
        """Send a heartbeat for a claimed task."""
        return self._unary_unary(
            path="/v1/runtime/send-task-heartbeat",
            rpc_method="/flwr.proto.Runtime/SendTaskHeartbeat",
            request=request,
            response_type=SendTaskHeartbeatResponse,
        )

    def PullTaskInput(self, request: PullTaskInputRequest) -> PullTaskInputResponse:
        """Pull the input for a claimed task."""
        return self._unary_unary(
            path="/v1/runtime/pull-task-input",
            rpc_method="/flwr.proto.Runtime/PullTaskInput",
            request=request,
            response_type=PullTaskInputResponse,
        )

    def PushTaskOutput(self, request: PushTaskOutputRequest) -> PushTaskOutputResponse:
        """Push the output of a claimed task."""
        return self._unary_unary(
            path="/v1/runtime/push-task-output",
            rpc_method="/flwr.proto.Runtime/PushTaskOutput",
            request=request,
            response_type=PushTaskOutputResponse,
        )

    def PushObject(self, request: PushObjectRequest) -> PushObjectResponse:
        """Push an object to the Runtime API."""
        return self._unary_unary(
            path="/v1/runtime/push-object",
            rpc_method="/flwr.proto.Runtime/PushObject",
            request=request,
            response_type=PushObjectResponse,
        )

    def PullObject(self, request: PullObjectRequest) -> PullObjectResponse:
        """Pull an object from the Runtime API."""
        return self._unary_unary(
            path="/v1/runtime/pull-object",
            rpc_method="/flwr.proto.Runtime/PullObject",
            request=request,
            response_type=PullObjectResponse,
        )

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest
    ) -> ConfirmMessageReceivedResponse:
        """Confirm that a message and its objects were received."""
        return self._unary_unary(
            path="/v1/runtime/confirm-message-received",
            rpc_method="/flwr.proto.Runtime/ConfirmMessageReceived",
            request=request,
            response_type=ConfirmMessageReceivedResponse,
        )

    def PushMessages(self, request: PushAppMessagesRequest) -> PushAppMessagesResponse:
        """Push app messages to the Runtime API."""
        return self._unary_unary(
            path="/v1/runtime/push-messages",
            rpc_method="/flwr.proto.Runtime/PushMessages",
            request=request,
            response_type=PushAppMessagesResponse,
        )

    def PullMessages(self, request: PullAppMessagesRequest) -> PullAppMessagesResponse:
        """Pull app messages from the Runtime API."""
        return self._unary_unary(
            path="/v1/runtime/pull-messages",
            rpc_method="/flwr.proto.Runtime/PullMessages",
            request=request,
            response_type=PullAppMessagesResponse,
        )

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
