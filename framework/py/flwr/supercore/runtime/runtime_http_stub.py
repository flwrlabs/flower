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
"""HTTP stub for the shared Runtime API."""

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
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.supercore.protobuf.client import ProtobufClient


# Match the method names exposed by the generated gRPC RuntimeStub.
# pylint: disable=invalid-name
class RuntimeHttpStub(ProtobufClient):
    """Protobuf-over-HTTP client for shared Runtime API methods."""

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
