# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Runtime API servicer hosted by SuperNode."""


import grpc

# pylint: disable=E0611
from flwr.proto import runtime_pb2_grpc
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
)
from flwr.proto.message_pb2 import (
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse
from flwr.proto.runtime_pb2 import (
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.supercore.interceptors import get_authenticated_task
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.servicer.runtime import RuntimeServicer
from flwr.supernode.nodestate import NodeState, NodeStateFactory

from . import runtime_handlers


# pylint: disable=C0103,W0613,W0201
class SuperNodeRuntimeServicer(RuntimeServicer, runtime_pb2_grpc.RuntimeServicer):
    """Runtime API servicer hosted by SuperNode."""

    def __init__(
        self,
        state_factory: NodeStateFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.objectstore_factory = objectstore_factory

    def state(self) -> NodeState:
        """Return the NodeState instance."""
        return self.state_factory.state()

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        return runtime_handlers.get_run(request, self.state_factory.state())

    def PullTaskInput(
        self, request: PullTaskInputRequest, context: grpc.ServicerContext
    ) -> PullTaskInputResponse:
        """Pull Message, Context, and Run."""
        task = get_authenticated_task()
        return runtime_handlers.pull_task_input(
            request, self.state_factory.state(), task
        )

    def PushTaskOutput(
        self, request: PushTaskOutputRequest, context: grpc.ServicerContext
    ) -> PushTaskOutputResponse:
        """Push Message and Context."""
        task = get_authenticated_task()
        return runtime_handlers.push_task_output(
            request, self.state_factory.state(), task
        )

    def PullMessages(
        self, request: PullAppMessagesRequest, context: grpc.ServicerContext
    ) -> PullAppMessagesResponse:
        """Pull messages for ClientApp; currently returns exactly one message."""
        task = get_authenticated_task()
        state = self.state_factory.state()
        return runtime_handlers.pull_messages(request, state, task)

    def PushMessages(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push messages for ClientApp; currently accepts exactly one message."""
        task = get_authenticated_task()
        return runtime_handlers.push_messages(request, self.state_factory.state(), task)

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        return runtime_handlers.get_nodes(request)

    def StartAutomation(
        self,
        request: StartAutomationRequest,
        context: grpc.ServicerContext,
    ) -> StartAutomationResponse:
        """Reject automation requests from ClientApp tasks."""
        return runtime_handlers.start_automation(request)

    def GetConnector(
        self, request: GetConnectorRequest, context: grpc.ServicerContext
    ) -> GetConnectorResponse:
        """Reject connector credential requests from ClientApp tasks."""
        return runtime_handlers.get_connector(request)

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        task = get_authenticated_task()
        return runtime_handlers.push_object(request, self.state_factory.state(), task)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        task = get_authenticated_task()
        return runtime_handlers.pull_object(request, self.state_factory.state(), task)

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        return runtime_handlers.confirm_message_received(
            request, self.state_factory.state()
        )
