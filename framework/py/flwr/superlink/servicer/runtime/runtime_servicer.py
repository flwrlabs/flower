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
"""Runtime API servicer hosted by SuperLink."""

import grpc

from flwr.proto import runtime_pb2_grpc  # pylint: disable=E0611
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
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
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.interceptors import get_authenticated_task
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.servicer.runtime import RuntimeServicer

from . import runtime_handlers


class SuperLinkRuntimeServicer(RuntimeServicer, runtime_pb2_grpc.RuntimeServicer):
    """Runtime API servicer hosted by SuperLink."""

    def __init__(
        self,
        state_factory: LinkStateFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.objectstore_factory = objectstore_factory

    def state(self) -> LinkState:
        """Return the LinkState instance."""
        return self.state_factory.state()

    def PullPendingTasks(
        self, request: PullPendingTasksRequest, context: grpc.ServicerContext
    ) -> PullPendingTasksResponse:
        """Process due automations, then pull pending tasks."""
        return runtime_handlers.pull_pending_tasks(request, self.state())

    def GetNodes(
        self, request: GetNodesRequest, context: grpc.ServicerContext
    ) -> GetNodesResponse:
        """Get available nodes."""
        task = get_authenticated_task()
        return runtime_handlers.get_nodes(request, self.state(), task)

    def PushMessages(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push a set of Messages."""
        task = get_authenticated_task()
        return runtime_handlers.push_messages(request, self.state(), task)

    def PullMessages(
        self, request: PullAppMessagesRequest, context: grpc.ServicerContext
    ) -> PullAppMessagesResponse:
        """Pull a set of Messages."""
        task = get_authenticated_task()
        return runtime_handlers.pull_messages(
            request,
            self.state(),
            task,
        )

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        return runtime_handlers.get_run(request, self.state())

    def GetConnector(
        self, request: GetConnectorRequest, context: grpc.ServicerContext
    ) -> GetConnectorResponse:
        """Return credentials authorized for the authenticated connector task."""
        task = get_authenticated_task()
        return runtime_handlers.get_connector(request, self.state(), task)

    def PullTaskInput(
        self, request: PullTaskInputRequest, context: grpc.ServicerContext
    ) -> PullTaskInputResponse:
        """Pull ServerApp process inputs."""
        task = get_authenticated_task()
        return runtime_handlers.pull_task_input(request, self.state(), task)

    def PushTaskOutput(
        self, request: PushTaskOutputRequest, context: grpc.ServicerContext
    ) -> PushTaskOutputResponse:
        """Push ServerApp process outputs."""
        task = get_authenticated_task()
        return runtime_handlers.push_task_output(request, self.state(), task)

    def StartAutomation(
        self,
        request: StartAutomationRequest,
        context: grpc.ServicerContext,
    ) -> StartAutomationResponse:
        """Start an automation."""
        task = get_authenticated_task()
        return runtime_handlers.start_automation(request, self.state(), task)

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        task = get_authenticated_task()
        return runtime_handlers.push_object(request, self.state(), task)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        task = get_authenticated_task()
        return runtime_handlers.pull_object(request, self.state(), task)

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        task = get_authenticated_task()
        return runtime_handlers.confirm_message_received(
            request,
            self.state(),
            task,
        )
