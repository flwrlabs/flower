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
"""FastAPI dependencies for the shared Runtime API."""

from typing import Annotated, Any, Protocol, TypeVar, cast

from fastapi import Depends, Request, Security

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
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    GetRunSeriesEventsRequest,
    GetRunSeriesEventsResponse,
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
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.corestate import CoreState
from flwr.supercore.error import ApiErrorCode, FlowerError

from .superexec import authenticate_superexec_request
from .task import TaskTokenDependency, authenticate_task


class RuntimeStateFactory(Protocol):
    """Factory providing state to the Runtime API."""

    def state(self) -> CoreState:
        """Return the Runtime state."""


StateT_contra = TypeVar("StateT_contra", bound=CoreState, contravariant=True)


class RuntimeHandlers(Protocol[StateT_contra]):
    """Component-specific handlers used by the shared Runtime API router."""

    def pull_pending_tasks(
        self, request: PullPendingTasksRequest, state: StateT_contra
    ) -> PullPendingTasksResponse:
        """Pull pending tasks."""

    def pull_task_input(
        self, request: PullTaskInputRequest, state: StateT_contra, task: Task
    ) -> PullTaskInputResponse:
        """Pull task input."""

    def push_task_output(
        self, request: PushTaskOutputRequest, state: StateT_contra, task: Task
    ) -> PushTaskOutputResponse:
        """Push task output."""

    def push_object(
        self, request: PushObjectRequest, state: StateT_contra, task: Task
    ) -> PushObjectResponse:
        """Push an object."""

    def pull_object(
        self, request: PullObjectRequest, state: StateT_contra, task: Task
    ) -> PullObjectResponse:
        """Pull an object."""

    def confirm_message_received(
        self,
        request: ConfirmMessageReceivedRequest,
        state: StateT_contra,
        task: Task,
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message receipt."""

    def start_automation(
        self, request: StartAutomationRequest, state: StateT_contra, task: Task
    ) -> StartAutomationResponse:
        """Start an automation."""

    def get_connector(
        self, request: GetConnectorRequest, state: StateT_contra, task: Task
    ) -> GetConnectorResponse:
        """Get connector credentials."""

    def push_messages(
        self, request: PushAppMessagesRequest, state: StateT_contra, task: Task
    ) -> PushAppMessagesResponse:
        """Push app messages."""

    def pull_messages(
        self, request: PullAppMessagesRequest, state: StateT_contra, task: Task
    ) -> PullAppMessagesResponse:
        """Pull app messages."""

    def get_nodes(
        self, request: GetNodesRequest, state: StateT_contra, task: Task
    ) -> GetNodesResponse:
        """Get available nodes."""

    def get_run_series_events(
        self,
        request: GetRunSeriesEventsRequest,
        state: StateT_contra,
        task: Task,
    ) -> GetRunSeriesEventsResponse:
        """Get events from the authenticated task's run series."""


def get_runtime_state(request: Request) -> CoreState:
    """Return the state configured for the Runtime API."""
    factory = cast(
        RuntimeStateFactory | None,
        getattr(request.app.state, "runtime_state_factory", None),
    )
    if factory is None:
        error_code, error_message = cast(
            tuple[ApiErrorCode, str],
            request.app.state.runtime_state_factory_error,
        )
        raise FlowerError(error_code, error_message)
    return factory.state()


def get_runtime_handlers(request: Request) -> RuntimeHandlers[Any]:
    """Return the handlers configured for the Runtime API."""
    handlers = cast(
        RuntimeHandlers[Any] | None,
        getattr(request.app.state, "runtime_handlers", None),
    )
    if handlers is None:
        raise RuntimeError("Runtime handlers are not initialized.")
    return handlers


RuntimeStateDependency = Annotated[CoreState, Depends(get_runtime_state)]
RuntimeHandlersDependency = Annotated[
    RuntimeHandlers[Any], Depends(get_runtime_handlers)
]


def get_task(
    request: Request,
    token: TaskTokenDependency,
    state: RuntimeStateDependency,
) -> Task:
    """Return the task authenticated by the Runtime task-token header."""
    return authenticate_task(request, token, state)


TaskDependency = Annotated[Task, Security(get_task)]


class SuperExecAuthDependency:
    """Authenticate one SuperExec Runtime operation."""

    def __init__(self, method: str) -> None:
        self.method = method

    def __call__(self, request: Request, state: RuntimeStateDependency) -> None:
        """Authenticate a SuperExec request using the configured Runtime state."""
        authenticate_superexec_request(request, state, self.method)
