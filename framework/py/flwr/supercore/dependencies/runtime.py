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

from types import ModuleType
from typing import Annotated, Protocol, cast

from fastapi import Depends, Request, Security

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.corestate import CoreState
from flwr.supercore.error import ApiErrorCode, FlowerError

from .superexec import authenticate_superexec_request
from .task import TaskTokenDependency, authenticate_task


class RuntimeStateFactory(Protocol):
    """Factory providing state to the Runtime API."""

    def state(self) -> CoreState:
        """Return the Runtime state."""


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


def get_runtime_handlers(request: Request) -> ModuleType:
    """Return the handlers configured for the Runtime API."""
    handlers = cast(
        ModuleType | None,
        getattr(request.app.state, "runtime_handlers", None),
    )
    if handlers is None:
        raise RuntimeError("Runtime handlers are not initialized.")
    return handlers


RuntimeStateDependency = Annotated[CoreState, Depends(get_runtime_state)]
RuntimeHandlersDependency = Annotated[ModuleType, Depends(get_runtime_handlers)]


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
