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
"""FastAPI task-token authentication dependency for SuperNode Runtime routes."""

from typing import Annotated

from fastapi import Depends, Request, Security

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.dependencies.task import TaskTokenDependency, authenticate_task
from flwr.supernode.nodestate import NodeState

from .nodestate import get_nodestate

NodeStateDependency = Annotated[NodeState, Depends(get_nodestate)]


def get_task(
    request: Request,
    token: TaskTokenDependency,
    state: NodeStateDependency,
) -> Task:
    """Return the task authenticated by the Runtime task-token header."""
    return authenticate_task(request, token, state)


TaskDependency = Annotated[Task, Security(get_task)]
