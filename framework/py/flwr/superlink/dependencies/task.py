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
"""FastAPI task-token authentication dependency for Runtime routes."""

from typing import Annotated

from fastapi import Depends, Request

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.error import ApiErrorCode, FlowerError

from .linkstate import get_linkstate

LinkStateDependency = Annotated[LinkState, Depends(get_linkstate)]


def get_task(request: Request, state: LinkStateDependency) -> Task:
    """Return the task authenticated by the Runtime task-token header."""
    tokens = request.headers.getlist(TASK_TOKEN_HEADER)
    # Match gRPC metadata validation and reject ambiguous credentials.
    token = tokens[0] if len(tokens) == 1 and tokens[0] else None
    task = state.get_task_by_token(token) if token is not None else None
    if task is None:
        raise FlowerError(
            ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED,
            "Runtime task-token authentication failed.",
        )
    return task
