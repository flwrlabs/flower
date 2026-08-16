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
"""FastAPI SuperExec authentication dependency for SuperNode Runtime routes."""

from typing import Annotated

from fastapi import Depends, Request

from flwr.supercore.dependencies.superexec import authenticate_superexec_request
from flwr.supernode.nodestate import NodeState

from .nodestate import get_nodestate


class SuperExecAuthDependency:
    """Authenticate one SuperExec Runtime operation."""

    def __init__(self, method: str) -> None:
        self.method = method

    def __call__(
        self,
        request: Request,
        state: Annotated[NodeState, Depends(get_nodestate)],
    ) -> None:
        """Authenticate a SuperExec request using SuperNode state."""
        authenticate_superexec_request(request, state, self.method)
