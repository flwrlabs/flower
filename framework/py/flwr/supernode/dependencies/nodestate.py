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
"""FastAPI dependency for SuperNode NodeState."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supernode.nodestate import NodeState, NodeStateFactory


def get_nodestate(request: Request) -> NodeState:
    """Return the SuperNode NodeState for the current request."""
    nodestate_factory = cast(
        NodeStateFactory | None,
        getattr(request.app.state, "nodestate_factory", None),
    )
    if nodestate_factory is None:
        raise FlowerError(
            ApiErrorCode.NODESTATE_NOT_INITIALIZED,
            "SuperNode NodeStateFactory is not initialized.",
        )

    return nodestate_factory.state()
