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
"""FastAPI dependencies shared by SuperLink routers."""


from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol, cast

from fastapi import HTTPException, Request, status
from starlette.datastructures import State

from flwr.server.superlink.linkstate import LinkState, LinkStateFactory


class _SuperLinkLifespanState(Protocol):
    """Subset of SuperLinkLifespan state needed by FastAPI dependencies."""

    state_factory: LinkStateFactory | None


async def get_linkstate(request: Request[State]) -> AsyncGenerator[LinkState, None]:
    """Yield the SuperLink LinkState for the current request."""
    superlink_lifespan = cast(
        _SuperLinkLifespanState | None,
        getattr(request.app.state, "superlink_lifespan", None),
    )
    if superlink_lifespan is None or superlink_lifespan.state_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SuperLink lifespan state is not initialized.",
        )

    yield superlink_lifespan.state_factory.state()
