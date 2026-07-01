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
"""SuperLink API."""


from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import INFO
from typing import TYPE_CHECKING

from fastapi import FastAPI

from flwr import __version__
from flwr.common import log
from flwr.supercore.routers import health
from flwr.superlink.routers import control, runtime

if TYPE_CHECKING:
    from flwr.superlink.cli.flower_superlink import SuperLinkLifespan


def create_app(
    *,
    superlink_lifespan: SuperLinkLifespan | None = None,
    start_legacy_grpc: bool = False,
) -> FastAPI:
    """Create the SuperLink FastAPI app.

    This FastAPI app can be started in two ways:
    1. Via `flower-superlink`: `superlink_lifespan` will be passed.
    2. Via `uvicorn flwr.superlink.main:app`: `superlink_lifespan` will be None.
    """

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[None]:
        """Own process-lifetime resources for the combined SuperLink service."""
        log(INFO, "FastAPI lifespan: startup")

        if superlink_lifespan is not None:
            # Store the SuperLinkLifespan where future REST routers can access shared
            # state through FastAPI dependencies
            fastapi_app.state.superlink_lifespan = superlink_lifespan

        if superlink_lifespan is not None and start_legacy_grpc:
            # Temporary compatibility path: start the existing gRPC APIs from
            # FastAPI lifespan
            superlink_lifespan.startup()

        try:
            yield
        finally:
            if superlink_lifespan is not None and start_legacy_grpc:
                superlink_lifespan.shutdown()

            log(INFO, "FastAPI lifespan: shutdown")

    fastapi_app = FastAPI(
        title="SuperLink API",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Core APIs
    fastapi_app.include_router(health.router)

    # SuperLink APIs
    fastapi_app.include_router(control.router)
    fastapi_app.include_router(runtime.router)

    return fastapi_app


app = create_app()
