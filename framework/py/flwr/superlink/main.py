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

from fastapi import FastAPI

from flwr import __version__
from flwr.common import log
from flwr.supercore.routers import health
from flwr.superlink.routers import runtime


def create_app() -> FastAPI:
    """Create the SuperLink FastAPI app."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log(INFO, "FastAPI lifespan: startup")
        yield
        log(INFO, "FastAPI lifespan: shutdown")

    fastapi_app = FastAPI(
        title="SuperLink API",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # SuperCore API routers
    fastapi_app.include_router(health.router)

    # SuperLink API routers
    fastapi_app.include_router(runtime.router)

    return fastapi_app


app = create_app()
