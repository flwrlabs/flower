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
"""SuperNode API."""


from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from logging import INFO

from fastapi import Depends, FastAPI

from flwr import __version__
from flwr.supercore import log
from flwr.supercore.dependencies.runtime import RuntimeHandlers
from flwr.supercore.dependencies.runtime_version import RuntimeVersionDependency
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.routers import health
from flwr.supercore.routers.runtime import router as runtime_router
from flwr.supernode.nodestate import NodeState, NodeStateFactory
from flwr.supernode.servicer.runtime import runtime_handlers

_RUNTIME_HANDLERS: RuntimeHandlers[NodeState] = runtime_handlers
_RUNTIME_VERSION_DEPENDENCY = Depends(
    RuntimeVersionDependency(
        component_name="SuperNode",
        connection_name="Caller <-> SuperNode Runtime API",
    )
)


def create_app(
    state_factory: NodeStateFactory | None = None,
    superexec_auth_secret: bytes | None = None,
) -> FastAPI:
    """Create the SuperNode FastAPI app."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        log(INFO, "FastAPI lifespan: startup")
        yield
        log(INFO, "FastAPI lifespan: shutdown")

    fastapi_app = FastAPI(
        title="SuperNode API",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    fastapi_app.state.runtime_state_factory = state_factory
    fastapi_app.state.runtime_state_factory_error = (
        ApiErrorCode.NODESTATE_NOT_INITIALIZED,
        "SuperNode NodeStateFactory is not initialized.",
    )
    fastapi_app.state.runtime_handlers = _RUNTIME_HANDLERS
    fastapi_app.state.superexec_auth_secret = superexec_auth_secret

    # Core APIs
    fastapi_app.include_router(health.router)

    # SuperNode APIs
    fastapi_app.include_router(
        runtime_router, dependencies=[_RUNTIME_VERSION_DEPENDENCY]
    )

    fastapi_app.add_middleware(ProtobufTranslationMiddleware)

    # Apply the FlowerError translation layer last to make it outermost
    fastapi_app.middleware("http")(http_error_translator)

    return fastapi_app


app = create_app()
