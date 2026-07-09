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

from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from logging import INFO
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts

from flwr import __version__
from flwr.common import log
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_SQLITE_DB_URL
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink import extensions
from flwr.superlink.federation import NoOpFederationManager

if TYPE_CHECKING:
    from flwr.superlink.cli.flower_superlink import SuperLinkLifespan


def generate_unique_route_id(route: APIRoute) -> str:
    """Generate stable route IDs from route handler names."""
    return route.name


def _merge_lifespan_state(
    lifespan_state: dict[str, object],
    extension_state: Mapping[str, object] | None,
) -> None:
    """Merge extension lifespan state into the app lifespan state."""
    if extension_state is None:
        return
    for key, value in extension_state.items():
        if key in lifespan_state:
            raise ValueError(
                f"Duplicate lifespan state key detected: {key}. "
                "Please ensure each SuperLink extension provides unique state keys."
            )
        lifespan_state[key] = value


def _create_default_linkstate_factory() -> LinkStateFactory:
    """Create the default LinkStateFactory for direct uvicorn startup."""
    objectstore_factory = ObjectStoreFactory(FLWR_IN_MEMORY_SQLITE_DB_URL)
    return LinkStateFactory(
        FLWR_IN_MEMORY_SQLITE_DB_URL,
        NoOpFederationManager(),
        objectstore_factory,
    )


def create_app(
    *,
    linkstate_factory: LinkStateFactory,
    superlink_lifespan: SuperLinkLifespan | None = None,
    start_legacy_grpc: bool = False,
) -> FastAPI:
    """Create the SuperLink FastAPI app.

    This FastAPI app can be started in two ways:
    1. Via `flower-superlink`: the CLI always passes a `linkstate_factory`.
       When FastAPI also starts the legacy gRPC APIs for compatibility, the CLI
       also passes a `superlink_lifespan` initialized with the same factory.
    2. Via `uvicorn flwr.superlink.main:app`: the module-level app uses an
       in-memory SQLite LinkStateFactory. Direct callers of `create_app` must
       provide their desired `linkstate_factory` explicitly.
    """

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[dict[str, object]]:
        """Own process-lifetime resources for the combined SuperLink service."""
        log(INFO, "FastAPI lifespan: startup")

        try:
            if superlink_lifespan is not None:
                # Store the SuperLinkLifespan where future REST routers can access
                # shared state through FastAPI dependencies
                fastapi_app.state.superlink_lifespan = superlink_lifespan

            if superlink_lifespan is not None and start_legacy_grpc:
                # Temporary compatibility path: start the existing gRPC APIs from
                # FastAPI lifespan
                superlink_lifespan.startup()

            fastapi_app.state.linkstate_factory = linkstate_factory

            lifespan_state: dict[str, object] = {}
            async with AsyncExitStack() as stack:
                for lifespan_context in extensions.get_lifespan_contexts():
                    extension_state = await stack.enter_async_context(
                        lifespan_context(fastapi_app)
                    )
                    _merge_lifespan_state(lifespan_state, extension_state)
                yield lifespan_state
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
        generate_unique_id_function=generate_unique_route_id,
    )

    # Core APIs
    # fastapi_app.include_router(health.router)

    # SuperLink APIs
    # fastapi_app.include_router(control.router)
    # fastapi_app.include_router(runtime.router)

    # Extension hooks
    extensions.configure_app(fastapi_app)

    validate_unique_route_operation_ids(fastapi_app)

    return fastapi_app


def validate_unique_route_operation_ids(fastapi_app: FastAPI) -> None:
    """Use route handler names as OpenAPI operation IDs.

    Call this only after all routers have been registered. Route handler names
    must be unique across the composed application.

    Example:

    - A handler named `create_api_key` produces operation ID `create_api_key`.
    - Two handlers with the same name produce an operation ID collision.
    """
    operation_ids = set()
    for route_context in iter_route_contexts(fastapi_app.routes):
        if isinstance(route_context.route, APIRoute):
            op_id = generate_unique_route_id(route_context.route)
            if op_id in operation_ids:
                raise ValueError(
                    f"Operation ID collision detected: {op_id}. "
                    "Please ensure all route handler function names are unique."
                )
            operation_ids.add(op_id)


# Temporary: we need a way to provision the FastAPI server
app = create_app(linkstate_factory=_create_default_linkstate_factory())
