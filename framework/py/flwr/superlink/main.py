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

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from logging import INFO
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts

from flwr.common import log
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.error import http_error_translator
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.version import package_version
from flwr.superlink import extensions
from flwr.superlink.config_loader import (
    SuperLinkLifespanConfig,
    get_federation_manager,
    get_objectstore_linkstate_factories,
    load_control_auth_plugins,
    load_control_event_log_plugin,
)
from flwr.superlink.dependencies.account import AccountAccessDependency
from flwr.superlink.routers.control import router as control_router
from flwr.superlink.routers.control.middlewares import (
    ControlAuthenticationMiddleware,
    ControlEventLogMiddleware,
    ControlLicenseMiddleware,
)

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


def create_app(
    config: SuperLinkLifespanConfig | None = None,
    superlink_lifespan_class: type[SuperLinkLifespan] | None = None,
) -> FastAPI:
    """Create the SuperLink FastAPI app and its shared lifespan resources."""
    if config is None:
        is_simulation = False
        database = os.getenv("FLWR_DATABASE", FLWR_IN_MEMORY_DB_NAME)
        authn_plugin, authz_plugin = load_control_auth_plugins(
            os.getenv("FLWR_ACCOUNT_AUTH_CONFIG"), verify_tls_cert=True
        )
        event_log_plugin = (
            load_control_event_log_plugin()
            if os.getenv("FLWR_ENABLE_EVENT_LOG") == "1"
            else None
        )
    else:
        is_simulation = config.simulation
        database = config.database
        authn_plugin, authz_plugin = config.authn_plugin, config.authz_plugin
        event_log_plugin = config.event_log_plugin

    federation_manager = get_federation_manager(is_simulation=is_simulation)
    _, linkstate_factory = get_objectstore_linkstate_factories(
        database, federation_manager
    )
    # Force initialization before exposing LinkState through FastAPI dependencies
    linkstate_factory.state()

    # Instantiate SuperLink lifespan for legacy gRPC server if required
    superlink_lifespan = None
    if config and not config.disable_grpc_api:
        if superlink_lifespan_class is None:
            raise RuntimeError(
                "A SuperLink lifespan class is required when legacy gRPC is enabled."
            )
        superlink_lifespan = superlink_lifespan_class(config, linkstate_factory)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[dict[str, object]]:
        """Own process-lifetime resources for the combined SuperLink service."""
        log(INFO, "FastAPI lifespan: startup")

        try:
            if superlink_lifespan:
                # Temporary compatibility path: start the existing gRPC APIs from
                # FastAPI lifespan
                superlink_lifespan.startup()

            lifespan_state: dict[str, object] = {}
            async with AsyncExitStack() as stack:
                for lifespan_context in extensions.get_lifespan_contexts():
                    extension_state = await stack.enter_async_context(
                        lifespan_context(fastapi_app)
                    )
                    _merge_lifespan_state(lifespan_state, extension_state)
                yield lifespan_state
        finally:
            if superlink_lifespan:
                superlink_lifespan.shutdown()

            log(INFO, "FastAPI lifespan: shutdown")

    fastapi_app = FastAPI(
        title="SuperLink API",
        version=package_version,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
        generate_unique_id_function=generate_unique_route_id,
    )
    fastapi_app.state.superlink_lifespan = superlink_lifespan
    fastapi_app.state.linkstate_factory = linkstate_factory
    fastapi_app.state.account_access_dep = AccountAccessDependency(
        authn_plugin, authz_plugin
    )
    fastapi_app.state.control_event_log_plugin = event_log_plugin

    # Core APIs
    # fastapi_app.include_router(health.router)

    # SuperLink APIs
    fastapi_app.include_router(control_router)
    fastapi_app.add_middleware(ControlEventLogMiddleware)
    fastapi_app.add_middleware(ProtobufTranslationMiddleware)
    fastapi_app.add_middleware(ControlLicenseMiddleware)
    fastapi_app.add_middleware(ControlAuthenticationMiddleware)
    # Register last so it is outermost and translates errors from every Control layer.
    fastapi_app.middleware("http")(http_error_translator)
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


def __getattr__(name: str) -> FastAPI:
    """Create the module-level FastAPI app lazily."""
    if name == "app":
        fastapi_app = create_app()
        globals()[name] = fastapi_app
        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
