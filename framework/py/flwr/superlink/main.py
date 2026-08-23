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

import asyncio
import os
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from logging import INFO
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

from flwr.supercore import log
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.error import http_error_translator
from flwr.supercore.http_logging import configure_uvicorn_logging
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.routers import health
from flwr.supercore.version import package_version
from flwr.superlink import extensions
from flwr.superlink.config_loader import (
    SuperLinkLifespanConfig,
    get_federation_manager,
    get_objectstore_linkstate_factories,
    load_control_authn_plugin,
    load_control_event_log_plugin,
)
from flwr.superlink.dependencies.account import AccountAccessDependency
from flwr.superlink.routers.control import router as control_router
from flwr.superlink.routers.control.middlewares import (
    ControlAuthenticationMiddleware,
    ControlEventLogMiddleware,
    ControlLicenseMiddleware,
)
from flwr.superlink.routers.runtime import responses_router
from flwr.superlink.routers.runtime import router as runtime_router

try:
    from flwr.ee import get_ee_linkstate_db as get_ee_linkstate_db
except ModuleNotFoundError as exc:
    if exc.name != "flwr.ee":
        raise

    def get_ee_linkstate_db() -> str:
        """Return the configured LinkState database."""
        return os.getenv("FLWR_DATABASE", FLWR_IN_MEMORY_DB_NAME)


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


def _get_middleware() -> list[Middleware]:
    """Return middleware in request execution order, outermost first."""
    return [
        *extensions.get_middleware(),
        Middleware(BaseHTTPMiddleware, dispatch=http_error_translator),
        Middleware(ControlAuthenticationMiddleware),
        Middleware(ControlLicenseMiddleware),
        Middleware(ProtobufTranslationMiddleware),
        Middleware(ControlEventLogMiddleware),
    ]


def create_app(  # pylint: disable=too-many-statements
    config: SuperLinkLifespanConfig | None = None,
    superlink_lifespan_class: type[SuperLinkLifespan] | None = None,
) -> FastAPI:
    """Create the SuperLink FastAPI app and its shared lifespan resources."""
    if config is None:
        is_simulation = False
        database = get_ee_linkstate_db()
        superexec_auth_secret = None
        authn_plugin = load_control_authn_plugin()
        event_log_plugin = (
            load_control_event_log_plugin()
            if os.getenv("FLWR_ENABLE_EVENT_LOG") == "1"
            else None
        )
    else:
        is_simulation = config.simulation
        database = config.database
        superexec_auth_secret = config.superexec_auth_secret
        authn_plugin = config.authn_plugin
        event_log_plugin = config.event_log_plugin

    federation_manager = get_federation_manager(is_simulation=is_simulation)
    _, linkstate_factory = get_objectstore_linkstate_factories(
        database, federation_manager
    )
    # Force initialization before exposing LinkState through FastAPI dependencies
    linkstate_factory.state()

    # Instantiate the lifespan which owns the Control and Fleet gRPC servers.
    superlink_lifespan = None
    if config:
        if superlink_lifespan_class is None:
            raise RuntimeError(
                "A SuperLink lifespan class is required when configuration is provided."
            )
        superlink_lifespan = superlink_lifespan_class(config, linkstate_factory)

    @asynccontextmanager
    async def lifespan(fastapi_app: FastAPI) -> AsyncIterator[dict[str, object]]:
        """Own process-lifetime resources for the combined SuperLink service."""
        log(INFO, "FastAPI lifespan: startup")
        superlink_startup_attempted = False
        superlink_shutdown = False

        def shutdown_superlink() -> None:
            """Shut down resources even when startup only partially succeeds."""
            nonlocal superlink_shutdown
            if (
                superlink_lifespan
                and superlink_startup_attempted
                and not superlink_shutdown
            ):
                superlink_lifespan.shutdown()
                superlink_shutdown = True

        try:
            extensions.set_notification_loop(asyncio.get_running_loop())
            lifespan_state: dict[str, object] = {}
            async with AsyncExitStack() as stack:
                for lifespan_context in extensions.get_lifespan_contexts():
                    extension_state = await stack.enter_async_context(
                        lifespan_context(fastapi_app)
                    )
                    _merge_lifespan_state(lifespan_state, extension_state)
                if superlink_lifespan:
                    # Start gRPC only after extension resources are ready, so a
                    # run-start notification cannot race lifespan initialization.
                    superlink_startup_attempted = True
                    superlink_lifespan.startup()

                try:
                    yield lifespan_state
                finally:
                    shutdown_superlink()
                    extensions.clear_notification_loop()
        finally:
            # Also clean up a partially started SuperLink when startup raises
            # before the inner shutdown block is reached.
            shutdown_superlink()
            extensions.clear_notification_loop()
            log(INFO, "FastAPI lifespan: shutdown")

    fastapi_app = FastAPI(
        title="SuperLink API",
        version=package_version,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
        generate_unique_id_function=generate_unique_route_id,
        middleware=_get_middleware(),
    )
    fastapi_app.state.superlink_lifespan = superlink_lifespan
    fastapi_app.state.linkstate_factory = linkstate_factory
    fastapi_app.state.superexec_auth_secret = superexec_auth_secret
    fastapi_app.state.account_access_dep = AccountAccessDependency(authn_plugin)
    fastapi_app.state.control_event_log_plugin = event_log_plugin

    # Core APIs
    fastapi_app.include_router(health.router)

    # SuperLink APIs
    fastapi_app.include_router(control_router)
    fastapi_app.include_router(runtime_router)
    fastapi_app.include_router(responses_router)

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
        configure_uvicorn_logging()
        fastapi_app = create_app()
        globals()[name] = fastapi_app
        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
