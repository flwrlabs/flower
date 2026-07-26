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
"""Tests for SuperLink FastAPI application construction."""

from typing import cast

from fastapi import FastAPI
from fastapi.routing import APIRoute, iter_route_contexts
from pytest import MonkeyPatch
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from flwr.supercore.error import http_error_translator
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.supercore.routers.health.router import health
from flwr.superlink.routers.control.middlewares import (
    ControlAuthenticationMiddleware,
    ControlEventLogMiddleware,
    ControlLicenseMiddleware,
)

from . import extensions, main


class _ExtensionMiddleware:
    """Test middleware contributed by a SuperLink extension."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)


def _create_app(
    monkeypatch: MonkeyPatch,
    extension_middleware: tuple[Middleware, ...] = (),
) -> FastAPI:
    """Create an app without routes or state contributed by an installed extension."""
    monkeypatch.setattr(extensions, "get_middleware", lambda: extension_middleware)
    monkeypatch.setattr(extensions, "configure_app", lambda _: None)
    return main.create_app()


def _control_middleware_classes() -> list[type[object]]:
    """Return Control middleware classes in request execution order."""
    return [
        BaseHTTPMiddleware,
        ControlAuthenticationMiddleware,
        ControlLicenseMiddleware,
        ProtobufTranslationMiddleware,
        ControlEventLogMiddleware,
    ]


def _middleware_classes(app: FastAPI) -> list[type[object]]:
    """Return configured middleware classes in request execution order."""
    return [cast(type[object], middleware.cls) for middleware in app.user_middleware]


def test_create_app_mounts_core_health_router(monkeypatch: MonkeyPatch) -> None:
    """Mount the health route from SuperCore without a readiness route."""
    app = _create_app(monkeypatch)

    health_routes = [
        route_context.route
        for route_context in iter_route_contexts(app.routes)
        if isinstance(route_context.route, APIRoute)
        and route_context.path_format == "/health"
    ]

    assert len(health_routes) == 1
    assert health_routes[0].endpoint is health
    assert health_routes[0].methods == {"GET", "HEAD"}
    assert health_routes[0].tags == ["Health"]
    assert all(
        route_context.path_format != "/ready"
        for route_context in iter_route_contexts(app.routes)
    )


def test_create_app_constructs_control_middleware_in_execution_order(
    monkeypatch: MonkeyPatch,
) -> None:
    """Construct the complete Control middleware stack in one explicit order."""
    app = _create_app(monkeypatch)

    assert _middleware_classes(app) == _control_middleware_classes()
    assert app.user_middleware[0].kwargs["dispatch"] is http_error_translator


def test_create_app_places_extension_middleware_before_control_middleware(
    monkeypatch: MonkeyPatch,
) -> None:
    """Place outer extension middleware before the complete Control stack."""
    app = _create_app(
        monkeypatch,
        extension_middleware=(Middleware(_ExtensionMiddleware),),
    )

    assert _middleware_classes(app) == [
        _ExtensionMiddleware,
        *_control_middleware_classes(),
    ]
