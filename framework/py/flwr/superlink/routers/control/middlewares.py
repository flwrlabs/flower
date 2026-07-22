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
"""Middleware for the Control API."""


from fastapi import Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from flwr.supercore.constant import UNAUTHENTICATED_PATHS
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.config_loader import get_license_plugin
from flwr.superlink.dependencies.account import AccountAccessDependency


class ControlLicenseMiddleware(BaseHTTPMiddleware):
    """Check Control API licenses when a license plugin is available."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._license_plugin = get_license_plugin()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Skip checks without a plugin and reject requests with an invalid license."""
        if self._license_plugin is None or not request.url.path.startswith("/control/"):
            return await call_next(request)

        if not await run_in_threadpool(self._license_plugin.check_license):
            raise FlowerError(
                ApiErrorCode.LICENSE_CHECK_FAILED,
                "License check failed.",
            )

        return await call_next(request)


class ControlAuthenticationMiddleware(BaseHTTPMiddleware):
    """Authenticate configured Control API routes before their handlers run."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Authenticate the request and preserve any refreshed token headers."""
        if (
            not request.url.path.startswith("/control")
            or request.url.path in UNAUTHENTICATED_PATHS
        ):
            return await call_next(request)

        account_access = getattr(request.app.state, "account_access_dep", None)
        if not isinstance(account_access, AccountAccessDependency):
            raise FlowerError(
                ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED,
                "SuperLink account authentication is not initialized: expected "
                f"AccountAccessDependency, got {type(account_access).__name__}.",
            )

        authentication_response = Response()
        # ``Response`` adds a default Content-Length header. This temporary
        # response only collects refreshed token headers, so it must not affect
        # the protobuf response returned by the endpoint.
        authentication_response.headers.raw.clear()
        request.state.account = account_access(request, authentication_response)
        response = await call_next(request)
        response.headers.raw.extend(authentication_response.headers.raw)
        return response
