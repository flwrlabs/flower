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
from google.protobuf.message import Message
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import UNAUTHENTICATED_PATHS
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.config_loader import get_license_plugin
from flwr.superlink.dependencies.account import AccountAccessDependency


class ControlEventLogMiddleware(BaseHTTPMiddleware):
    """Write event logs around Control API handler calls."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Write events before and after a Control handler call."""
        # Event logging is optional and only applies after the translation middleware
        # has parsed a recognized Control API protobuf request.
        event_log_plugin: EventLogWriterPlugin | None = getattr(
            request.app.state, "control_event_log_plugin", None
        )
        protobuf_request = getattr(request.state, "protobuf_request", None)
        if event_log_plugin is None or not isinstance(protobuf_request, Message):
            return await call_next(request)

        # Authentication runs before event logging and stores the account, except for
        # unauthenticated Control routes where the actor remains unknown.
        account_info = getattr(request.state, "account", None)
        if not isinstance(account_info, AccountInfo):
            account_info = None

        def write_before_event() -> None:
            """Compose and write the event preceding handler execution."""
            event_log_plugin.write_log(
                event_log_plugin.compose_log_before_event(
                    request=protobuf_request,
                    context=request,
                    account_info=account_info,
                    method_name=request.url.path,
                )
            )

        def write_after_event(
            result: Message | BaseException | None,
        ) -> None:
            """Compose and write the event following handler execution."""
            event_log_plugin.write_log(
                event_log_plugin.compose_log_after_event(
                    request=protobuf_request,
                    context=request,
                    account_info=account_info,
                    method_name=request.url.path,
                    response=result,
                )
            )

        # Record the request before invoking the Control handler. Plugin work is
        # synchronous and may perform I/O, so keep it off the event loop.
        await run_in_threadpool(write_before_event)
        try:
            response = await call_next(request)
        except BaseException as exc:
            # Match the interceptor by recording handler failures before propagating
            # them to the outer error-translation middleware.
            await run_in_threadpool(write_after_event, exc)
            raise

        result = getattr(request.state, "protobuf_response", None)
        # A protobuf Message is a unary response and must be checked before the
        # iterable protocols, following ProtobufTranslationMiddleware's dispatch.
        if isinstance(result, Message):
            await run_in_threadpool(write_after_event, result)
        else:
            # Not yet implemented
            pass

        return response


def _is_control_path(path: str) -> bool:
    """Return whether the path belongs to a Control API endpoint."""
    return path.startswith("/control/")


class ControlLicenseMiddleware(BaseHTTPMiddleware):
    """Check Control API licenses when a license plugin is available."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._license_plugin = get_license_plugin()

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Skip checks without a plugin and reject requests with an invalid license."""
        if self._license_plugin is None or not _is_control_path(request.url.path):
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
            not _is_control_path(request.url.path)
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
