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
"""Middleware and route helpers for the Control API."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast, get_type_hints

from fastapi import Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from google.protobuf.message import DecodeError, Message
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.superlink.dependencies.account import AccountAccessDependency

RouteKey = tuple[str, str]


class ControlAuthenticationMiddleware(BaseHTTPMiddleware):
    """Authenticate configured Control API routes before their handlers run."""

    def __init__(self, app: ASGIApp, authenticated_paths: set[str]) -> None:
        super().__init__(app)
        self._authenticated_paths = authenticated_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint  # type: ignore[type-arg]
    ) -> Response:
        """Authenticate the request and preserve any refreshed token headers."""
        if request.url.path not in self._authenticated_paths:
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


class ProtobufTranslationMiddleware(BaseHTTPMiddleware):
    """Deserialize configured protobuf request bodies before handlers run."""

    def __init__(
        self,
        app: ASGIApp,
        request_types: Mapping[RouteKey, type[Message]],
    ) -> None:
        super().__init__(app)
        self._request_types = request_types

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint  # type: ignore[type-arg]
    ) -> Response:
        """Parse the protobuf request body and make it available to dependencies."""
        request_type = self._request_types.get((request.method, request.url.path))
        if request_type is not None:
            self._check_request_media_type(request)
            request.state.protobuf_request = self._parse_request(
                await request.body(), request_type
            )
        return await call_next(request)

    @staticmethod
    def _check_request_media_type(request: Request) -> None:  # type: ignore[type-arg]
        content_type = request.headers.get("content-type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != PROTOBUF_MEDIA_TYPE:
            raise FlowerError(
                ApiErrorCode.UNSUPPORTED_CONTENT_TYPE,
                f"Unsupported Content-Type: {content_type!r}",
            )

    @staticmethod
    def _parse_request(body: bytes, request_type: type[Message]) -> Message:
        message = request_type()
        try:
            message.ParseFromString(body)
        except DecodeError as exc:
            raise FlowerError(
                ApiErrorCode.INVALID_PROTOBUF_PAYLOAD,
                f"Invalid protobuf payload: {exc!r}",
            ) from exc
        return message


class ProtobufRoute(APIRoute):
    """Serialize protobuf messages returned directly by endpoint handlers."""

    def __init__(
        self,
        path: str,
        endpoint: Callable[..., object],
        **kwargs: Any,
    ) -> None:
        async def protobuf_endpoint(*args: Any, **endpoint_kwargs: Any) -> Response:
            if inspect.iscoroutinefunction(endpoint):
                result = await cast(
                    Awaitable[object], endpoint(*args, **endpoint_kwargs)
                )
            else:
                result = await run_in_threadpool(endpoint, *args, **endpoint_kwargs)

            if not isinstance(result, Message):
                raise FlowerError(
                    ApiErrorCode.INVALID_HANDLER_RESPONSE,
                    "Invalid response returned from Control handler: "
                    f"{result!r} ({type(result).__name__})",
                )
            return Response(
                content=result.SerializeToString(), media_type=PROTOBUF_MEDIA_TYPE
            )

        protobuf_endpoint.__name__ = endpoint.__name__
        endpoint_signature = inspect.signature(endpoint)
        endpoint_hints = get_type_hints(endpoint, include_extras=True)
        protobuf_signature = endpoint_signature.replace(
            parameters=[
                parameter.replace(
                    annotation=endpoint_hints.get(parameter.name, parameter.annotation)
                )
                for parameter in endpoint_signature.parameters.values()
            ],
            return_annotation=Response,
        )
        protobuf_endpoint.__signature__ = protobuf_signature  # type: ignore[attr-defined]
        super().__init__(path, protobuf_endpoint, **kwargs)
