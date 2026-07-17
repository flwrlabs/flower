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
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast, get_type_hints

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from fastapi.routing import APIRoute
from google.protobuf.message import Message
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.framing import frame_message
from flwr.superlink.dependencies.account import AccountAccessDependency


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


class ProtobufRoute(APIRoute):
    """Translate protobuf handler results into concrete HTTP responses.

    A Control handler returns one protobuf ``Message`` for a unary RPC, or a
    synchronous iterable of messages for a unary-stream RPC. FastAPI normally
    attempts to serialize these return values as JSON. This route intercepts
    the result first, serializes unary messages as ``application/protobuf``,
    and frames stream items as ``application/flower-protobuf-stream``.
    """

    def __init__(
        self,
        path: str,
        endpoint: Callable[..., object],
        **kwargs: Any,
    ) -> None:
        async def protobuf_endpoint(*args: Any, **endpoint_kwargs: Any) -> Response:
            # FastAPI resolves the original endpoint's dependencies using the
            # signature installed below, then calls this wrapper. Invoke async
            # handlers on the event loop and synchronous handlers in a worker
            # thread, matching FastAPI's usual execution model.
            result: object
            if inspect.iscoroutinefunction(endpoint):
                result = endpoint(*args, **endpoint_kwargs)
            else:
                result = await run_in_threadpool(endpoint, *args, **endpoint_kwargs)

            # A coroutine endpoint returns an awaitable. Resolve it before
            # choosing the protobuf response representation below.
            if inspect.isawaitable(result):
                result = await cast(Awaitable[object], result)

            return self._response_for(result)

        # Retain the original name in route metadata, OpenAPI operation IDs,
        # exceptions, and logs instead of exposing the internal wrapper name.
        protobuf_endpoint.__name__ = endpoint.__name__

        # FastAPI uses a callable's signature—not its call site—to discover
        # dependencies. Resolve annotations in the endpoint's module because
        # postponed annotations cannot be resolved from this middleware module.
        endpoint_signature = inspect.signature(endpoint)
        endpoint_hints = get_type_hints(endpoint, include_extras=True)

        # Do not use functools.wraps. It would expose a generator return type to
        # FastAPI, which could then serialize it before this route sees it. Keep
        # the original parameters for dependency injection, but declare a
        # concrete Response return type so this class owns protobuf serialization.
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

    @staticmethod
    def _response_for(result: object) -> Response:
        """Return the HTTP response matching a protobuf handler result."""
        # ``Message`` is also the most specific contract and must be checked
        # first. Unary responses are not framed; framing is reserved for streams.
        if isinstance(result, Message):
            return Response(
                content=result.SerializeToString(), media_type=PROTOBUF_MEDIA_TYPE
            )

        # Synchronous generators and other iterables are streamed lazily too.
        # Starlette advances a synchronous iterator outside the event loop.
        if isinstance(result, Iterable):
            return StreamingResponse(
                (frame_message(message) for message in cast(Iterable[Message], result)),
                media_type=PROTOBUF_STREAM_MEDIA_TYPE,
            )

        raise FlowerError(
            ApiErrorCode.INVALID_HANDLER_RESPONSE,
            "Invalid response returned from Control handler: expected a protobuf "
            "Message or Iterable[Message], got "
            f"{result!r} ({type(result).__name__})",
        )
