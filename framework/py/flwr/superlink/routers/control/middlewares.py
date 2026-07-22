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
from collections.abc import Awaitable, Callable
from typing import Any, cast, get_type_hints

from fastapi import Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from flwr.supercore.constant import UNAUTHENTICATED_PATHS
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.translation import PROTOBUF_REQUEST_TYPES
from flwr.superlink.dependencies.account import AccountAccessDependency

_HTTP_REQUEST_PARAMETER = "_protobuf_http_request"


class ControlAuthenticationMiddleware(BaseHTTPMiddleware):
    """Authenticate configured Control API routes before their handlers run."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Authenticate the request and preserve any refreshed token headers."""
        route_key = (request.method, request.url.path)
        if (
            route_key not in PROTOBUF_REQUEST_TYPES
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


class ProtobufRoute(APIRoute):
    """Make protobuf handler results available to the translation middleware.

    A Control handler returns one protobuf ``Message`` for a unary RPC, or a
    synchronous iterable of messages for a unary-stream RPC. This route stores
    that result in the shared request state and returns an empty HTTP response.
    ``ProtobufTranslationMiddleware`` serializes the stored result after inner
    response-side middleware has run.
    """

    def __init__(
        self,
        path: str,
        endpoint: Callable[..., object],
        **kwargs: Any,
    ) -> None:
        async def protobuf_endpoint(*args: Any, **endpoint_kwargs: Any) -> Response:
            # The signature installed below asks FastAPI to inject the HTTP request
            # for shared state access. Remove it before calling the original handler.
            http_request = cast(Request, endpoint_kwargs.pop(_HTTP_REQUEST_PARAMETER))
            # Invoke async handlers on the event loop and synchronous handlers in a
            # worker thread, matching FastAPI's usual execution model.
            result: object
            if inspect.iscoroutinefunction(endpoint):
                result = endpoint(*args, **endpoint_kwargs)
            else:
                result = await run_in_threadpool(endpoint, *args, **endpoint_kwargs)

            # Resolve any awaitable result before storing it in the request state.
            if inspect.isawaitable(result):
                result = await cast(Awaitable[object], result)

            http_request.state.protobuf_response = result
            return Response()

        # Retain the original name in route metadata, OpenAPI operation IDs,
        # exceptions, and logs instead of exposing the internal wrapper name.
        protobuf_endpoint.__name__ = endpoint.__name__

        # FastAPI uses a callable's signature—not its call site—to discover
        # dependencies. Resolve annotations in the endpoint's module because
        # postponed annotations cannot be resolved from this middleware module.
        endpoint_signature = inspect.signature(endpoint)
        endpoint_hints = get_type_hints(endpoint, include_extras=True)
        if _HTTP_REQUEST_PARAMETER in endpoint_signature.parameters:
            raise TypeError(
                f"{endpoint.__name__} parameter {_HTTP_REQUEST_PARAMETER!r} is reserved"
            )

        # Do not use functools.wraps. It would expose a generator return type to
        # FastAPI, which could then serialize it before this route sees it. Keep
        # the original dependency parameters, inject the HTTP request for shared
        # state access, and declare the wrapper's concrete Response return type.
        parameters = [
            parameter.replace(
                annotation=endpoint_hints.get(parameter.name, parameter.annotation)
            )
            for parameter in endpoint_signature.parameters.values()
        ]
        # Inject the HTTP request without exposing it to the original handler.
        http_request_parameter = inspect.Parameter(
            _HTTP_REQUEST_PARAMETER,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=Request,
        )
        # A keyword-only parameter must precede **kwargs in a valid signature.
        variadic_keyword_index = next(
            (
                index
                for index, parameter in enumerate(parameters)
                if parameter.kind is inspect.Parameter.VAR_KEYWORD
            ),
            len(parameters),
        )
        parameters.insert(variadic_keyword_index, http_request_parameter)
        # Tell FastAPI that this wrapper returns an HTTP response.
        protobuf_signature = endpoint_signature.replace(
            parameters=parameters,
            return_annotation=Response,
        )
        protobuf_endpoint.__signature__ = protobuf_signature  # type: ignore[attr-defined]
        super().__init__(path, protobuf_endpoint, **kwargs)
