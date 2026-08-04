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
"""FastAPI route helpers for protobuf RPC APIs."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast, get_type_hints

from fastapi import Request
from fastapi.responses import Response
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool

_HTTP_REQUEST_PARAMETER = "_protobuf_http_request"


class ProtobufRoute(APIRoute):
    """Capture protobuf handler results for the translation middleware.

    Handlers return a protobuf ``Message`` for unary RPCs or a synchronous
    iterable of messages for unary-stream RPCs. The wrapper stores that result
    in the shared request state and returns an empty placeholder response.
    ``ProtobufTranslationMiddleware`` later serializes the stored result.
    """

    def __init__(
        self,
        path: str,
        endpoint: Callable[..., object],
        **kwargs: Any,
    ) -> None:
        async def protobuf_endpoint(*args: Any, **endpoint_kwargs: Any) -> Response:
            # Remove the injected HTTP request before calling the original handler.
            http_request = cast(Request, endpoint_kwargs.pop(_HTTP_REQUEST_PARAMETER))
            # Match FastAPI's execution model for async and synchronous handlers.
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

        # Retain the handler name in route metadata, operation IDs, and logs.
        protobuf_endpoint.__name__ = endpoint.__name__

        # Resolve postponed annotations in the original handler's module.
        endpoint_signature = inspect.signature(endpoint)
        endpoint_hints = get_type_hints(endpoint, include_extras=True)
        if _HTTP_REQUEST_PARAMETER in endpoint_signature.parameters:
            raise TypeError(
                f"{endpoint.__name__} parameter {_HTTP_REQUEST_PARAMETER!r} is reserved"
            )

        # Build the signature FastAPI uses for dependency injection. Preserve the
        # handler parameters, inject the HTTP request for state access, and advertise
        # the placeholder Response instead of the handler's protobuf return type.
        parameters = [
            parameter.replace(
                annotation=endpoint_hints.get(parameter.name, parameter.annotation)
            )
            for parameter in endpoint_signature.parameters.values()
        ]
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
        protobuf_signature = endpoint_signature.replace(
            parameters=parameters,
            return_annotation=Response,
        )
        protobuf_endpoint.__signature__ = protobuf_signature  # type: ignore[attr-defined]
        super().__init__(path, protobuf_endpoint, **kwargs)
