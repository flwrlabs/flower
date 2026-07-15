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
"""FastAPI routing helpers for protobuf RPC APIs."""


from __future__ import annotations

import inspect
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
)
from typing import TypeVar, cast, get_args, get_origin, get_type_hints

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from google.protobuf.message import DecodeError, Message
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import State

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.framing import async_iter_framed_bytes, frame_message

RequestT = TypeVar("RequestT", bound=Message)
ResponseT = TypeVar("ResponseT", bound=Message)
StreamT = TypeVar("StreamT", bound=Message)


def _check_request_media_type(request: Request[State]) -> None:
    """Ensure the request body uses the protobuf media type."""
    content_type = request.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type != PROTOBUF_MEDIA_TYPE:
        raise FlowerError(
            ApiErrorCode.UNSUPPORTED_CONTENT_TYPE,
            f"Unsupported Content-Type: {content_type!r}",
        )


def _request_type_and_dependency_parameters(
    func: Callable[..., object],
) -> tuple[type[RequestT], list[inspect.Parameter]]:
    """Get the protobuf request type and FastAPI dependency parameters."""
    parameters = list(inspect.signature(func).parameters.values())
    if not parameters or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(
            f"{func.__name__} must accept a protobuf request as its first "
            "positional parameter"
        )

    try:
        hints = get_type_hints(func, include_extras=True)
    except (NameError, TypeError) as exc:
        raise TypeError(f"{func.__name__} has unresolved protobuf annotations") from exc

    request_type = hints.get(parameters[0].name)
    if not isinstance(request_type, type) or not issubclass(request_type, Message):
        raise TypeError(
            f"{func.__name__} request parameter must be annotated with a protobuf "
            "Message type"
        )

    dependency_parameters = [
        parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
        for parameter in parameters[1:]
    ]
    for parameter in dependency_parameters:
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"{func.__name__} dependency parameter {parameter.name!r} must be "
                "positional-or-keyword or keyword-only"
            )
        if parameter.name in {"http_request", "http_response"}:
            # The generated endpoint signature reserves these names for FastAPI.
            raise TypeError(
                f"{func.__name__} dependency parameter {parameter.name!r} is reserved"
            )

    return cast(type[RequestT], request_type), dependency_parameters


def _message_response_type(func: Callable[..., object]) -> type[ResponseT]:
    """Validate and return a unary protobuf response type."""
    try:
        hints = get_type_hints(func)
    except (NameError, TypeError) as exc:
        raise TypeError(f"{func.__name__} has unresolved protobuf annotations") from exc

    response_type = hints.get("return")
    if not isinstance(response_type, type) or not issubclass(response_type, Message):
        raise TypeError(
            f"{func.__name__} return value must be annotated with a protobuf "
            "Message type"
        )

    return cast(type[ResponseT], response_type)


def _stream_response_type(func: Callable[..., object]) -> type[StreamT]:
    """Validate and return a streamed protobuf response item type."""
    try:
        hints = get_type_hints(func)
    except (NameError, TypeError) as exc:
        raise TypeError(f"{func.__name__} has unresolved protobuf annotations") from exc

    return_type = hints.get("return")
    origin = get_origin(return_type)
    args = get_args(return_type)
    if (
        origin not in (AsyncIterable, AsyncIterator, Iterable, Iterator)
        or len(args) != 1
    ):
        raise TypeError(
            f"{func.__name__} return value must be annotated as AsyncIterable[T], "
            "AsyncIterator[T], Iterable[T], or Iterator[T]"
        )

    item_type = args[0]
    if not isinstance(item_type, type) or not issubclass(item_type, Message):
        raise TypeError(
            f"{func.__name__} stream item must be annotated with a protobuf "
            "Message type"
        )

    return cast(type[StreamT], item_type)


def _build_endpoint_signature(
    dependency_parameters: list[inspect.Parameter],
) -> inspect.Signature:
    """Build the signature FastAPI uses to resolve an endpoint's dependencies."""
    http_request_parameter = inspect.Parameter(
        "http_request",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=cast(type[Request[State]], Request),
    )
    # Dependencies use FastAPI's mutable response to set headers.
    http_response_parameter = inspect.Parameter(
        "http_response",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=Response,
    )
    keyword_dependency_parameters = [
        inspect.Parameter(
            parameter.name,
            inspect.Parameter.KEYWORD_ONLY,
            default=parameter.default,
            annotation=parameter.annotation,
        )
        for parameter in dependency_parameters
    ]
    return inspect.Signature(
        parameters=[
            http_request_parameter,
            http_response_parameter,
            *keyword_dependency_parameters,
        ],
        return_annotation=Response,
    )


def _parse_protobuf_body(body: bytes, message_type: type[RequestT]) -> RequestT:
    """Parse a serialized protobuf message from an HTTP request body."""
    message = message_type()
    try:
        message.ParseFromString(body)
    except DecodeError as exc:
        raise FlowerError(
            ApiErrorCode.INVALID_PROTOBUF_PAYLOAD,
            f"Invalid protobuf payload: {exc!r}",
        ) from exc
    return message


def _is_async_handler(func: Callable[..., object]) -> bool:
    """Return whether a handler executes on the event loop."""
    return inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func)


async def _call_handler(
    func: Callable[..., object],
    proto_request: Message,
    dependency_values: dict[str, object],
) -> object:
    """Call asynchronous handlers directly and synchronous handlers in a thread."""
    if _is_async_handler(func):
        result = func(proto_request, **dependency_values)
    else:
        result = await run_in_threadpool(func, proto_request, **dependency_values)
    if inspect.isawaitable(result):
        return await result
    return result


class ProtobufRouter:
    """Add protobuf RPC request and response handling to a FastAPI router."""

    def __init__(self, router: APIRouter) -> None:
        self.router = router

    def unary_unary(
        self,
        path: str,
    ) -> Callable[
        [Callable[..., ResponseT | Awaitable[ResponseT]]],
        Callable[..., Awaitable[Response]],
    ]:
        """Register an endpoint with one protobuf request and response."""

        def decorator(
            func: Callable[..., ResponseT | Awaitable[ResponseT]],
        ) -> Callable[..., Awaitable[Response]]:
            request_type_and_dependencies: tuple[
                type[Message], list[inspect.Parameter]
            ] = cast(
                tuple[type[Message], list[inspect.Parameter]],
                _request_type_and_dependency_parameters(func),
            )
            request_type = request_type_and_dependencies[0]
            dependency_parameters = request_type_and_dependencies[1]
            _message_response_type(func)

            async def wrapper(
                http_request: Request[State],
                http_response: Response,
                **dependency_values: object,
            ) -> Response:
                _check_request_media_type(http_request)
                proto_request = _parse_protobuf_body(
                    await http_request.body(), request_type
                )
                result = await _call_handler(func, proto_request, dependency_values)
                # Fail clearly when a handler violates its declared response contract.
                if not isinstance(result, Message):
                    raise FlowerError(
                        ApiErrorCode.INVALID_HANDLER_RESPONSE,
                        "Invalid response returned from unary handler: "
                        f"{result!r} ({type(result).__name__})",
                    )
                response = Response(
                    content=result.SerializeToString(),
                    media_type=PROTOBUF_MEDIA_TYPE,
                )
                # The wrapper replaces FastAPI's injected response, so preserve
                # headers written by dependencies, such as refreshed tokens.
                response.headers.raw.extend(http_response.headers.raw)
                return response

            wrapper.__name__ = func.__name__
            wrapper.__signature__ = (  # type: ignore[attr-defined]
                _build_endpoint_signature(dependency_parameters)
            )
            self.router.post(path)(wrapper)
            return wrapper

        return decorator

    def unary_stream(
        self,
        path: str,
    ) -> Callable[[Callable[..., object]], Callable[..., Awaitable[Response]]]:
        """Register an endpoint with one protobuf request and a response stream."""

        def decorator(
            func: Callable[..., object],
        ) -> Callable[..., Awaitable[Response]]:
            request_type_and_dependencies: tuple[
                type[Message], list[inspect.Parameter]
            ] = cast(
                tuple[type[Message], list[inspect.Parameter]],
                _request_type_and_dependency_parameters(func),
            )
            request_type = request_type_and_dependencies[0]
            dependency_parameters = request_type_and_dependencies[1]
            _stream_response_type(func)

            async def wrapper(
                http_request: Request[State],
                http_response: Response,
                **dependency_values: object,
            ) -> Response:
                _check_request_media_type(http_request)
                proto_request = _parse_protobuf_body(
                    await http_request.body(), request_type
                )
                result = await _call_handler(func, proto_request, dependency_values)

                content: AsyncIterable[bytes] | Iterable[bytes]
                # Select framing based on the stream type and reject invalid results.
                if isinstance(result, AsyncIterable):
                    content = async_iter_framed_bytes(
                        cast(AsyncIterable[Message], result)
                    )
                elif isinstance(result, Iterable):
                    content = (
                        frame_message(message)
                        for message in cast(Iterable[Message], result)
                    )
                else:
                    raise FlowerError(
                        ApiErrorCode.INVALID_HANDLER_RESPONSE,
                        "Invalid response returned from stream handler: "
                        f"{result!r} ({type(result).__name__})",
                    )

                response = StreamingResponse(
                    content,
                    media_type=PROTOBUF_STREAM_MEDIA_TYPE,
                )
                # The wrapper replaces FastAPI's injected response, so preserve
                # headers written by dependencies, such as refreshed tokens.
                response.headers.raw.extend(http_response.headers.raw)
                return response

            wrapper.__name__ = func.__name__
            wrapper.__signature__ = (  # type: ignore[attr-defined]
                _build_endpoint_signature(dependency_parameters)
            )
            self.router.post(path)(wrapper)
            return wrapper

        return decorator
