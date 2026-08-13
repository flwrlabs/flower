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
"""Reusable protobuf-over-HTTP client infrastructure."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self, TypeVar

import httpx
from google.protobuf.message import DecodeError, Message

from .constants import PROTOBUF_MEDIA_TYPE

ResponseT = TypeVar("ResponseT", bound=Message)


@dataclass(frozen=True)
class ProtobufRequestContext:
    """Context available to protobuf HTTP client interceptors."""

    rpc_method: str
    message: Message
    request: httpx.Request


ProtobufCall = Callable[[ProtobufRequestContext], httpx.Response]


class ProtobufClientInterceptor(Protocol):
    """Intercept a protobuf-over-HTTP request and response."""

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Process a request around the next interceptor or HTTP transport."""


def _wrap_interceptor(
    interceptor: ProtobufClientInterceptor,
    call_next: ProtobufCall,
) -> ProtobufCall:
    """Wrap one interceptor around the next call in the chain."""

    def call(context: ProtobufRequestContext) -> httpx.Response:
        return interceptor.intercept(context, call_next)

    return call


class ProtobufClient:
    """Client providing shared protobuf-over-HTTP request handling."""

    def __init__(
        self,
        base_url: str,
        *,
        interceptors: Sequence[ProtobufClientInterceptor] = (),
        verify: bool | str = True,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._interceptors = tuple(interceptors)
        self._client = httpx.Client(
            verify=verify,
            timeout=timeout,
            follow_redirects=True,
        )

    def _unary_unary(
        self,
        *,
        path: str,
        rpc_method: str,
        request: Message,
        response_type: type[ResponseT],
    ) -> ResponseT:
        """Send a unary request and parse its unary protobuf response."""
        path = path if path.startswith("/") else f"/{path}"
        http_request = self._client.build_request(
            method="POST",
            url=f"{self._base_url}{path}",
            content=request.SerializeToString(deterministic=True),
            headers={
                "content-type": PROTOBUF_MEDIA_TYPE,
                "accept": PROTOBUF_MEDIA_TYPE,
            },
        )
        context = ProtobufRequestContext(
            rpc_method=rpc_method,
            message=request,
            request=http_request,
        )
        response = self._send(context)
        if response.is_error:
            response.raise_for_status()

        result = response_type()
        try:
            result.ParseFromString(response.content)
        except DecodeError as exc:
            raise ValueError("Invalid protobuf response payload") from exc
        return result

    def _send(self, context: ProtobufRequestContext) -> httpx.Response:
        """Send a request through the configured interceptor chain."""

        def send(current_context: ProtobufRequestContext) -> httpx.Response:
            return self._client.send(current_context.request)

        call_next: ProtobufCall = send
        for interceptor in reversed(self._interceptors):
            call_next = _wrap_interceptor(interceptor, call_next)

        return call_next(context)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> Self:
        """Return this client from a context manager."""
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Close this client when leaving a context manager."""
        self.close()
