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

import ssl
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self, TypeVar

import httpx
from google.protobuf.message import DecodeError, Message

from .constants import (
    FRAME_HEADER_SIZE,
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)

if TYPE_CHECKING:
    from flwr.supercore.retry import RetryInvoker

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

    def __init__(  # pylint: disable=too-many-arguments
        self,
        base_url: str,
        *,
        interceptors: Sequence[ProtobufClientInterceptor] = (),
        verify: ssl.SSLContext | bool | str = True,
        timeout: float = 30.0,
        retry_invoker: RetryInvoker | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._interceptors = tuple(interceptors)
        self._retry_invoker = retry_invoker
        self._client = httpx.Client(
            verify=verify,
            timeout=timeout,
            follow_redirects=True,
        )

    @classmethod
    def from_server_address(  # pylint: disable=too-many-arguments
        cls,
        server_address: str,
        insecure: bool,
        root_certificates: bytes | str | None,
        interceptors: Sequence[ProtobufClientInterceptor],
        *,
        retry_invoker: RetryInvoker | None = None,
    ) -> Self:
        """Create a protobuf-over-HTTP client from a server address."""
        if insecure and root_certificates is not None:
            raise ValueError(
                "Invalid configuration: 'root_certificates' should not be provided "
                "when 'insecure' is set to True."
            )

        scheme = "http" if insecure else "https"
        verify: ssl.SSLContext | str | bool = not insecure
        if not insecure and root_certificates is not None:
            if isinstance(root_certificates, str):
                verify = root_certificates
            else:
                verify = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                verify.load_verify_locations(cadata=root_certificates.decode("ascii"))

        return cls(
            f"{scheme}://{server_address}",
            interceptors=interceptors,
            verify=verify,
            retry_invoker=retry_invoker,
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
        content = request.SerializeToString(deterministic=True)

        def send() -> httpx.Response:
            # Build a fresh request for every attempt so interceptors can safely
            # mutate it without leaking state into a retry.
            http_request = self._client.build_request(
                method="POST",
                url=f"{self._base_url}{path}",
                content=content,
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
            http_response = self._send(context)
            http_response.raise_for_status()
            return http_response

        response = (
            self._retry_invoker.invoke(send)
            if self._retry_invoker is not None
            else send()
        )

        result = response_type()
        try:
            result.ParseFromString(response.content)
        except DecodeError as exc:
            raise ValueError("Invalid protobuf response payload") from exc
        return result

    def _unary_stream(  # pylint: disable=too-many-arguments
        self,
        *,
        path: str,
        rpc_method: str,
        request: Message,
        response_type: type[ResponseT],
        timeout: float | None = None,
    ) -> Generator[ResponseT, None, None]:
        """Send a unary request and iterate over its streaming response."""
        path = path if path.startswith("/") else f"/{path}"
        content = request.SerializeToString(deterministic=True)

        def send() -> httpx.Response:
            http_request = self._client.build_request(
                method="POST",
                url=f"{self._base_url}{path}",
                content=content,
                headers={
                    "content-type": PROTOBUF_MEDIA_TYPE,
                    "accept": PROTOBUF_STREAM_MEDIA_TYPE,
                },
                timeout=self._client.timeout if timeout is None else timeout,
            )
            context = ProtobufRequestContext(
                rpc_method=rpc_method,
                message=request,
                request=http_request,
            )
            http_response = self._send(context, stream=True)
            try:
                # Validate only the response status here. The body remains open and
                # is consumed incrementally by `iter_messages` below.
                http_response.raise_for_status()
            except BaseException:
                # A failed attempt is never handed to the iterator, so release it
                # here before the retry invoker starts another attempt.
                http_response.close()
                raise
            return http_response

        def iter_messages() -> Generator[ResponseT, None, None]:
            # Open the response only when iteration starts so an unused stream does
            # not acquire a connection that the unstarted generator cannot release.
            response = (
                self._retry_invoker.invoke(send)
                if self._retry_invoker is not None
                else send()
            )
            buffer = bytearray()
            payload_size: int | None = None
            try:
                for chunk in response.iter_bytes():
                    buffer.extend(chunk)
                    while True:
                        # HTTP chunks do not necessarily align with frame boundaries.
                        # Wait until the four-byte payload size is available, then
                        # wait again until the complete payload has arrived.
                        if payload_size is None:
                            if len(buffer) < FRAME_HEADER_SIZE:
                                break
                            payload_size = int.from_bytes(
                                buffer[:FRAME_HEADER_SIZE], "big"
                            )
                            del buffer[:FRAME_HEADER_SIZE]

                        if len(buffer) < payload_size:
                            break

                        payload = bytes(buffer[:payload_size])
                        del buffer[:payload_size]
                        payload_size = None

                        result = response_type()
                        try:
                            result.ParseFromString(payload)
                        except DecodeError as exc:
                            raise ValueError("Invalid protobuf stream payload") from exc
                        yield result

                if payload_size is not None or buffer:
                    raise ValueError("Truncated protobuf stream frame")
            finally:
                # Closing the generator cancels the underlying HTTP stream too.
                response.close()

        return iter_messages()

    def _send(
        self, context: ProtobufRequestContext, *, stream: bool = False
    ) -> httpx.Response:
        """Send a request through the configured interceptor chain."""

        def send(current_context: ProtobufRequestContext) -> httpx.Response:
            if stream:
                return self._client.send(current_context.request, stream=True)
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
