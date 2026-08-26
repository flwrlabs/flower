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
"""Cancellation helpers for protobuf-over-HTTP streams."""

from __future__ import annotations

from collections.abc import Iterable
from threading import Event

from fastapi import Request
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

_STREAM_CONTEXT_STATE_KEY = "protobuf_stream_context"


class ProtobufStreamContext:
    """Track whether a protobuf-over-HTTP stream is still active."""

    def __init__(self) -> None:
        self._cancelled = Event()

    def is_active(self) -> bool:
        """Return whether the client can still consume the stream."""
        return not self._cancelled.is_set()

    def cancel(self) -> None:
        """Mark the stream as cancelled."""
        self._cancelled.set()


def get_protobuf_stream_context(request: Request) -> ProtobufStreamContext:
    """Return the request-scoped protobuf stream context."""
    stream_context = getattr(request.state, _STREAM_CONTEXT_STATE_KEY, None)
    if not isinstance(stream_context, ProtobufStreamContext):
        stream_context = ProtobufStreamContext()
        setattr(request.state, _STREAM_CONTEXT_STATE_KEY, stream_context)
    return stream_context


def peek_protobuf_stream_context(request: Request) -> ProtobufStreamContext | None:
    """Return an existing request-scoped stream context, if one was created."""
    stream_context = getattr(request.state, _STREAM_CONTEXT_STATE_KEY, None)
    return stream_context if isinstance(stream_context, ProtobufStreamContext) else None


class CancellableProtobufStreamingResponse(StreamingResponse):
    """Cancel the associated stream context when the response stops."""

    def __init__(
        self,
        content: Iterable[bytes],
        stream_context: ProtobufStreamContext,
        media_type: str,
    ) -> None:
        super().__init__(content=content, media_type=media_type)
        self._stream_context = stream_context

    async def listen_for_disconnect(self, receive: Receive) -> None:
        """Cancel the stream immediately after receiving an HTTP disconnect."""
        try:
            await super().listen_for_disconnect(receive)
        finally:
            self._stream_context.cancel()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve the response and cancel the context during final cleanup."""
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._stream_context.cancel()
