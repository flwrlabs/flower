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
"""Tests for protobuf-over-HTTP stream cancellation helpers."""

import asyncio
import time
from collections.abc import Iterator

from fastapi import Request
from starlette.types import Message

from .streaming import (
    CancellableProtobufStreamingResponse,
    ProtobufStreamContext,
    get_protobuf_stream_context,
    peek_protobuf_stream_context,
)


def _request() -> Request:
    """Create a minimal HTTP request."""
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/control/stream-logs",
            "headers": [],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
        }
    )


def test_get_stream_context_reuses_request_scoped_instance() -> None:
    """Return the same stream context throughout one request."""
    request = _request()

    first = get_protobuf_stream_context(request)
    second = get_protobuf_stream_context(request)

    assert first is second
    assert peek_protobuf_stream_context(request) is first
    assert first.is_active()


def test_response_cancels_context_after_normal_completion() -> None:
    """Mark the context inactive after the response body is exhausted."""
    context = ProtobufStreamContext()
    response = CancellableProtobufStreamingResponse(
        iter([b"frame"]), context, "application/flower-protobuf-stream"
    )
    sent: list[Message] = []

    async def run_response() -> None:
        async def receive() -> Message:
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def send(message: Message) -> None:
            sent.append(message)

        await response(
            {"type": "http", "asgi": {"spec_version": "2.0"}},
            receive,
            send,
        )

    asyncio.run(run_response())

    assert not context.is_active()
    assert sent[-1] == {
        "type": "http.response.body",
        "body": b"",
        "more_body": False,
    }


def test_disconnect_unblocks_synchronous_stream_iteration() -> None:
    """Signal a blocked synchronous iterator as soon as the client disconnects."""
    context = ProtobufStreamContext()

    def content() -> Iterator[bytes]:
        while context.is_active():
            time.sleep(0.001)
        yield from ()

    response = CancellableProtobufStreamingResponse(
        content(), context, "application/flower-protobuf-stream"
    )

    async def run_response() -> None:
        async def receive() -> Message:
            return {"type": "http.disconnect"}

        async def send(_message: Message) -> None:
            return None

        await asyncio.wait_for(
            response(
                {"type": "http", "asgi": {"spec_version": "2.0"}},
                receive,
                send,
            ),
            timeout=1.0,
        )

    asyncio.run(run_response())

    assert not context.is_active()
