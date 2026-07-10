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
"""Tests for protobuf FastAPI routing helpers."""


from collections.abc import AsyncIterator, Iterator
from threading import get_ident
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI
from fastapi.testclient import TestClient

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.routing import ProtobufRouter


def test_unary_unary_parses_and_returns_protobuf() -> None:
    """The router parses protobuf requests and preserves FastAPI dependencies."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)
    seen_run_ids: list[int] = []
    handler_thread_ids: list[int] = []

    def get_limit() -> int:
        return 10

    async def get_event_loop_thread_id() -> int:
        return get_ident()

    @protobuf_router.unary_unary("/rpc/ListRuns")
    def list_runs(
        request: ListRunsRequest,
        limit: Annotated[int, Depends(get_limit)],
        event_loop_thread_id: Annotated[int, Depends(get_event_loop_thread_id)],
    ) -> ListRunsResponse:
        seen_run_ids.append(request.run_id)
        handler_thread_ids.append(get_ident())
        assert handler_thread_ids[-1] != event_loop_thread_id
        return ListRunsResponse(now=str(limit))

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/rpc/ListRuns",
        content=ListRunsRequest(run_id=7).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    proto_response = ListRunsResponse()
    proto_response.ParseFromString(response.content)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(PROTOBUF_MEDIA_TYPE)
    assert seen_run_ids == [7]
    assert proto_response.now == "10"


def test_sync_stream_handler_runs_in_threadpool() -> None:
    """The router creates synchronous streams outside the event-loop thread."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)
    handler_thread_ids: list[int] = []

    async def get_event_loop_thread_id() -> int:
        return get_ident()

    @protobuf_router.unary_stream("/rpc/StreamLogs")
    def stream_logs(
        _request: StreamLogsRequest,
        event_loop_thread_id: Annotated[int, Depends(get_event_loop_thread_id)],
    ) -> Iterator[StreamLogsResponse]:
        handler_thread_ids.append(get_ident())
        assert handler_thread_ids[-1] != event_loop_thread_id
        return iter([StreamLogsResponse(log_output="done")])

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/rpc/StreamLogs",
        content=StreamLogsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert handler_thread_ids


def test_unary_unary_rejects_non_protobuf_response() -> None:
    """The router reports a clear error for invalid unary response values."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)

    @protobuf_router.unary_unary("/rpc/ListRuns")
    def list_runs(_request: ListRunsRequest) -> ListRunsResponse:
        return cast(ListRunsResponse, object())

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/rpc/ListRuns",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == ("Invalid response returned from unary handler")


def test_unary_stream_rejects_non_iterable_response() -> None:
    """The router reports a clear error for invalid stream response values."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)

    @protobuf_router.unary_stream("/rpc/StreamLogs")
    def stream_logs(_request: StreamLogsRequest) -> Iterator[StreamLogsResponse]:
        return cast(Iterator[StreamLogsResponse], None)

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/rpc/StreamLogs",
        content=StreamLogsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Invalid response returned from stream handler"
    )


def test_unary_stream_returns_framed_protobuf_stream() -> None:
    """The router frames every protobuf message in a streamed response."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)

    @protobuf_router.unary_stream("/rpc/StreamLogs")
    async def stream_logs(
        request: StreamLogsRequest,
    ) -> AsyncIterator[StreamLogsResponse]:
        yield StreamLogsResponse(log_output=str(request.run_id))
        yield StreamLogsResponse(log_output="done")

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/rpc/StreamLogs",
        content=StreamLogsRequest(run_id=7).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    frames = _parse_frames(response.content)
    messages = [StreamLogsResponse.FromString(frame) for frame in frames]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(PROTOBUF_STREAM_MEDIA_TYPE)
    assert [message.log_output for message in messages] == ["7", "done"]


def _parse_frames(content: bytes) -> list[bytes]:
    """Return protobuf payloads from a concatenated four-byte-length stream."""
    frames = []
    offset = 0
    while offset < len(content):
        payload_size = int.from_bytes(content[offset : offset + 4], "big")
        offset += 4
        frames.append(content[offset : offset + payload_size])
        offset += payload_size
    return frames
