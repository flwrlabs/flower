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

import pytest
from fastapi import APIRouter, Depends, FastAPI, Response
from fastapi.testclient import TestClient

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.supercore.error import ApiErrorCode, http_error_translator
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.routing import ProtobufRouter


def _create_app() -> FastAPI:
    """Create a FastAPI app with Flower error translation enabled."""
    app = FastAPI()
    app.middleware("http")(http_error_translator)
    return app


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

    @protobuf_router.unary_unary("/list-runs")
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
        "/list-runs",
        content=ListRunsRequest(run_id=7).SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )
    proto_response = ListRunsResponse()
    proto_response.ParseFromString(response.content)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(PROTOBUF_MEDIA_TYPE)
    assert seen_run_ids == [7]
    assert proto_response.now == "10"


def test_unary_unary_preserves_dependency_response_headers() -> None:
    """The router preserves headers set by FastAPI dependencies."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)

    def set_refreshed_tokens(response: Response) -> None:
        response.headers["x-access-token"] = "new-access-token"

    @protobuf_router.unary_unary("/list-runs")
    def list_runs(
        _request: ListRunsRequest,
        _: Annotated[None, Depends(set_refreshed_tokens)],
    ) -> ListRunsResponse:
        return ListRunsResponse()

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 200
    assert response.headers["x-access-token"] == "new-access-token"


def test_unary_unary_rejects_http_response_dependency_parameter() -> None:
    """The router should reject the response parameter reserved by its wrapper."""
    protobuf_router = ProtobufRouter(APIRouter())

    with pytest.raises(
        TypeError,
        match="list_runs dependency parameter 'http_response' is reserved",
    ):

        @protobuf_router.unary_unary("/list-runs")
        def list_runs(
            _request: ListRunsRequest,
            http_response: Annotated[None, Depends(lambda: None)],
        ) -> ListRunsResponse:
            del http_response
            return ListRunsResponse()


def test_sync_stream_handler_runs_in_threadpool() -> None:
    """The router creates synchronous streams outside the event-loop thread."""
    app = FastAPI()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)
    handler_thread_ids: list[int] = []

    async def get_event_loop_thread_id() -> int:
        return get_ident()

    def set_refreshed_tokens(response: Response) -> None:
        response.headers["x-access-token"] = "new-access-token"

    @protobuf_router.unary_stream("/rpc/StreamLogs")
    def stream_logs(
        _request: StreamLogsRequest,
        event_loop_thread_id: Annotated[int, Depends(get_event_loop_thread_id)],
        _: Annotated[None, Depends(set_refreshed_tokens)],
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
    assert response.headers["x-access-token"] == "new-access-token"
    assert handler_thread_ids


def test_unary_unary_rejects_non_protobuf_response() -> None:
    """The router reports a clear error for invalid unary response values."""
    app = _create_app()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)

    @protobuf_router.unary_unary("/list-runs")
    def list_runs(_request: ListRunsRequest) -> ListRunsResponse:
        return cast(ListRunsResponse, object())

    app.include_router(fastapi_router)
    client = TestClient(app)

    response = client.post(
        "/list-runs",
        content=ListRunsRequest().SerializeToString(),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
    )

    assert response.status_code == 500
    assert response.json()["code"] == ApiErrorCode.INVALID_HANDLER_RESPONSE


def test_unary_stream_rejects_non_iterable_response() -> None:
    """The router reports a clear error for invalid stream response values."""
    app = _create_app()
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
    assert response.json()["code"] == ApiErrorCode.INVALID_HANDLER_RESPONSE


@pytest.mark.parametrize(
    ("content", "headers", "error_code", "status_code"),
    [
        (
            ListRunsRequest().SerializeToString(),
            {"content-type": "application/json"},
            ApiErrorCode.UNSUPPORTED_CONTENT_TYPE,
            415,
        ),
        (
            b"\x80",
            {"content-type": PROTOBUF_MEDIA_TYPE},
            ApiErrorCode.INVALID_PROTOBUF_PAYLOAD,
            400,
        ),
    ],
)
def test_unary_unary_rejects_invalid_requests_with_flower_error(
    content: bytes,
    headers: dict[str, str],
    error_code: ApiErrorCode,
    status_code: int,
) -> None:
    """The router translates invalid requests into catalogued Flower errors."""
    app = _create_app()
    fastapi_router = APIRouter()
    protobuf_router = ProtobufRouter(fastapi_router)

    @protobuf_router.unary_unary("/list-runs")
    def list_runs(_request: ListRunsRequest) -> ListRunsResponse:
        return ListRunsResponse()

    app.include_router(fastapi_router)
    response = TestClient(app).post("/list-runs", content=content, headers=headers)

    assert response.status_code == status_code
    assert response.json()["code"] == error_code


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
