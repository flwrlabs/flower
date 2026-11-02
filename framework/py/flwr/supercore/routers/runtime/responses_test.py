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
"""Tests for the Runtime Responses endpoint."""
# pylint: disable=redefined-outer-name

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock, Mock

import anyio
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError
from starlette.types import Message, Scope

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.dependencies.runtime import get_runtime_state
from flwr.supercore.typing import JSONObject

from . import responses
from .responses import ModelResponse, router


def _client(state: Mock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runtime_state] = lambda: state
    return TestClient(app)


def _state() -> Mock:
    state = Mock(spec=CoreState)
    state.get_task_by_token.return_value = Task(
        task_id=123, run_id=789, type=TaskType.AGENT_APP
    )
    return state


def _stream_request() -> JSONObject:
    """Create one normalized streaming model request payload."""
    return {"model": "model", "input": "hello", "stream": True}


@pytest.mark.parametrize("authorization", [None, "Basic task-token"])
def test_responses_requires_bearer_authentication(
    authorization: str | None,
) -> None:
    """Reject missing and non-Bearer task credentials."""
    headers = {"Authorization": authorization} if authorization else {}

    response = _client(_state()).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello"},
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


@pytest.mark.parametrize(
    "extra,code",
    [
        ({"temperature": 0.5}, "unsupported_parameter"),
        ({"stream": "true"}, "invalid_request"),
    ],
)
def test_responses_rejects_invalid_requests(extra: JSONObject, code: str) -> None:
    """Reject unsupported fields and invalid request types before inference."""
    response = _client(_state()).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello"} | extra,
        headers={"Authorization": "Bearer task-token"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == code


@pytest.fixture
def state(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Configure inference without changing task authentication."""
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://provider/v1/responses")
    monkeypatch.setenv("FLWR_MODEL_API_KEY", "provider-key")
    state = _state()
    state.get_tasks.return_value = [state.get_task_by_token.return_value]
    return state


def _mock_provider(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.AsyncBaseTransport,
) -> None:
    """Exercise real HTTP/SSE parsing with an in-memory transport."""
    client = httpx.AsyncClient
    monkeypatch.setattr(
        "flwr.supercore.model_provider.httpx.AsyncClient",
        lambda **kwargs: client(transport=transport, **kwargs),
    )


@pytest.mark.parametrize(
    "stream,key", [(False, "provider-key"), (True, "provider-key"), (False, "")]
)
def test_model_response_accounts_usage_and_preserves_output(
    state: Mock, monkeypatch: pytest.MonkeyPatch, stream: bool, key: str
) -> None:
    """Forward JSON/SSE unchanged and account usage against the AgentApp task."""
    backend = MagicMock()
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    monkeypatch.setattr("flwr.supercore.tracing._load_backend", lambda: backend)
    monkeypatch.setenv("FLWR_MODEL_API_KEY", key)
    body: JSONObject = {
        "object": "response",
        "output": [],
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    # Keep unknown frames intact, split UTF-8/CRLF, and account at EOF.
    wire = (
        ": heartbeat\r\n\r\nid: 42\r\nevent: custom\r\ndata: not json\r\n\r\n"
        'event: response.output_text.delta\r\ndata: {"delta":"café"}\r\n\r\n'
        f'event: response.completed\r\ndata: {json.dumps({"response": body})}'
    ).encode()

    class ByteStream(httpx.AsyncByteStream):
        """Split SSE at byte boundaries without changing its contents."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for value in wire:
                yield bytes([value])

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == (
            f"Bearer {key}" if key else None
        )
        if request.url.path == "/v1/responses":
            return httpx.Response(307, headers={"location": "/v1/model-responses"})
        if stream:
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=ByteStream()
            )
        return httpx.Response(200, json=body)

    _mock_provider(monkeypatch, httpx.MockTransport(handle))
    with _client(state) as client:
        result = client.post(
            "/v1/runtime/responses",
            headers={"Authorization": "Bearer task-token"},
            json=_stream_request() | {"stream": stream},
        )
    assert result.status_code == 200
    if stream:
        assert result.content == wire
        span = backend.trace_span.return_value.__enter__.return_value
        span.add_event.assert_called_once_with("provider.first_text")
    else:
        assert result.json() == body
    state.add_task_usage.assert_called_once()
    task_id, usage = state.add_task_usage.call_args.args
    assert task_id == 123 and usage.provider == "model"
    assert (usage.input_tokens, usage.output_tokens) == (11, 7)


@pytest.mark.parametrize(
    "mode,status",
    [
        ("http_error", 503),
        ("invalid_response", 502),
        ("missing_key", 500),
        ("invalid_endpoint", 500),
        ("usage_error", 500),
    ],
)
def test_model_errors_return_json(
    state: Mock, monkeypatch: pytest.MonkeyPatch, mode: str, status: int
) -> None:
    """Return provider, configuration and accounting failures as JSON errors."""
    if mode == "missing_key":
        monkeypatch.setenv(
            "FLWR_MODEL_API_ENDPOINT", "https://api.flower.ai/v1/responses"
        )
        monkeypatch.delenv("FLWR_MODEL_API_KEY", raising=False)
    elif mode == "invalid_endpoint":
        monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://provider/v1")
    response = httpx.Response(503 if mode == "http_error" else 200, json={})
    if mode == "usage_error":
        state.add_task_usage.side_effect = SQLAlchemyError("Usage store unavailable")
        response = httpx.Response(
            200, json={"object": "response", "usage": {"input_tokens": 1}}
        )
    _mock_provider(monkeypatch, httpx.MockTransport(lambda _request: response))
    with _client(state) as client:
        result = client.post(
            "/v1/runtime/responses",
            headers={"Authorization": "Bearer task-token"},
            json=_stream_request() | {"stream": False},
        )
        assert result.status_code == status
        assert "error" in result.json()
        if status == 500:
            assert result.json()["error"]["message"] == "Internal server error."
    assert state.add_task_usage.call_count == (1 if mode == "usage_error" else 0)


async def _check_model_cancellation(
    state: Mock, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    """Close quiet upstream streams and bound blocked error delivery."""
    entered, closed, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class QuietStream(httpx.AsyncByteStream):
        """Keep inference idle until cancellation closes the connection."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await anyio.sleep_forever()
            yield b""

        async def aclose(self) -> None:
            closed.set()

    def handle(_request: httpx.Request) -> httpx.Response:
        if stop == "error_deadline":
            return httpx.Response(503, json={})
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=QuietStream()
        )

    _mock_provider(monkeypatch, httpx.MockTransport(handle))
    monkeypatch.setattr(
        responses, "_model_response_timeout", lambda: 0.05 if "deadline" in stop else 3
    )
    monkeypatch.setattr(responses, "_POLL_INTERVAL", 0.01)
    scope: Scope = {"type": "http", "app": FastAPI(), "asgi": {"spec_version": "2.3"}}

    async def receive() -> Message:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if stop == "broken_pipe":
            entered.set()
            raise BrokenPipeError
        if stop == "error_deadline" and message["type"] == "http.response.body":
            entered.set()
            await anyio.sleep_forever()

    async def trigger() -> None:
        await entered.wait()
        if stop == "task":
            state.get_tasks.return_value = []
        elif stop == "disconnect":
            disconnected.set()

    response = ModelResponse(
        state, state.get_task_by_token.return_value, _stream_request()
    )
    async with asyncio.timeout(4):
        if "deadline" in stop:
            with pytest.raises(ExceptionGroup) as error:
                await asyncio.gather(response(scope, receive, send), trigger())
            assert any(isinstance(exc, TimeoutError) for exc in error.value.exceptions)
        else:
            await asyncio.gather(response(scope, receive, send), trigger())
    if stop != "error_deadline":
        assert closed.is_set()


@pytest.mark.parametrize(
    "stop", ["task", "disconnect", "broken_pipe", "deadline", "error_deadline"]
)
def test_model_cancellation_closes_provider(
    state: Mock, monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    """Exercise cancellation without requiring an async pytest plugin."""
    asyncio.run(_check_model_cancellation(state, monkeypatch, stop))
