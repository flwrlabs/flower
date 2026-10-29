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
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
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


def test_responses_rejects_unsupported_fields() -> None:
    """Do not silently discard unsupported Open Responses fields."""
    state = _state()

    response = _client(state).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello", "temperature": 0.5},
        headers={"Authorization": "Bearer task-token"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_parameter"
    state.create_task.assert_not_called()


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


@pytest.mark.parametrize("stream", [False, True])
def test_model_response_accounts_each_round(
    state: Mock, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """Preserve credentials and usage, including a terminal SSE event at EOF."""
    backend = MagicMock()
    monkeypatch.setenv("FLWR_TRACING_ENABLED", "1")
    monkeypatch.setattr("flwr.supercore.tracing._load_backend", lambda: backend)
    body: JSONObject = {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer provider-key"
        if request.url.path == "/v1/responses":
            return httpx.Response(307, headers={"location": "/v1/model-responses"})
        if not stream:
            return httpx.Response(200, json=body)
        # Event types come from SSE names. The final event has no blank separator.
        text = 'event: response.output_text.delta\ndata: {"delta":"Hi"}\n\n'
        text += f"event: response.completed\ndata: {json.dumps({'response': body})}"
        return httpx.Response(
            200, headers={"content-type": "Text/Event-Stream; charset=utf-8"}, text=text
        )

    _mock_provider(monkeypatch, httpx.MockTransport(handle))
    with _client(state) as client:
        for _ in range(2):
            result = client.post(
                "/v1/runtime/responses",
                headers={"Authorization": "Bearer task-token"},
                json=_stream_request() | {"stream": stream},
            )
            assert result.status_code == 200
            if stream:
                assert "event: response.completed\n" in result.text
            else:
                assert result.json() == body
    attributes = backend.trace_span.return_value.__enter__.return_value.set_attribute
    assert all(
        call.args[0] != "error.type" for call in attributes.call_args_list
    ), attributes.call_args_list
    if stream:
        assert (
            backend.trace_span.return_value.__enter__.return_value.add_event.call_count
            == 2
        )
    assert state.add_task_usage.call_count == 2
    for call in state.add_task_usage.call_args_list:
        task_id, usage = call.args
        assert task_id == 123 and usage.provider == "model"
        assert (usage.input_tokens, usage.output_tokens) == (11, 7)
    state.create_task.assert_not_called()


@pytest.mark.parametrize(
    "mode,status",
    [
        ("http_error", 503),
        ("invalid_response", 502),
        ("missing_key", 500),
        ("invalid_endpoint", 500),
    ],
)
def test_model_errors_never_fall_back(
    state: Mock, monkeypatch: pytest.MonkeyPatch, mode: str, status: int
) -> None:
    """Return JSON errors without duplicating inference or exposing configuration."""
    if mode == "missing_key":
        monkeypatch.setenv(
            "FLWR_MODEL_API_ENDPOINT", "https://api.flower.ai/v1/responses"
        )
        monkeypatch.delenv("FLWR_MODEL_API_KEY", raising=False)
    elif mode == "invalid_endpoint":
        monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://provider/v1")
    response = httpx.Response(503 if mode == "http_error" else 200, json={})
    _mock_provider(monkeypatch, httpx.MockTransport(lambda _request: response))
    with _client(state) as client:
        result = client.post(
            "/v1/runtime/responses",
            headers={"Authorization": "Bearer task-token"},
            json=_stream_request() | {"stream": False},
        )
        assert result.status_code == status and "error" in result.text
        if status == 500:
            assert result.json()["error"]["message"] == "Internal server error."
        assert "event: response.completed\n" not in result.text
        assert "event: response.incomplete\n" not in result.text
    state.add_task_usage.assert_not_called()
    state.create_task.assert_not_called()


async def _check_model_cancellation(  # pylint: disable=too-many-locals
    state: Mock,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
    phase: str,
    spec_version: str,
) -> None:
    """Close quiet upstream requests even before response headers arrive."""
    entered, closed, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class QuietStream(httpx.AsyncByteStream):
        """Keep the body idle until upstream cleanup."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await anyio.sleep_forever()
            yield b""

        async def aclose(self) -> None:
            closed.set()

    async def handle(_request: httpx.Request) -> httpx.Response:
        if phase == "opening":
            try:
                entered.set()
                await anyio.sleep_forever()
            finally:
                closed.set()
        if phase in {"json_blocked", "error_blocked"}:
            return httpx.Response(
                503 if phase == "error_blocked" else 200,
                json={"object": "response", "output": []},
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=QuietStream()
        )

    _mock_provider(monkeypatch, httpx.MockTransport(handle))
    monkeypatch.setattr(
        responses, "_model_response_timeout", lambda: 0.05 if stop == "deadline" else 3
    )
    scope: Scope = {
        "type": "http",
        "app": FastAPI(),
        "asgi": {"spec_version": spec_version},
    }

    async def receive() -> Message:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    response = ModelResponse(
        Request(scope, receive),
        state,
        state.get_task_by_token.return_value,
        _stream_request()
        | {"stream": phase not in {"opening", "json_blocked", "error_blocked"}},
    )

    async def trigger() -> None:
        await entered.wait()
        if stop == "task":
            state.get_tasks.return_value = []
        elif stop == "disconnect":
            disconnected.set()

    messages: list[Message] = []

    async def send(message: Message) -> None:
        await anyio.sleep(0)
        if (
            phase in {"blocked", "json_blocked", "error_blocked"}
            and message["type"] == "http.response.body"
        ):
            entered.set()
            await anyio.sleep_forever()
        messages.append(message)

    async with asyncio.timeout(4):
        if stop == "deadline":
            # Expiry aborts delivery without an additional error grace period.
            with pytest.raises(ExceptionGroup) as error:
                await asyncio.gather(response(scope, receive, send), trigger())
            assert any(isinstance(exc, TimeoutError) for exc in error.value.exceptions)
        else:
            await asyncio.gather(response(scope, receive, send), trigger())
    if phase not in {"json_blocked", "error_blocked"}:
        assert closed.is_set()
    assert not any(b"event: error" in message.get("body", b"") for message in messages)


@pytest.mark.parametrize(
    "stop,phase,spec_version",
    [
        ("task", "opening", "2.4"),
        ("task", "streaming", "2.4"),
        ("disconnect", "streaming", "2.4"),
        ("disconnect", "streaming", "2.0"),
        ("task", "streaming", "2.0"),
        ("deadline", "opening", "2.4"),
        ("deadline", "streaming", "2.4"),
        ("deadline", "blocked", "2.4"),
        ("deadline", "json_blocked", "2.4"),
        ("deadline", "error_blocked", "2.4"),
    ],
)
def test_model_cancellation_closes_provider(
    state: Mock,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
    phase: str,
    spec_version: str,
) -> None:
    """Run cancellation coverage without requiring an async pytest plugin."""
    asyncio.run(
        _check_model_cancellation(state, monkeypatch, stop, phase, spec_version)
    )


def test_stream_is_forwarded_without_validation(
    state: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve comments, unknown fields and malformed events across byte boundaries."""
    body = (
        b": heartbeat\r\n\r\nid: 42\r\nretry: 100\r\n"
        b"event: custom\r\ndata: not json\r\n\r\n"
        b'data: {"type":"response.output_text.delta","delta":"caf\xc3\xa9"}\r\n\r\n'
        b'event: response.completed\r\ndata: {"response":'
        b' {"usage": {"input_tokens": 1}}}\r\n\r\n'
        b"data: [DONE]\n\n"
    )

    class ByteStream(httpx.AsyncByteStream):
        """Split the wire payload at every possible byte boundary."""

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for value in body:
                yield bytes([value])

    _mock_provider(
        monkeypatch,
        httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ByteStream(),
            )
        ),
    )
    result = _client(state).post(
        "/v1/runtime/responses",
        headers={"Authorization": "Bearer task-token"},
        json=_stream_request(),
    )
    assert result.content == body
    state.add_task_usage.assert_called_once()
