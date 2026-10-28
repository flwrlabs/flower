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
from threading import Event
from typing import cast
from unittest.mock import AsyncMock, Mock

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
        ("missing_terminal", 200),
        ("invalid_terminal", 200),
        ("truncated", 200),
        ("full", 429),
    ],
)
def test_model_errors_never_fall_back(
    state: Mock, monkeypatch: pytest.MonkeyPatch, mode: str, status: int
) -> None:
    """Do not duplicate provider execution after a state-path failure."""
    if mode in {"http_error", "invalid_response"}:
        response = httpx.Response(503 if mode == "http_error" else 200, json={})
    elif mode in {"missing_terminal", "invalid_terminal"}:
        event: JSONObject = {"type": "response.completed"}
        if mode == "invalid_terminal":
            event = {"type": "response.incomplete", "response": {"object": "invalid"}}
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(event)}\n\n",
        )
    else:
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, text="data: {}\n\n"
        )
    _mock_provider(monkeypatch, httpx.MockTransport(lambda _request: response))
    with _client(state) as client:
        if mode == "full":
            cast(FastAPI, client.app).state.model_response_capacity = asyncio.Semaphore(
                0
            )
        result = client.post(
            "/v1/runtime/responses",
            headers={"Authorization": "Bearer task-token"},
            json=_stream_request()
            | {"stream": mode not in {"http_error", "invalid_response"}},
        )
        assert result.status_code == status and "error" in result.text
        assert "event: response.completed\n" not in result.text
        assert "event: response.incomplete\n" not in result.text
    state.add_task_usage.assert_not_called()
    state.create_task.assert_not_called()


async def _check_model_cancellation(
    state: Mock, monkeypatch: pytest.MonkeyPatch, stop: str, phase: str
) -> None:
    """Cancel quiet requests and restore their single available slot."""
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
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=QuietStream()
        )

    _mock_provider(monkeypatch, httpx.MockTransport(handle))
    monkeypatch.setattr(
        responses, "_model_response_timeout", lambda: 0.05 if stop == "deadline" else 3
    )
    monkeypatch.setattr(responses, "_ERROR_DELIVERY_TIMEOUT", 0.05)
    scope: Scope = {"type": "http", "app": FastAPI(), "asgi": {"spec_version": "2.4"}}
    scope["app"].state.model_response_capacity = asyncio.Semaphore(1)

    async def receive() -> Message:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    response = ModelResponse(
        Request(scope, receive),
        state,
        state.get_task_by_token.return_value,
        _stream_request() | {"stream": phase != "opening"},
    )

    async def trigger() -> None:
        await entered.wait()
        assert response.capacity.locked()
        if stop == "task":
            state.get_tasks.return_value = []
        elif stop == "disconnect":
            disconnected.set()

    messages: list[Message] = []

    async def send(message: Message) -> None:
        await anyio.sleep(0)
        if phase == "blocked" and message["type"] == "http.response.body":
            await anyio.sleep_forever()
        messages.append(message)

    async with asyncio.timeout(4):
        await asyncio.gather(response(scope, receive, send), trigger())
    assert closed.is_set() and not response.capacity.locked()
    if stop == "deadline" and phase != "blocked":
        assert any(b"model_response_timeout" in m.get("body", b"") for m in messages)
        assert messages[0]["status"] == (504 if phase == "opening" else 200)


@pytest.mark.parametrize(
    "stop,phase",
    [
        ("task", "opening"),
        ("disconnect", "streaming"),
        ("deadline", "opening"),
        ("deadline", "streaming"),
        ("deadline", "blocked"),
    ],
)
def test_model_cancellation_closes_provider_and_releases_slot(
    state: Mock, monkeypatch: pytest.MonkeyPatch, stop: str, phase: str
) -> None:
    """Run cancellation coverage without requiring an async pytest plugin."""
    asyncio.run(_check_model_cancellation(state, monkeypatch, stop, phase))


def test_model_timeout_releases_slot_while_usage_write_finishes(
    state: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not wait for a stalled usage write after the response deadline."""
    release, finished, status_waiting = Event(), Event(), Event()

    def store_usage(*_args: object) -> None:
        release.wait(3)
        finished.set()

    def get_tasks(**_kwargs: object) -> list[Task]:
        if state.add_task_usage.called:
            status_waiting.set()
            release.wait(3)
        return [state.get_task_by_token.return_value]

    state.add_task_usage.side_effect = store_usage
    state.get_tasks.side_effect = get_tasks
    monkeypatch.setattr(responses, "_model_response_timeout", lambda: 0.25)
    monkeypatch.setattr(responses, "_POLL_INTERVAL", 0.01)
    _mock_provider(
        monkeypatch,
        httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"object": "response", "usage": {"input_tokens": 1}},
            )
        ),
    )

    async def check() -> None:
        app = FastAPI()
        app.state.model_response_capacity = asyncio.Semaphore(1)
        scope: Scope = {"type": "http", "app": app}
        response = ModelResponse(
            Request(scope, AsyncMock(return_value={"type": "http.request"})),
            state,
            state.get_task_by_token.return_value,
            _stream_request() | {"stream": False},
        )
        send = AsyncMock()
        try:
            async with asyncio.timeout(2):
                await response(scope, AsyncMock(), send)
            assert not response.capacity.locked() and not finished.is_set()
            assert status_waiting.is_set()
            assert send.call_args_list[0].args[0]["status"] == 504
            state.add_task_usage.assert_called_once()
        finally:
            release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 1)

    asyncio.run(check())
