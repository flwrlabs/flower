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
from unittest.mock import AsyncMock, Mock, patch

import anyio
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse
from starlette.types import Message, Scope

from flwr.app.constants import DEFAULT_TTL
from flwr.app.message_type import MessageType
from flwr.app.metadata import Metadata
from flwr.common.constant import Status, SubStatus
from flwr.proto.task_pb2 import Task, TaskEvent, TaskStatus  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate import CoreState
from flwr.supercore.date import now
from flwr.supercore.dependencies.runtime import get_runtime_state
from flwr.supercore.json_message.base import make_json_message
from flwr.supercore.json_message.model_message import ModelResponse
from flwr.supercore.typing import JSONObject

from . import responses
from .responses import (
    _DirectResponse,
    _Exchange,
    _model_response_timeout,
    _ResponsesError,
    _sse_frame,
    _stream_response,
    _wait_for_response,
    router,
)


def test_model_response_timeout_uses_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the Runtime response timeout independently of provider settings."""
    monkeypatch.delenv("FLWR_MODEL_RESPONSE_TIMEOUT", raising=False)

    assert _model_response_timeout() == 900.0

    monkeypatch.setenv("FLWR_MODEL_RESPONSE_TIMEOUT", "1800")

    assert _model_response_timeout() == 1800.0


def _client(state: Mock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runtime_state] = lambda: state
    return TestClient(app)


def _state() -> Mock:
    state = Mock(spec=CoreState)
    state.get_node_id.return_value = 789
    state.get_task_by_token.return_value = Task(
        task_id=123, run_id=789, type=TaskType.AGENT_APP
    )
    state.create_task.return_value = 456
    state.store_task_message.return_value = True
    return state


def _reply(request_message_id: str) -> ModelResponse:
    metadata = Metadata(
        run_id=789,
        message_id="response-message-id",
        src_node_id=789,
        dst_node_id=789,
        reply_to_message_id=request_message_id,
        group_id="",
        created_at=now().timestamp(),
        ttl=DEFAULT_TTL,
        message_type=MessageType.QUERY,
        src_task_id=456,
        dst_task_id=123,
    )
    return make_json_message(
        ModelResponse,
        metadata=metadata,
        payload={
            "object": "response",
            "id": "resp_1",
            "status": "completed",
            "output": [],
        },
    )


def _event(event_id: int, event: str, data: str | None = None) -> TaskEvent:
    """Create one child model task event."""
    return TaskEvent(
        id=event_id,
        run_id=789,
        task_id=456,
        event=event,
        data=data or f'{{"type":"{event}"}}',
    )


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


@pytest.mark.parametrize("direct_models", ["", "other-model"])
def test_responses_returns_correlated_model_response(
    monkeypatch: pytest.MonkeyPatch, direct_models: str
) -> None:
    """Recheck and claim the reply for default and unlisted-model execution."""
    monkeypatch.setenv("FLWR_RUNTIME_MODEL_DIRECT_MODELS", direct_models)
    state = _state()
    state.get_task_message.side_effect = [
        [],
        [_reply("request-message-id")],
    ]
    state.get_tasks.return_value = [
        Task(task_id=456, status=TaskStatus(status=Status.FINISHED))
    ]

    response = _client(state).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello"},
        headers={"Authorization": "Bearer task-token"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "resp_1"
    request = state.store_task_message.call_args.args[0]
    assert request.metadata.src_task_id == 123
    assert request.metadata.dst_task_id == 456
    assert request.metadata.src_node_id == 789
    assert request.metadata.dst_node_id == 789
    assert state.get_task_message.call_count == 2
    state.get_task_message.assert_called_with(
        dst_task_ids=[123],
        src_task_ids=[456],
        limit=1,
        order_by="created_at",
    )


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


def test_responses_streams_only_child_task_events() -> None:
    """Relay ordered child-task events and consume the final reply."""
    state = _state()
    state.get_task_events.return_value = [
        _event(1, "response.created", '{\n  "type": "response.created"\n}'),
        _event(2, "response.completed"),
    ]
    state.get_task_message.side_effect = lambda **_: [
        _reply(state.store_task_message.call_args.args[0].metadata.message_id)
    ]

    with patch("flwr.supercore.routers.runtime.responses.trace_span") as trace:
        response = _client(state).post(
            "/v1/runtime/responses",
            json={"model": "model", "input": "hello", "stream": True},
            headers={"Authorization": "Bearer task-token"},
        )
    assert [call.args[0] for call in trace.call_args_list] == [
        "model.round_trip",
        "model.create",
        "model.request_publish",
        "model.response_wait",
        "model.first_text_wait",
    ]

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        "event: response.created\n"
        "data: {\n"
        'data:   "type": "response.created"\n'
        "data: }\n\n"
        'event: response.completed\ndata: {"type":"response.completed"}\n\n'
    )
    state.get_task_events.assert_called_once_with(
        task_ids=[456], after_task_event_id=None
    )


@pytest.mark.parametrize(
    "event_name",
    ["response.created\ninjected", "response.created\rinjected"],
)
def test_sse_frame_rejects_event_name_line_breaks(event_name: str) -> None:
    """Reject event names that could inject an SSE field or frame."""
    with pytest.raises(_ResponsesError) as exc_info:
        _sse_frame(TaskEvent(event=event_name, data="{}"))

    assert exc_info.value.status_code == 502
    assert exc_info.value.code == "invalid_model_event"


def test_responses_waits_for_reply_before_terminal_event() -> None:
    """Consume the final reply before exposing its terminal stream event."""
    state = _state()
    state.get_task_events.return_value = [
        _event(
            1,
            "response.output_text.delta",
            '{"type":"response.output_text.delta","delta":"x"}',
        ),
        _event(2, "response.completed"),
    ]
    state.get_task_message.side_effect = [[], [_reply("request-message-id")]]
    state.get_tasks.return_value = [
        Task(task_id=456, status=TaskStatus(status=Status.RUNNING))
    ]

    response = _client(state).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello", "stream": True},
        headers={"Authorization": "Bearer task-token"},
    )

    assert "event: response.output_text.delta" in response.text
    assert "event: response.completed" in response.text
    assert "event: error" not in response.text
    assert state.get_task_message.call_count == 2


def test_responses_reports_missing_terminal_event() -> None:
    """Fail when the model reply has no matching terminal stream event."""
    state = _state()
    state.get_task_events.return_value = [
        _event(
            1,
            "response.output_text.delta",
            '{"type":"response.output_text.delta","delta":"x"}',
        )
    ]
    state.get_task_message.side_effect = lambda **_: [
        _reply(state.store_task_message.call_args.args[0].metadata.message_id)
    ]

    response = _client(state).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello", "stream": True},
        headers={"Authorization": "Bearer task-token"},
    )

    assert response.text.endswith(
        "event: error\n"
        'data: {"type":"error","code":"model_stream_failed",'
        '"message":"Model stream ended before a terminal event.",'
        '"param":null,"sequence_number":1}\n\n'
    )


def test_responses_maps_unexpected_errors() -> None:
    """Return an OpenAI-style error envelope for unexpected failures."""
    with patch(
        "flwr.supercore.routers.runtime.responses._authenticate",
        side_effect=RuntimeError("unexpected"),
    ):
        response = _client(_state()).post(
            "/v1/runtime/responses",
            json={"model": "model", "input": "hello"},
            headers={"Authorization": "Bearer task-token"},
        )

    assert response.status_code == 500
    assert response.json()["error"] == {
        "message": "Internal server error.",
        "type": "server_error",
        "param": None,
        "code": "internal_error",
    }


def test_responses_stops_and_drains_when_response_wait_is_cancelled() -> None:
    """Stop the child task and drain its reply when response waiting is cancelled."""
    state = _state()
    state.get_task_message.return_value = []
    state.get_tasks.return_value = [Task(task_id=456)]

    async def wait_until_polled() -> None:
        while not state.get_tasks.called:
            await asyncio.sleep(0)

    async def cancel_response_wait() -> None:
        request = Mock(spec=Request)
        request.is_disconnected = AsyncMock(return_value=False)
        exchange = _Exchange(
            agent_task_id=123,
            model_task_id=456,
        )
        with patch("flwr.supercore.routers.runtime.responses._POLL_INTERVAL", new=10):
            response_wait = asyncio.create_task(
                _wait_for_response(request, state, exchange)
            )
            await asyncio.wait_for(wait_until_polled(), timeout=1)
            response_wait.cancel()
            with pytest.raises(asyncio.CancelledError):
                await response_wait

    asyncio.run(cancel_response_wait())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses request ended early."
    )
    assert state.get_task_message.call_count == 2


def test_responses_stops_and_drains_when_client_disconnects() -> None:
    """Stop the child task and drain its reply when the client disconnects."""
    state = _state()
    state.get_task_message.return_value = []
    request = Mock(spec=Request)
    request.is_disconnected = AsyncMock(return_value=True)

    async def wait_for_response() -> None:
        exchange = _Exchange(
            agent_task_id=123,
            model_task_id=456,
        )
        with pytest.raises(_ResponsesError) as exc_info:
            await _wait_for_response(request, state, exchange)
        assert exc_info.value.code == "client_disconnected"

    asyncio.run(wait_for_response())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses request ended early."
    )
    state.get_task_message.assert_called_once_with(
        dst_task_ids=[123],
        src_task_ids=[456],
        limit=1,
        order_by="created_at",
    )


def test_responses_times_out_if_model_task_is_not_launched() -> None:
    """Stop waiting when no executor launches the child model task."""
    state = _state()
    state.get_task_message.return_value = []
    state.get_tasks.return_value = [
        Task(
            task_id=456,
            status=TaskStatus(status=Status.PENDING),
        )
    ]
    request = Mock(spec=Request)
    request.is_disconnected = AsyncMock(return_value=False)

    async def wait_for_response() -> None:
        exchange = _Exchange(
            agent_task_id=123,
            model_task_id=456,
        )
        with patch(
            "flwr.supercore.routers.runtime.responses._model_task_launch_timeout",
            return_value=0.0,
        ):
            with pytest.raises(_ResponsesError) as exc_info:
                await _wait_for_response(request, state, exchange)
        assert exc_info.value.status_code == 504
        assert exc_info.value.code == "model_task_launch_timeout"

    asyncio.run(wait_for_response())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses request ended early."
    )


def test_responses_times_out_if_running_model_task_does_not_respond() -> None:
    """Stop waiting when a running model task does not produce a response."""
    state = _state()
    state.get_task_message.return_value = []
    state.get_tasks.return_value = [
        Task(
            task_id=456,
            status=TaskStatus(status=Status.RUNNING),
        )
    ]
    request = Mock(spec=Request)
    request.is_disconnected = AsyncMock(return_value=False)

    async def wait_for_response() -> None:
        exchange = _Exchange(
            agent_task_id=123,
            model_task_id=456,
        )
        with patch(
            "flwr.supercore.routers.runtime.responses._model_response_timeout",
            return_value=0.0,
        ):
            with pytest.raises(_ResponsesError) as exc_info:
                await _wait_for_response(request, state, exchange)
        assert exc_info.value.status_code == 504
        assert exc_info.value.code == "model_response_timeout"

    asyncio.run(wait_for_response())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses request ended early."
    )


def test_responses_stops_and_drains_when_stream_task_group_is_cancelled() -> None:
    """Shield child-task cleanup from stream task-group cancellation."""
    state = _state()
    state.get_task_events.return_value = []
    state.get_task_message.return_value = []
    state.get_tasks.return_value = [
        Task(task_id=456, status=TaskStatus(status=Status.RUNNING))
    ]

    async def wait_until_polled() -> None:
        while not state.get_tasks.called:
            await asyncio.sleep(0)

    async def cancel_stream() -> None:
        stream = _stream_response(
            state,
            state.get_task_by_token.return_value,
            _stream_request(),
        )

        async def consume_stream() -> None:
            await anext(stream)

        with patch("flwr.supercore.routers.runtime.responses._POLL_INTERVAL", new=10):
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(consume_stream)
                await asyncio.wait_for(wait_until_polled(), timeout=1)
                task_group.cancel_scope.cancel()

    asyncio.run(cancel_stream())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses stream ended early."
    )
    assert state.get_task_message.call_count == 2


def test_responses_does_not_start_model_task_before_stream_iteration() -> None:
    """Do not create a model task if the client disconnects before iteration."""
    state = _state()
    response_started = anyio.Event()

    async def disconnect_before_streaming() -> None:
        response = StreamingResponse(
            _stream_response(
                state,
                state.get_task_by_token.return_value,
                _stream_request(),
            )
        )

        async def receive() -> Message:
            await response_started.wait()
            return {"type": "http.disconnect"}

        async def send(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_started.set()
                await anyio.sleep_forever()

        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
        }
        await response(scope, receive, send)

    asyncio.run(disconnect_before_streaming())

    state.create_task.assert_not_called()


@pytest.fixture
def direct(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Enable one model without changing task authentication."""
    monkeypatch.setenv("FLWR_RUNTIME_MODEL_DIRECT_MODELS", "model")
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
        "flwr.supercore.task_process.model.provider.httpx.AsyncClient",
        lambda **kwargs: client(transport=transport, **kwargs),
    )


@pytest.mark.parametrize("stream", [False, True])
def test_direct_response_accounts_each_round(
    direct: Mock, monkeypatch: pytest.MonkeyPatch, stream: bool
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
    with _client(direct) as client:
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
    assert direct.add_task_usage.call_count == 2
    for call in direct.add_task_usage.call_args_list:
        task_id, usage = call.args
        assert task_id == 123 and usage.provider == "model"
        assert (usage.input_tokens, usage.output_tokens) == (11, 7)
    direct.create_task.assert_not_called()


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
def test_direct_errors_never_fall_back(
    direct: Mock, monkeypatch: pytest.MonkeyPatch, mode: str, status: int
) -> None:
    """Do not duplicate provider execution after a direct-path failure."""
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
    with _client(direct) as client:
        if mode == "full":
            cast(FastAPI, client.app).state.direct_model_capacity = asyncio.Semaphore(0)
        result = client.post(
            "/v1/runtime/responses",
            headers={"Authorization": "Bearer task-token"},
            json=_stream_request()
            | {"stream": mode not in {"http_error", "invalid_response"}},
        )
        assert result.status_code == status and "error" in result.text
        assert "event: response.completed\n" not in result.text
        assert "event: response.incomplete\n" not in result.text
    direct.add_task_usage.assert_not_called()
    direct.create_task.assert_not_called()


async def _check_direct_cancellation(
    direct: Mock, monkeypatch: pytest.MonkeyPatch, stop: str, phase: str
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
    monkeypatch.setattr(responses, "_DIRECT_ERROR_DELIVERY_TIMEOUT", 0.05)
    scope: Scope = {"type": "http", "app": FastAPI(), "asgi": {"spec_version": "2.4"}}
    scope["app"].state.direct_model_capacity = asyncio.Semaphore(1)

    async def receive() -> Message:
        await disconnected.wait()
        return {"type": "http.disconnect"}

    response = _DirectResponse(
        Request(scope, receive),
        direct,
        direct.get_task_by_token.return_value,
        _stream_request() | {"stream": phase != "opening"},
    )

    async def trigger() -> None:
        await entered.wait()
        assert response.capacity.locked()
        if stop == "task":
            direct.get_tasks.return_value = []
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
def test_direct_cancellation_closes_provider_and_releases_slot(
    direct: Mock, monkeypatch: pytest.MonkeyPatch, stop: str, phase: str
) -> None:
    """Run cancellation coverage without requiring an async pytest plugin."""
    asyncio.run(_check_direct_cancellation(direct, monkeypatch, stop, phase))


def test_direct_timeout_releases_slot_while_usage_write_finishes(
    direct: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not wait for a stalled usage write after the response deadline."""
    release, finished, status_waiting = Event(), Event(), Event()

    def store_usage(*_args: object) -> None:
        release.wait(3)
        finished.set()

    def get_tasks(**_kwargs: object) -> list[Task]:
        if direct.add_task_usage.called:
            status_waiting.set()
            release.wait(3)
        return [direct.get_task_by_token.return_value]

    direct.add_task_usage.side_effect = store_usage
    direct.get_tasks.side_effect = get_tasks
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
        app.state.direct_model_capacity = asyncio.Semaphore(1)
        scope: Scope = {"type": "http", "app": app}
        response = _DirectResponse(
            Request(scope, AsyncMock(return_value={"type": "http.request"})),
            direct,
            direct.get_task_by_token.return_value,
            _stream_request() | {"stream": False},
        )
        send = AsyncMock()
        try:
            async with asyncio.timeout(2):
                await response(scope, AsyncMock(), send)
            assert not response.capacity.locked() and not finished.is_set()
            assert status_waiting.is_set()
            assert send.call_args_list[0].args[0]["status"] == 504
            direct.add_task_usage.assert_called_once()
        finally:
            release.set()
        assert await anyio.to_thread.run_sync(finished.wait, 1)

    asyncio.run(check())
