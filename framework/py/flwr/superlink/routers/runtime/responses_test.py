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

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from flwr.common.constant import Status, SubStatus
from flwr.proto.task_pb2 import Task, TaskEvent, TaskStatus  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.model_message import ModelResponse
from flwr.superlink.dependencies.linkstate import get_linkstate

from .responses import (
    _Exchange,
    _ResponsesError,
    _stream_response,
    _wait_for_response,
    router,
)


def _client(state: Mock) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_linkstate] = lambda: state
    return TestClient(app)


def _state() -> Mock:
    state = Mock(spec=LinkState)
    state.get_task_by_token.return_value = Task(
        task_id=123, run_id=789, type=TaskType.AGENT_APP
    )
    state.create_task.return_value = 456
    state.store_task_message.return_value = True
    return state


def _reply(request_message_id: str) -> ModelResponse:
    return ModelResponse(
        dst_task_id=123,
        response={
            "object": "response",
            "id": "resp_1",
            "status": "completed",
            "output": [],
        },
        reply_to_message_id=request_message_id,
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


def test_responses_returns_correlated_model_response() -> None:
    """Recheck and claim the direct reply after its child task finishes."""
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
        _event(1, "response.created"),
        _event(2, "response.completed"),
    ]
    state.get_task_message.side_effect = lambda **_: [
        _reply(state.store_task_message.call_args.args[0].metadata.message_id)
    ]

    response = _client(state).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello", "stream": True},
        headers={"Authorization": "Bearer task-token"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text == (
        'event: response.created\ndata: {"type":"response.created"}\n\n'
        'event: response.completed\ndata: {"type":"response.completed"}\n\n'
    )
    state.get_task_events.assert_called_once_with(
        run_id=789, task_ids=[456], after_task_event_id=None
    )


def test_responses_waits_for_terminal_event_after_reply() -> None:
    """Do not truncate a terminal event that becomes visible after its reply."""
    state = _state()
    state.get_task_events.side_effect = [
        [
            _event(
                1,
                "response.output_text.delta",
                '{"type":"response.output_text.delta","delta":"x"}',
            )
        ],
        [_event(2, "response.completed")],
    ]
    state.get_task_message.side_effect = lambda **_: [
        _reply(state.store_task_message.call_args.args[0].metadata.message_id)
    ]

    with patch("flwr.superlink.routers.runtime.responses._POLL_INTERVAL", new=0):
        response = _client(state).post(
            "/v1/runtime/responses",
            json={"model": "model", "input": "hello", "stream": True},
            headers={"Authorization": "Bearer task-token"},
        )

    assert "event: response.output_text.delta" in response.text
    assert "event: response.completed" in response.text
    assert "event: error" not in response.text
    assert state.get_task_events.call_count == 2


def test_responses_reports_missing_terminal_event() -> None:
    """Fail when the model reply has no matching terminal stream event."""
    state = _state()
    state.get_task_events.return_value = []
    state.get_task_message.side_effect = lambda **_: [
        _reply(state.store_task_message.call_args.args[0].metadata.message_id)
    ]

    with patch("flwr.superlink.routers.runtime.responses._TERMINAL_EVENT_GRACE", new=0):
        response = _client(state).post(
            "/v1/runtime/responses",
            json={"model": "model", "input": "hello", "stream": True},
            headers={"Authorization": "Bearer task-token"},
        )

    assert "event: error" in response.text
    assert "Model stream ended before a terminal event." in response.text


def test_responses_maps_unexpected_errors() -> None:
    """Return an OpenAI-style error envelope for unexpected failures."""
    with patch(
        "flwr.superlink.routers.runtime.responses._authenticate",
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
            run_id=789,
        )
        with patch("flwr.superlink.routers.runtime.responses._POLL_INTERVAL", new=10):
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
            run_id=789,
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
            run_id=789,
        )
        with patch(
            "flwr.superlink.routers.runtime.responses._model_task_launch_timeout",
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
            run_id=789,
        )
        with patch(
            "flwr.superlink.routers.runtime.responses._DEFAULT_MODEL_RESPONSE_TIMEOUT",
            new=0.0,
        ):
            with pytest.raises(_ResponsesError) as exc_info:
                await _wait_for_response(request, state, exchange)
        assert exc_info.value.status_code == 504
        assert exc_info.value.code == "model_response_timeout"

    asyncio.run(wait_for_response())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses request ended early."
    )


def test_responses_stops_and_drains_when_stream_is_cancelled() -> None:
    """Stop the child task and drain its reply when the stream is cancelled."""
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
            _Exchange(agent_task_id=123, model_task_id=456, run_id=789),
        )
        with patch("flwr.superlink.routers.runtime.responses._POLL_INTERVAL", new=10):
            next_event = asyncio.ensure_future(anext(stream))
            await asyncio.wait_for(wait_until_polled(), timeout=1)
            next_event.cancel()
            with pytest.raises(asyncio.CancelledError):
                await next_event

    asyncio.run(cancel_stream())

    state.finish_task.assert_called_once_with(
        456, SubStatus.STOPPED, "Responses stream ended early."
    )
    assert state.get_task_message.call_count == 2
