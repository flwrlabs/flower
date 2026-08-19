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

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flwr.proto.task_pb2 import Task, TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.model_message import ModelRequest, ModelResponse
from flwr.superlink.dependencies.linkstate import get_linkstate

from .responses import router


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


def _capture_requests(state: Mock) -> list[ModelRequest]:
    """Capture task messages stored by the endpoint."""
    requests: list[ModelRequest] = []

    def store(request: ModelRequest) -> bool:
        requests.append(request)
        return True

    state.store_task_message.side_effect = store
    return requests


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
    """Create a child model task and claim only its direct reply."""
    state = _state()
    stored_requests = _capture_requests(state)
    state.get_task_message.side_effect = lambda **_: [
        _reply(stored_requests[0].metadata.message_id)
    ]

    response = _client(state).post(
        "/v1/runtime/responses",
        json={"model": "model", "input": "hello"},
        headers={"Authorization": "Bearer task-token"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "resp_1"
    request = stored_requests[0]
    assert request.metadata.src_task_id == 123
    assert request.metadata.dst_task_id == 456
    state.get_task_message.assert_called_once_with(
        dst_task_ids=[123],
        reply_to_message_ids=[request.metadata.message_id],
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
    """Relay ordered events and consume the final correlated reply."""
    state = _state()
    stored_requests = _capture_requests(state)
    state.get_task_events.return_value = [
        TaskEvent(
            id=1,
            run_id=789,
            task_id=456,
            event="response.created",
            data='{"type":"response.created"}',
        ),
        TaskEvent(
            id=2,
            run_id=789,
            task_id=456,
            event="response.completed",
            data='{"type":"response.completed"}',
        ),
    ]
    state.get_task_message.side_effect = lambda **_: [
        _reply(stored_requests[0].metadata.message_id)
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
