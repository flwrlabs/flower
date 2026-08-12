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
"""Tests for the Runtime task-token authentication dependency."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.error import ApiErrorCode, FlowerError

from .task import TaskDependency, get_task


def _make_request(tokens: list[str]) -> Request:
    """Return a minimal request with the specified task-token headers."""
    return Request(
        {
            "type": "http",
            "headers": [
                (TASK_TOKEN_HEADER.encode(), token.encode()) for token in tokens
            ],
        }
    )


def test_get_task_returns_task_for_valid_token() -> None:
    """Return the task associated with a valid task token."""
    state = Mock(spec=LinkState)
    expected_task = Task(task_id=123)
    state.get_task_by_token.return_value = expected_task

    task = get_task(_make_request(["valid-token"]), "valid-token", state)

    assert task is expected_task
    state.get_task_by_token.assert_called_once_with("valid-token")


@pytest.mark.parametrize("token", [None, "invalid-token"])
def test_get_task_rejects_missing_or_invalid_token(token: str | None) -> None:
    """Reject requests without a task matching the supplied token."""
    state = Mock(spec=LinkState)
    state.get_task_by_token.return_value = None

    with pytest.raises(FlowerError) as exc_info:
        get_task(_make_request([token] if token else []), token, state)

    assert exc_info.value.code == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
    assert exc_info.value.message == "Runtime task-token authentication failed."
    if token is None:
        state.get_task_by_token.assert_not_called()
    else:
        state.get_task_by_token.assert_called_once_with(token)


def test_get_task_rejects_duplicate_token_headers() -> None:
    """Reject duplicate task-token headers without querying state."""
    state = Mock(spec=LinkState)

    with pytest.raises(FlowerError) as exc_info:
        get_task(
            _make_request(["valid-token", "another-token"]),
            "valid-token",
            state,
        )

    assert exc_info.value.code == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
    assert exc_info.value.message == "Runtime task-token authentication failed."
    state.get_task_by_token.assert_not_called()


def test_task_dependency_declares_task_token_security_scheme() -> None:
    """Describe task-token authentication in the generated OpenAPI schema."""
    app = FastAPI()

    @app.get("/runtime/test")
    def runtime_endpoint(task: TaskDependency) -> None:
        _ = task

    schema = app.openapi()

    assert schema["components"]["securitySchemes"]["RuntimeTaskToken"] == {
        "type": "apiKey",
        "description": "Task token issued by the Runtime API.",
        "in": "header",
        "name": TASK_TOKEN_HEADER,
    }
    assert schema["paths"]["/runtime/test"]["get"]["security"] == [
        {"RuntimeTaskToken": []}
    ]
