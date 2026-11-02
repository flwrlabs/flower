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
from fastapi import Request

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.corestate import CoreState
from flwr.supercore.error import ApiErrorCode, FlowerError

from .task import authenticate_task


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


def test_authenticate_task_returns_task_for_valid_token() -> None:
    """Return the task associated with a valid task token."""
    state = Mock(spec=CoreState)
    expected_task = Task(task_id=123)
    state.get_task_by_token.return_value = expected_task

    task = authenticate_task(_make_request(["valid-token"]), "valid-token", state)

    assert task is expected_task
    state.get_task_by_token.assert_called_once_with("valid-token")


@pytest.mark.parametrize("token", [None, "invalid-token"])
def test_authenticate_task_rejects_missing_or_invalid_token(
    token: str | None,
) -> None:
    """Reject requests without a task matching the supplied token."""
    state = Mock(spec=CoreState)
    state.get_task_by_token.return_value = None

    with pytest.raises(FlowerError) as exc_info:
        authenticate_task(_make_request([token] if token else []), token, state)

    assert exc_info.value.code == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
    assert exc_info.value.message == "Runtime task-token authentication failed."
    if token is None:
        state.get_task_by_token.assert_not_called()
    else:
        state.get_task_by_token.assert_called_once_with(token)


def test_authenticate_task_rejects_duplicate_token_headers() -> None:
    """Reject duplicate task-token headers without querying state."""
    state = Mock(spec=CoreState)

    with pytest.raises(FlowerError) as exc_info:
        authenticate_task(
            _make_request(["valid-token", "another-token"]),
            "valid-token",
            state,
        )

    assert exc_info.value.code == ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED
    assert exc_info.value.message == "Runtime task-token authentication failed."
    state.get_task_by_token.assert_not_called()
