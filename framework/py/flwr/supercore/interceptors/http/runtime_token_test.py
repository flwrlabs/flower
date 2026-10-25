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
"""Tests for the HTTP Runtime task-token interceptor."""

from unittest.mock import Mock

import httpx
import pytest

from flwr.proto.runtime_pb2 import PullTaskInputRequest  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .runtime_token import RuntimeTokenHttpInterceptor


def _context() -> ProtobufRequestContext:
    return ProtobufRequestContext(
        rpc_method="/flwr.proto.Runtime/PullTaskInput",
        message=PullTaskInputRequest(),
        request=httpx.Request("POST", "http://runtime.example"),
    )


def test_adds_task_token_header() -> None:
    """Attach the task token to the outgoing request."""
    context = _context()
    response = httpx.Response(200)
    call_next = Mock(return_value=response)

    result = RuntimeTokenHttpInterceptor("task-token").intercept(context, call_next)

    assert result is response
    assert context.request.headers[TASK_TOKEN_HEADER] == "task-token"
    call_next.assert_called_once_with(context)


def test_rejects_empty_token() -> None:
    """Reject an empty task token."""
    with pytest.raises(ValueError, match="non-empty"):
        RuntimeTokenHttpInterceptor("")


def test_rejects_existing_task_token_header() -> None:
    """Reject a task token supplied by another client layer."""
    context = _context()
    context.request.headers[TASK_TOKEN_HEADER] = "existing-token"

    with pytest.raises(RuntimeError, match=TASK_TOKEN_HEADER):
        RuntimeTokenHttpInterceptor("task-token").intercept(context, Mock())
