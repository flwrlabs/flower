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
"""Tests for the HTTP Runtime API stub."""

from unittest.mock import patch

import pytest
from google.protobuf.message import Message

from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
)
from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.runtime import RuntimeHttpStub


@pytest.mark.parametrize(
    ("method_name", "path", "request_message", "response_type"),
    [
        (
            "PullPendingTasks",
            "/v1/runtime/pull-pending-tasks",
            PullPendingTasksRequest(),
            PullPendingTasksResponse,
        ),
        (
            "ClaimTask",
            "/v1/runtime/claim-task",
            ClaimTaskRequest(task_id=123),
            ClaimTaskResponse,
        ),
        (
            "GetRun",
            "/v1/runtime/get-run",
            GetRunRequest(run_id=123),
            GetRunResponse,
        ),
    ],
)
def test_runtime_method(
    method_name: str,
    path: str,
    request_message: Message,
    response_type: type[Message],
) -> None:
    """Call one shared Runtime HTTP endpoint."""
    response = response_type()
    stub = RuntimeHttpStub("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(stub, method_name)(request_message)

    assert result is response
    call.assert_called_once_with(
        path=path,
        rpc_method=f"/flwr.proto.Runtime/{method_name}",
        request=request_message,
        response_type=response_type,
    )
