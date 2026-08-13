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

from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
)
from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.runtime import RuntimeHttpStub


def test_pull_pending_tasks() -> None:
    """Call the PullPendingTasks HTTP endpoint."""
    request = PullPendingTasksRequest()
    response = PullPendingTasksResponse()
    stub = RuntimeHttpStub("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = stub.PullPendingTasks(request=request)

    assert result is response
    call.assert_called_once_with(
        path="/v1/runtime/pull-pending-tasks",
        rpc_method="/flwr.proto.Runtime/PullPendingTasks",
        request=request,
        response_type=PullPendingTasksResponse,
    )


def test_claim_task() -> None:
    """Call the ClaimTask HTTP endpoint."""
    request = ClaimTaskRequest(task_id=123)
    response = ClaimTaskResponse(token="token")
    stub = RuntimeHttpStub("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = stub.ClaimTask(request)

    assert result is response
    call.assert_called_once_with(
        path="/v1/runtime/claim-task",
        rpc_method="/flwr.proto.Runtime/ClaimTask",
        request=request,
        response_type=ClaimTaskResponse,
    )


def test_get_run() -> None:
    """Call the GetRun HTTP endpoint."""
    request = GetRunRequest(run_id=123)
    response = GetRunResponse()
    stub = RuntimeHttpStub("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = stub.GetRun(request)

    assert result is response
    call.assert_called_once_with(
        path="/v1/runtime/get-run",
        rpc_method="/flwr.proto.Runtime/GetRun",
        request=request,
        response_type=GetRunResponse,
    )
