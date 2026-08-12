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
"""Tests for the protobuf-over-HTTP Runtime API client."""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest
from google.protobuf.message import Message

from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.auth import (
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
)
from flwr.supercore.constant import (
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE

from . import RuntimeHttpStub

_TIMESTAMP = 1000
_NONCE = "nonce"


@pytest.mark.parametrize(
    ("method_name", "path", "request_message", "expected"),
    [
        (
            "PullPendingTasks",
            "/v1/runtime/pull-pending-tasks",
            PullPendingTasksRequest(),
            PullPendingTasksResponse(tasks=[Task(task_id=123)]),
        ),
        (
            "ClaimTask",
            "/v1/runtime/claim-task",
            ClaimTaskRequest(task_id=123),
            ClaimTaskResponse(token="task-token"),
        ),
        (
            "GetRun",
            "/v1/runtime/get-run",
            GetRunRequest(run_id=123),
            GetRunResponse(),
        ),
    ],
)
def test_runtime_method_sends_and_receives_protobuf(
    method_name: str,
    path: str,
    request_message: Message,
    expected: Message,
) -> None:
    """Send a protobuf request and parse the protobuf response."""
    response = Mock(content=expected.SerializeToString())

    with patch("flwr.supercore.runtime.http_stub.requests.Session") as session_class:
        session_class.return_value.post.return_value = response
        stub = RuntimeHttpStub(
            "https://runtime.example/", verify="ca.pem", timeout=10.0
        )
        result = getattr(stub, method_name)(request_message)

    assert result == expected
    session_class.return_value.post.assert_called_once_with(
        f"https://runtime.example{path}",
        data=request_message.SerializeToString(deterministic=True),
        headers={"content-type": PROTOBUF_MEDIA_TYPE},
        verify="ca.pem",
        timeout=10.0,
    )
    response.raise_for_status.assert_called_once_with()


@patch(
    "flwr.supercore.runtime.http_stub.now",
    return_value=datetime.fromtimestamp(_TIMESTAMP, UTC),
)
@patch("flwr.supercore.runtime.http_stub.secrets.token_hex", return_value=_NONCE)
@pytest.mark.parametrize(
    ("method_name", "rpc_method", "request_message", "response_message"),
    [
        (
            "PullPendingTasks",
            "/flwr.proto.Runtime/PullPendingTasks",
            PullPendingTasksRequest(),
            PullPendingTasksResponse(),
        ),
        (
            "ClaimTask",
            "/flwr.proto.Runtime/ClaimTask",
            ClaimTaskRequest(task_id=123),
            ClaimTaskResponse(),
        ),
        (
            "GetRun",
            "/flwr.proto.Runtime/GetRun",
            GetRunRequest(run_id=123),
            GetRunResponse(),
        ),
    ],
)
def test_runtime_method_adds_superexec_authentication(
    _token_hex: Mock,
    _now: Mock,
    method_name: str,
    rpc_method: str,
    request_message: Message,
    response_message: Message,
) -> None:
    """Sign SuperExec Runtime requests with their canonical method names."""
    http_response = Mock(content=response_message.SerializeToString())
    master_secret = b"master-secret"
    body_sha256 = compute_request_body_sha256(request_message)

    with patch("flwr.supercore.runtime.http_stub.requests.Session") as session_class:
        session_class.return_value.post.return_value = http_response
        stub = RuntimeHttpStub(
            "http://runtime.example", superexec_auth_secret=master_secret
        )
        getattr(stub, method_name)(request_message)

    headers = session_class.return_value.post.call_args.kwargs["headers"]
    assert headers == {
        "content-type": PROTOBUF_MEDIA_TYPE,
        SUPEREXEC_AUTH_TIMESTAMP_HEADER: str(_TIMESTAMP),
        SUPEREXEC_AUTH_NONCE_HEADER: _NONCE,
        SUPEREXEC_AUTH_BODY_SHA256_HEADER: body_sha256,
        SUPEREXEC_AUTH_SIGNATURE_HEADER: compute_superexec_signature(
            auth_secret=derive_auth_secret(master_secret),
            method=rpc_method,
            timestamp=_TIMESTAMP,
            nonce=_NONCE,
            body_sha256=body_sha256,
        ),
    }


def test_close_closes_session() -> None:
    """Close the underlying requests session."""
    with patch("flwr.supercore.runtime.http_stub.requests.Session") as session_class:
        stub = RuntimeHttpStub("http://runtime.example")
        stub.close()

    session_class.return_value.close.assert_called_once_with()
