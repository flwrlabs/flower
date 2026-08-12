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

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
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
_METHOD = "/flwr.proto.Runtime/PullPendingTasks"


def test_pull_pending_tasks_sends_and_receives_protobuf() -> None:
    """Send a protobuf request and parse the protobuf response."""
    request = PullPendingTasksRequest()
    expected = PullPendingTasksResponse(tasks=[Task(task_id=123)])
    response = Mock(content=expected.SerializeToString())

    with patch("flwr.supercore.runtime.http_stub.requests.Session") as session_class:
        session_class.return_value.post.return_value = response
        stub = RuntimeHttpStub(
            "https://runtime.example/", verify="ca.pem", timeout=10.0
        )
        result = stub.PullPendingTasks(request)

    assert result == expected
    session_class.return_value.post.assert_called_once_with(
        "https://runtime.example/v1/runtime/pull-pending-tasks",
        data=request.SerializeToString(deterministic=True),
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
def test_pull_pending_tasks_adds_superexec_authentication(
    _token_hex: Mock, _now: Mock
) -> None:
    """Sign PullPendingTasks when a SuperExec secret is configured."""
    request = PullPendingTasksRequest()
    response = Mock(content=PullPendingTasksResponse().SerializeToString())
    master_secret = b"master-secret"
    body_sha256 = compute_request_body_sha256(request)

    with patch("flwr.supercore.runtime.http_stub.requests.Session") as session_class:
        session_class.return_value.post.return_value = response
        stub = RuntimeHttpStub(
            "http://runtime.example", superexec_auth_secret=master_secret
        )
        stub.PullPendingTasks(request)

    headers = session_class.return_value.post.call_args.kwargs["headers"]
    assert headers == {
        "content-type": PROTOBUF_MEDIA_TYPE,
        SUPEREXEC_AUTH_TIMESTAMP_HEADER: str(_TIMESTAMP),
        SUPEREXEC_AUTH_NONCE_HEADER: _NONCE,
        SUPEREXEC_AUTH_BODY_SHA256_HEADER: body_sha256,
        SUPEREXEC_AUTH_SIGNATURE_HEADER: compute_superexec_signature(
            auth_secret=derive_auth_secret(master_secret),
            method=_METHOD,
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
