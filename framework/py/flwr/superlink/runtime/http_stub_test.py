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
"""Tests for the SuperLink protobuf-over-HTTP Runtime API client."""

from unittest.mock import Mock, patch

from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE

from .http_stub import RuntimeHttpStub


def test_get_nodes_sends_task_token_and_parses_response() -> None:
    """Send a task-authenticated request and parse the protobuf response."""
    request = GetNodesRequest()
    expected = GetNodesResponse(nodes=[Node(node_id=123)])
    response = Mock(content=expected.SerializeToString())

    with patch("flwr.supercore.runtime.http_stub.requests.Session") as session_class:
        session_class.return_value.post.return_value = response
        stub = RuntimeHttpStub(
            "https://runtime.example/",
            task_token="task-token",
            verify="ca.pem",
            timeout=10.0,
        )
        result = stub.GetNodes(request)

    assert result == expected
    session_class.return_value.post.assert_called_once_with(
        "https://runtime.example/v1/runtime/get-nodes",
        data=request.SerializeToString(deterministic=True),
        headers={
            "content-type": PROTOBUF_MEDIA_TYPE,
            TASK_TOKEN_HEADER: "task-token",
        },
        verify="ca.pem",
        timeout=10.0,
    )
    response.raise_for_status.assert_called_once_with()
