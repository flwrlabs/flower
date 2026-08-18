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
"""Tests for the Runtime HTTP client."""

from unittest.mock import Mock, patch

import pytest

from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.runtime import RuntimeHttpClient

_UNARY_UNARY_PATHS = (
    "pull-pending-tasks",
    "claim-task",
    "get-run",
    "send-task-heartbeat",
    "pull-task-input",
    "push-task-output",
    "push-object",
    "pull-object",
    "confirm-message-received",
    "push-messages",
    "pull-messages",
    "push-logs",
    "get-nodes",
    "create-task",
    "start-automation",
    "push-task-message",
    "push-task-events",
    "pull-task-message",
    "record-task-usage",
    "get-connector",
)
_RESPONSE_NAME_OVERRIDES = {
    "push-messages": "PushAppMessagesResponse",
    "pull-messages": "PullAppMessagesResponse",
}


@pytest.mark.parametrize(
    "endpoint",
    _UNARY_UNARY_PATHS,
)
def test_runtime_method(endpoint: str) -> None:
    """Call one shared Runtime HTTP endpoint."""
    method_name = endpoint.title().replace("-", "")
    request = Mock()
    response = Mock()
    client = RuntimeHttpClient("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(client, method_name)(request)

    assert result is response
    call.assert_called_once()
    assert call.call_args.kwargs["path"] == f"/v1/runtime/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == f"/flwr.proto.Runtime/{method_name}"
    assert call.call_args.kwargs["request"] is request
    expected_response_name = _RESPONSE_NAME_OVERRIDES.get(
        endpoint, f"{method_name}Response"
    )
    assert call.call_args.kwargs["response_type"].__name__ == expected_response_name
