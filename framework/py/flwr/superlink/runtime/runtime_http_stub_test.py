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
"""Tests for the HTTP SuperLink Runtime API stub."""

from unittest.mock import Mock, patch

import pytest

from flwr.supercore.protobuf.client import ProtobufClient
from flwr.superlink.runtime import RuntimeHttpStub

_UNARY_UNARY_PATHS = (
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


@pytest.mark.parametrize(
    "endpoint",
    _UNARY_UNARY_PATHS,
)
def test_serverapp_runtime_method(endpoint: str) -> None:
    """Call one Runtime endpoint required by flwr-serverapp."""
    method_name = endpoint.title().replace("-", "")
    request = Mock()
    response = Mock()
    stub = RuntimeHttpStub("http://runtime.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(stub, method_name)(request)

    assert result is response
    call.assert_called_once()
    assert call.call_args.kwargs["path"] == f"/v1/runtime/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == f"/flwr.proto.Runtime/{method_name}"
    assert call.call_args.kwargs["request"] is request
    assert call.call_args.kwargs["response_type"].__name__ == f"{method_name}Response"
