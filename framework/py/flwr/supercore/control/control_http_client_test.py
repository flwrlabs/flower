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
"""Tests for the Control HTTP client."""

from unittest.mock import Mock, patch

import pytest

from flwr.supercore.control import ControlHttpClient
from flwr.supercore.protobuf.client import ProtobufClient

_UNARY_UNARY_ENDPOINTS = (
    "start-run",
    "stop-run",
    "start-automation",
    "list-automations",
    "stop-automation",
    "list-runs",
    "list-run-series",
    "get-run-series",
    "list-run-series-events",
    "get-login-details",
    "get-auth-tokens",
    "refresh-auth-tokens",
    "list-connectors",
    "disconnect-connector",
    "begin-connector-oauth",
    "complete-connector-oauth",
    "pull-artifacts",
    "register-node",
    "unregister-node",
    "list-nodes",
    "list-federations",
    "list-apps",
    "add-app",
    "remove-app",
    "show-federation",
    "create-federation",
    "archive-federation",
    "add-node-to-federation",
    "remove-node-from-federation",
    "remove-account-from-federation",
    "create-invitation",
    "list-invitations",
    "accept-invitation",
    "reject-invitation",
    "revoke-invitation",
    "configure-simulation-federation",
)
_METHOD_NAME_OVERRIDES = {
    "begin-connector-oauth": "BeginConnectorOAuth",
    "complete-connector-oauth": "CompleteConnectorOAuth",
}


@pytest.mark.parametrize("endpoint", _UNARY_UNARY_ENDPOINTS)
def test_unary_unary_method(endpoint: str) -> None:
    """Delegate a Control unary-unary method to the shared HTTP client."""
    method_name = _METHOD_NAME_OVERRIDES.get(
        endpoint, endpoint.title().replace("-", "")
    )
    request = Mock()
    response = Mock()
    client = ControlHttpClient("http://control.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(client, method_name)(request)

    assert result is response
    call.assert_called_once()
    assert call.call_args.kwargs["path"] == f"/v1/control/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == f"/flwr.proto.Control/{method_name}"
    assert call.call_args.kwargs["request"] is request
    assert call.call_args.kwargs["response_type"].__name__ == f"{method_name}Response"


@pytest.mark.parametrize("endpoint", ("stream-logs", "stream-run-events"))
def test_unary_stream_method(endpoint: str) -> None:
    """Delegate a Control unary-stream method to the shared HTTP client."""
    method_name = endpoint.title().replace("-", "")
    request = Mock()
    response_stream = Mock()
    client = ControlHttpClient("http://control.example")

    with patch.object(
        ProtobufClient, "_unary_stream", return_value=response_stream
    ) as call:
        result = getattr(client, method_name)(request)

    assert result is response_stream
    call.assert_called_once()
    assert call.call_args.kwargs["path"] == f"/v1/control/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == f"/flwr.proto.Control/{method_name}"
    assert call.call_args.kwargs["request"] is request
    assert call.call_args.kwargs["response_type"].__name__ == f"{method_name}Response"
