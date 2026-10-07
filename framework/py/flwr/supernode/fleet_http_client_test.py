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
"""Tests for the Fleet HTTP client."""

from unittest.mock import Mock, patch

import pytest

from flwr.supercore.protobuf.client import ProtobufClient

from .fleet_http_client import FleetHttpClient

_ENDPOINTS = (
    "register-node",
    "activate-node",
    "deactivate-node",
    "unregister-node",
    "send-node-heartbeat",
    "pull-messages",
    "push-messages",
    "get-run",
    "get-fab",
    "push-object",
    "pull-object",
    "confirm-message-received",
)
_MESSAGE_NAME_OVERRIDES = {
    "register-node": "RegisterNodeFleet",
    "unregister-node": "UnregisterNodeFleet",
}


@pytest.mark.parametrize("endpoint", _ENDPOINTS)
def test_fleet_method(endpoint: str) -> None:
    """Map each Fleet RPC to its protobuf HTTP endpoint."""
    method_name = endpoint.title().replace("-", "")
    request = Mock()
    response = Mock()
    client = FleetHttpClient("http://fleet.example")

    with patch.object(ProtobufClient, "_unary_unary", return_value=response) as call:
        result = getattr(client, method_name)(request)

    assert result is response
    assert call.call_args.kwargs["path"] == f"/v1/fleet/{endpoint}"
    assert call.call_args.kwargs["rpc_method"] == (f"/flwr.proto.Fleet/{method_name}")
    assert call.call_args.kwargs["request"] is request
    message_name = _MESSAGE_NAME_OVERRIDES.get(endpoint, method_name)
    assert call.call_args.kwargs["response_type"].__name__ == (
        f"{message_name}Response"
    )
    client.close()
