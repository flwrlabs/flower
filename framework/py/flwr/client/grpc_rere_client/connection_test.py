# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
# ======================================================================
"""Tests for the gRPC request-response connection helpers."""

from flwr.proto.fleet_pb2 import ActivateNodeResponse  # pylint: disable=E0611
from flwr.supercore.heartbeat import HeartbeatConfig

from .connection import _heartbeat_config_from_response


def test_heartbeat_config_adopts_superlink_profile() -> None:
    """Use the timings provided by the SuperLink activation response."""
    response = ActivateNodeResponse.FromString(
        ActivateNodeResponse(
            heartbeat_interval=60,
            heartbeat_rpc_timeout=45,
            app_heartbeat_rpc_timeout=180,
            clientapp_token_lease=3600,
        ).SerializeToString()
    )

    assert _heartbeat_config_from_response(response) == HeartbeatConfig(
        interval=60,
        rpc_timeout=45,
        app_rpc_timeout=180,
        clientapp_token_lease=3600,
    )
