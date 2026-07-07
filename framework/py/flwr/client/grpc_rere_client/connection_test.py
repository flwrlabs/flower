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
"""Tests for the gRPC request-response connection."""


from typing import cast
from unittest.mock import Mock, patch

from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeResponse,
    DeactivateNodeResponse,
    RegisterNodeFleetResponse,
    UnregisterNodeFleetResponse,
)
from flwr.proto.fleet_pb2_grpc import FleetStub  # pylint: disable=E0611
from flwr.proto.heartbeat_pb2 import SendNodeHeartbeatResponse  # pylint: disable=E0611
from flwr.supercore.interceptors import (
    RpcErrorTranslationClientInterceptor,
    RuntimeVersionClientInterceptor,
)
from flwr.supercore.primitives.asymmetric import generate_key_pairs

from .connection import grpc_request_response
from .node_auth_client_interceptor import NodeAuthClientInterceptor


class _Stub:
    def __init__(self, _channel: Mock) -> None:
        pass

    def RegisterNode(self, _request: object) -> RegisterNodeFleetResponse:
        return RegisterNodeFleetResponse()

    def ActivateNode(self, _request: object) -> ActivateNodeResponse:
        return ActivateNodeResponse(node_id=1)

    def DeactivateNode(self, _request: object) -> DeactivateNodeResponse:
        return DeactivateNodeResponse()

    def UnregisterNode(self, _request: object) -> UnregisterNodeFleetResponse:
        return UnregisterNodeFleetResponse()

    def SendNodeHeartbeat(
        self, _request: object, **_kwargs: object
    ) -> SendNodeHeartbeatResponse:
        return SendNodeHeartbeatResponse(success=True)


def test_grpc_request_response_installs_error_translation() -> None:
    """The request-response Fleet channel should format FlowerError RPC failures."""
    channel = Mock()
    authentication_keys = generate_key_pairs()
    retry_invoker = Mock()

    with (
        patch(
            "flwr.client.grpc_rere_client.connection.create_channel",
            return_value=channel,
        ) as create_channel,
        patch("flwr.client.grpc_rere_client.connection.wrap_stub"),
        grpc_request_response(
            server_address="localhost:9092",
            insecure=True,
            retry_invoker=retry_invoker,
            authentication_keys=authentication_keys,
            adapter_cls=cast(type[FleetStub], _Stub),
        ),
    ):
        pass

    interceptors = create_channel.call_args.kwargs["interceptors"]
    assert isinstance(interceptors[0], RpcErrorTranslationClientInterceptor)
    assert isinstance(interceptors[1], RuntimeVersionClientInterceptor)
    assert isinstance(interceptors[2], NodeAuthClientInterceptor)
