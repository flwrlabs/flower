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
# ==============================================================================
"""Tests for Fleet API gRPC adapter servicer."""


from typing import get_type_hints
from unittest.mock import Mock, patch

from ..grpc_rere.fleet_servicer import FleetServicer
from .grpc_adapter_servicer import GrpcAdapterServicer


def test_rpc_completion() -> None:
    """Test if the GrpcAdapter servicer can handle all requests for Fleet API."""
    # Prepare
    all_method_names = (name for name in dir(FleetServicer) if name[0].isupper())
    servicer = GrpcAdapterServicer(Mock(), Mock(), False)

    # Execute
    for method_name in all_method_names:
        method = getattr(FleetServicer, method_name)

        # Find the request type from the method's type hints
        type_hints = get_type_hints(method)
        request_type = type_hints["request"]

        # Patch the `_handle` method to simulate the request handling
        with patch(
            "flwr.server.superlink.fleet.grpc_adapter.grpc_adapter_servicer._handle"
        ) as mock_handle:
            # Simulate a request
            request = Mock(grpc_message_name=request_type.__qualname__)
            context = Mock()

            # Call the method
            try:
                servicer.SendReceive(request, context)
            except ValueError as e:
                raise AssertionError(
                    f"GrpcAdapterServicer does not support '{method_name}' RPC."
                ) from e

            # Assert: `_handle` was called with the correct parameters
            mock_handle.assert_called_once_with(
                request, context, request_type, getattr(servicer, method_name)
            )
