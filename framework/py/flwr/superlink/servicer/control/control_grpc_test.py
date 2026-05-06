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
"""Tests for Control API gRPC server wiring."""


from pathlib import Path
from unittest.mock import Mock, patch

from flwr.supercore.interceptors import RuntimeVersionServerInterceptor
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin, NoOpControlAuthzPlugin

from .control_grpc import run_control_api_grpc


def test_run_control_api_grpc_adds_runtime_version_interceptor() -> None:
    """Control API server should observe runtime-version metadata."""
    grpc_server = Mock()
    grpc_server.bound_address = "127.0.0.1:9093"
    with patch(
        "flwr.superlink.servicer.control.control_grpc.generic_create_grpc_server",
        return_value=grpc_server,
    ) as create_grpc_server:
        run_control_api_grpc(
            address="127.0.0.1:9093",
            state_factory=Mock(),
            objectstore_factory=Mock(),
            certificates=None,
            authn_plugin=NoOpControlAuthnPlugin(Path(), False),
            authz_plugin=NoOpControlAuthzPlugin(Path(), False),
        )

    interceptors = create_grpc_server.call_args.kwargs["interceptors"]
    assert any(
        isinstance(interceptor, RuntimeVersionServerInterceptor)
        for interceptor in interceptors
    )
