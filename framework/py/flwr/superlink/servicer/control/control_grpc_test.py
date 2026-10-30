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


from unittest.mock import Mock, patch

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Agent,
    AgentConfig,
    CreateAgentRequest,
    ListAgentsRequest,
)
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.interceptors import (
    RpcErrorTranslationServerInterceptor,
    RuntimeVersionServerInterceptor,
)
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin

from .control_grpc import run_control_api_grpc
from .control_servicer import ControlServicer


def test_agent_rpc_methods_use_authenticated_owner() -> None:
    """Create and list through the shared handlers using the caller's account."""
    factory = Mock()
    state = factory.state.return_value
    account = AccountInfo(flwr_aid="alice", account_name="alice")
    config = AgentConfig(name="Research Agent", app_spec="@flower/research")
    agent = Agent(id=42, config=config)
    state.create_agent.return_value = agent
    state.list_agents.return_value = [agent]
    servicer = ControlServicer(factory, Mock(), NoOpControlAuthnPlugin())

    with patch(
        "flwr.superlink.servicer.control.control_servicer._get_account",
        return_value=account,
    ):
        created = servicer.CreateAgent(CreateAgentRequest(config=config), Mock())
        listed = servicer.ListAgents(ListAgentsRequest(), Mock())

    assert created.agent == agent
    assert listed.agents == [agent]
    state.create_agent.assert_called_once_with(flwr_aid="alice", config=config)
    state.list_agents.assert_called_once_with("alice")


def test_run_control_api_grpc_adds_expected_interceptors() -> None:
    """Control API server should install shared server interceptors."""
    grpc_server = Mock()
    grpc_server.bound_address = "127.0.0.1:9093"
    with (
        patch(
            "flwr.superlink.servicer.control.control_grpc.get_license_plugin",
            return_value=None,
        ),
        patch(
            "flwr.superlink.servicer.control.control_grpc.generic_create_grpc_server",
            return_value=grpc_server,
        ) as create_grpc_server,
    ):
        run_control_api_grpc(
            address="127.0.0.1:9093",
            state_factory=Mock(),
            objectstore_factory=Mock(),
            certificates=None,
            authn_plugin=NoOpControlAuthnPlugin(),
        )

    interceptors = create_grpc_server.call_args.kwargs["interceptors"]
    assert any(
        isinstance(interceptor, RpcErrorTranslationServerInterceptor)
        for interceptor in interceptors
    )
    assert any(
        isinstance(interceptor, RuntimeVersionServerInterceptor)
        for interceptor in interceptors
    )
