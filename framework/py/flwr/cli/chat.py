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
"""Flower command line interface `chat` command."""


from flwr.cli.constant import CHAT_SUPERGRID_CONNECTION_NAME
from flwr.cli.flower_config import read_superlink_connection
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    ListFederationsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.supercore.constant import DEFAULT_FEDERATION_SIMULATION

from .chat_app import ChatApplication
from .utils import (
    flwr_cli_grpc_exc_handler,
    init_channel_from_connection,
    load_cli_auth_plugin_from_connection,
)


def chat() -> None:
    """Start an interactive chat session with the Flower agent."""
    superlink_connection = read_superlink_connection(CHAT_SUPERGRID_CONNECTION_NAME)

    if superlink_connection.address is None:
        raise ValueError("The SuperGrid connection has no address.")
    auth_plugin = load_cli_auth_plugin_from_connection(superlink_connection.address)
    channel = init_channel_from_connection(superlink_connection, auth_plugin)
    stub = ControlStub(channel)
    try:
        # Verify stored credentials before showing the interactive prompt.
        with flwr_cli_grpc_exc_handler():
            response: ListFederationsResponse = stub.ListFederations(
                ListFederationsRequest()
            )
        federation = _resolve_chat_federation(
            superlink_connection.federation, list(response.federations)
        )
        ChatApplication(stub, federation, auth_plugin).run()
    finally:
        channel.close()


def _resolve_chat_federation(
    configured_federation: str | None, federations: list[Federation]
) -> str | None:
    """Resolve the federation used for agent discovery and chat runs."""
    if configured_federation is not None:
        return configured_federation

    active_federations = [
        federation for federation in federations if not federation.archived
    ]
    default_suffix = f"/{DEFAULT_FEDERATION_SIMULATION}"
    for federation in active_federations:
        if federation.name.endswith(default_suffix):
            return federation.name
    return active_federations[0].name if active_federations else None
