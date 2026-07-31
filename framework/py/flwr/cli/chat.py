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
from flwr.proto.control_pb2 import ListFederationsRequest  # pylint: disable=E0611
from flwr.proto.control_pb2_grpc import ControlStub

from .chat_app import ChatApplication
from .utils import flwr_cli_grpc_exc_handler, init_channel_from_connection


def chat() -> None:
    """Start an interactive chat session with the Flower agent."""
    superlink_connection = read_superlink_connection(CHAT_SUPERGRID_CONNECTION_NAME)

    channel = init_channel_from_connection(superlink_connection)
    stub = ControlStub(channel)
    try:
        # Verify stored credentials before showing the interactive prompt.
        with flwr_cli_grpc_exc_handler():
            stub.ListFederations(ListFederationsRequest())
        ChatApplication(stub, superlink_connection.federation).run()
    finally:
        channel.close()
