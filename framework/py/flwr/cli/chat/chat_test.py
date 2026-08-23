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
"""Tests for the CLI `chat` command."""


import importlib
from unittest.mock import Mock, patch

import click
import pytest

from flwr.cli.constant import CHAT_SUPERGRID_CONNECTION_NAME
from flwr.cli.typing import SuperLinkConnection
from flwr.proto.control_pb2 import ListFederationsResponse  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611

chat_module = importlib.import_module("flwr.cli.chat.chat")


def test_chat_requires_login_before_interactive_application() -> None:
    """Chat should fail before launching the app if the user has not logged in."""
    superlink_connection = SuperLinkConnection(
        name=CHAT_SUPERGRID_CONNECTION_NAME,
        address="supergrid.flower.ai",
    )
    channel = Mock()
    stub = Mock()
    stub.ListFederations.side_effect = click.ClickException(
        "Missing authentication tokens. Please login first."
    )

    with (
        patch.object(
            chat_module,
            "read_superlink_connection",
            return_value=superlink_connection,
        ),
        patch.object(
            chat_module,
            "init_channel_from_connection",
            return_value=channel,
        ),
        patch.object(chat_module, "load_cli_auth_plugin_from_connection"),
        patch.object(chat_module, "ControlStub", return_value=stub),
        patch.object(chat_module, "ChatApplication") as mock_chat_application,
    ):
        with pytest.raises(click.ClickException, match="login first"):
            chat_module.chat()

    mock_chat_application.assert_not_called()
    channel.close.assert_called_once()


def test_chat_runs_interactive_application() -> None:
    """Chat should launch the interactive application after authentication."""
    superlink_connection = SuperLinkConnection(
        name=CHAT_SUPERGRID_CONNECTION_NAME,
        address="supergrid.flower.ai",
    )
    channel = Mock()
    stub = Mock()
    federations = [Federation(name="@flower/flower-agent-execution")]
    stub.ListFederations.return_value = ListFederationsResponse(federations=federations)
    auth_plugin = Mock()

    with (
        patch.object(
            chat_module,
            "read_superlink_connection",
            return_value=superlink_connection,
        ),
        patch.object(
            chat_module,
            "init_channel_from_connection",
            return_value=channel,
        ) as mock_init_channel,
        patch.object(
            chat_module,
            "load_cli_auth_plugin_from_connection",
            return_value=auth_plugin,
        ),
        patch.object(chat_module, "ControlStub", return_value=stub),
        patch.object(chat_module, "ChatApplication") as mock_chat_application,
    ):
        chat_module.chat()

    stub.ListFederations.assert_called_once()
    mock_init_channel.assert_called_once_with(superlink_connection, auth_plugin)
    mock_chat_application.assert_called_once_with(stub, federations, auth_plugin)
    mock_chat_application.return_value.run.assert_called_once_with()
    channel.close.assert_called_once()
