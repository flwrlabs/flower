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
"""Tests for the Flower CLI federation list command."""

from unittest.mock import MagicMock, Mock, patch

from flwr.cli.typing import SuperLinkConnection

from .ls import ls


def test_ls_uses_control_http_client() -> None:
    """List federations through the authenticated HTTP client."""
    connection = SuperLinkConnection(
        name="remote",
        address="control.example:443",
        root_certificates="/ca.pem",
    )
    output_context = MagicMock()
    output_context.__enter__.return_value = False
    client = Mock()

    with (
        patch("flwr.cli.federation.ls.cli_output_handler", return_value=output_context),
        patch("flwr.cli.federation.ls.migrate"),
        patch(
            "flwr.cli.federation.ls.read_superlink_connection",
            return_value=connection,
        ),
        patch(
            "flwr.cli.federation.ls.init_http_client_from_connection",
            return_value=client,
        ) as init_client,
        patch("flwr.cli.federation.ls._list_federations", return_value=[]),
        patch("flwr.cli.federation.ls.Console"),
    ):
        ls(Mock(args=[]), superlink="remote")

    init_client.assert_called_once_with(connection)
    client.close.assert_called_once_with()
