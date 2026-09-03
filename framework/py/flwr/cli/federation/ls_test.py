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

from unittest.mock import ANY, MagicMock, Mock, patch

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
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    auth_plugin = Mock()
    runtime_interceptor = Mock()
    auth_interceptor = Mock()

    with (
        patch("flwr.cli.federation.ls.cli_output_handler", return_value=output_context),
        patch("flwr.cli.federation.ls.migrate"),
        patch(
            "flwr.cli.federation.ls.read_superlink_connection",
            return_value=connection,
        ),
        patch(
            "flwr.cli.federation.ls.load_certificate_in_connection",
            return_value=b"certificate",
        ),
        patch(
            "flwr.cli.federation.ls.load_cli_auth_plugin_from_connection",
            return_value=auth_plugin,
        ),
        patch(
            "flwr.cli.federation.ls.RuntimeVersionHttpInterceptor",
            return_value=runtime_interceptor,
        ),
        patch(
            "flwr.cli.federation.ls.CliAccountAuthHttpInterceptor",
            return_value=auth_interceptor,
        ) as auth_interceptor_factory,
        patch(
            "flwr.cli.federation.ls.ControlHttpClient.from_server_address",
            return_value=client_context,
        ) as client_factory,
        patch("flwr.cli.federation.ls._list_federations", return_value=[]),
        patch("flwr.cli.federation.ls.Console"),
        patch("flwr.cli.federation.ls.log_superlink_connection"),
    ):
        ls(Mock(args=[]), superlink="remote")

    auth_plugin.load_tokens.assert_called_once_with()
    auth_interceptor_factory.assert_called_once_with(
        auth_plugin,
        refresh_tokens=ANY,
    )
    client_factory.assert_called_once_with(
        server_address="control.example:443",
        insecure=False,
        root_certificates=b"certificate",
        interceptors=[runtime_interceptor, auth_interceptor],
    )
    client_context.__exit__.assert_called_once()
