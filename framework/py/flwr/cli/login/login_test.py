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
"""Tests for the Flower CLI login command."""

from unittest.mock import MagicMock, Mock, patch

from flwr.cli.typing import SuperLinkConnection
from flwr.proto.control_pb2 import GetLoginDetailsResponse  # pylint: disable=E0611
from flwr.supercore.auth.typing import AccountAuthCredentials

from .login import login


def test_login_uses_control_http_client() -> None:
    """Use one HTTP client for login details and token polling."""
    connection = SuperLinkConnection(
        name="remote",
        address="control.example:443",
        root_certificates="/ca.pem",
    )
    response = GetLoginDetailsResponse(
        authn_type="oidc",
        device_code="device-code",
        verification_uri_complete="https://identity.example/device",
        expires_in=300,
        interval=5,
    )
    credentials = AccountAuthCredentials("access-token", "refresh-token")
    client = Mock()
    client.GetLoginDetails.return_value = response
    client_context = MagicMock()
    client_context.__enter__.return_value = client
    auth_plugin = Mock()
    auth_plugin.login.return_value = credentials
    interceptor = Mock()

    with (
        patch("flwr.cli.login.login.warn_if_federation_config_overrides"),
        patch("flwr.cli.login.login.migrate"),
        patch(
            "flwr.cli.login.login.read_superlink_connection",
            return_value=connection,
        ),
        patch(
            "flwr.cli.login.login.load_certificate_in_connection",
            return_value=b"certificate",
        ),
        patch(
            "flwr.cli.login.login.RuntimeVersionHttpInterceptor",
            return_value=interceptor,
        ),
        patch(
            "flwr.cli.login.login.ControlHttpClient.from_server_address",
            return_value=client_context,
        ) as client_factory,
        patch(
            "flwr.cli.login.login.load_cli_auth_plugin_from_connection",
            return_value=auth_plugin,
        ),
        patch("flwr.cli.login.login.typer.secho"),
    ):
        login(Mock(args=[]))

    client_factory.assert_called_once_with(
        server_address="control.example:443",
        insecure=False,
        root_certificates=b"certificate",
        interceptors=[interceptor],
    )
    auth_plugin.login.assert_called_once()
    assert auth_plugin.login.call_args.args[1] is client
    auth_plugin.store_tokens.assert_called_once_with(credentials)
    client_context.__exit__.assert_called_once()
