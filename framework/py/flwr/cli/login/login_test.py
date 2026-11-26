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

import json
from functools import partial
from unittest.mock import Mock, patch

import httpx
import pytest
from typer.testing import CliRunner

from flwr.cli.app import app
from flwr.cli.typing import SuperLinkConnection
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsResponse,
    RefreshAuthTokensRequest,
    RefreshAuthTokensResponse,
)
from flwr.supercore.auth.typing import AccountAuthCredentials, AccountAuthLoginDetails


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
        verification_uri="https://identity.example/device",
        user_code="ABCD-EFGH",
        expires_in=300,
        interval=5,
    )
    credentials = AccountAuthCredentials("access-token", "refresh-token")
    client = Mock()
    client.GetLoginDetails.return_value = response
    auth_plugin = Mock()
    auth_plugin.login.return_value = credentials

    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch("flwr.cli.login.login.migrate"),
        patch(
            "flwr.cli.login.login.read_superlink_connection",
            return_value=connection,
        ),
        patch(
            "flwr.cli.login.login.init_http_client_from_connection",
            return_value=client,
        ) as init_client,
        patch(
            "flwr.cli.login.login.load_cli_auth_plugin_from_connection",
            return_value=auth_plugin,
        ),
        patch("flwr.cli.login.login.typer.secho"),
        patch("flwr.cli.login.login.complete_signup") as signup,
    ):
        result = CliRunner().invoke(app, ["login", "remote"])

    assert result.exit_code == 0
    init_client.assert_called_once()
    assert init_client.call_args.args[0] == connection
    auth_plugin.login.assert_called_once_with(
        AccountAuthLoginDetails(
            authn_type="oidc",
            device_code="device-code",
            verification_uri_complete="",
            expires_in=300,
            interval=5,
            verification_uri="https://identity.example/device",
            user_code="ABCD-EFGH",
        ),
        client,
    )
    auth_plugin.store_tokens.assert_called_once_with(credentials)
    client.close.assert_called_once_with()
    signup.assert_not_called()


@pytest.mark.parametrize(
    ("replies", "input_text", "expected_error"),
    [
        ([(200, "alice")], None, None),
        (
            [(200, None), (409, None), (422, None), (200, "alice")],
            "taken\ninvalid!\nalice\n",
            None,
        ),
        ([(200, None)], None, "interactive terminal"),
        ([(403, None)], "", "SuperLink denied signup"),
        ([(307, None)], "", "HTTP 307"),
        ([(401, None), (200, "alice")], None, None),
        (
            [(200, None), (401, None), (200, "alice")],
            "alice\n",
            None,
        ),
        (
            [(200, None), (401, None), (401, None)],
            "alice\n",
            "SuperLink denied signup",
        ),
    ],
)
def test_login_completes_advertised_signup(
    replies: list[tuple[int, str | None]],
    input_text: str | None,
    expected_error: str | None,
) -> None:
    """Complete signup before storing credentials, using only the configured origin."""
    connection = SuperLinkConnection(name="remote", address="control.example:8443")
    credentials = AccountAuthCredentials("access-token", "refresh-token")
    control_client = Mock()
    control_client.GetLoginDetails.return_value = GetLoginDetailsResponse(
        authn_type="oidc", supports_cli_signup=True
    )
    refreshed = AccountAuthCredentials("new-access-token", "new-refresh-token")
    control_client.RefreshAuthTokens.return_value = RefreshAuthTokensResponse(
        access_token=refreshed.access_token, refresh_token=refreshed.refresh_token
    )
    auth_plugin = Mock()
    auth_plugin.login.return_value = credentials
    expected_responses = iter(replies)
    requests: list[tuple[str, dict[str, str] | None]] = []
    retry_at = next(
        (index + 1 for index, (status, _) in enumerate(replies) if status == 401),
        None,
    )

    def send(request: httpx.Request) -> httpx.Response:
        resource = {"GET": "handle", "POST": "signup"}[request.method]
        assert request.url == f"https://control.example:8443/v1/user/{resource}"
        expected = (
            refreshed
            if retry_at is not None and len(requests) >= retry_at
            else credentials
        )
        assert request.headers["Authorization"] == f"Bearer {expected.access_token}"
        auth_plugin.store_tokens.assert_not_called()
        payload = json.loads(request.content) if request.content else None
        requests.append((request.method, payload))
        status, handle = next(expected_responses)
        headers = {"Location": "https://other.example/v1/user/signup"}
        if status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        return httpx.Response(
            status,
            json={"handle": handle},
            headers=headers,
        )

    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch("flwr.cli.login.login.migrate"),
        patch(
            "flwr.cli.login.login.read_superlink_connection",
            return_value=connection,
        ),
        patch(
            "flwr.cli.login.login.init_http_client_from_connection",
            return_value=control_client,
        ),
        patch(
            "flwr.cli.login.login.load_cli_auth_plugin_from_connection",
            return_value=auth_plugin,
        ),
        patch(
            "flwr.cli.login.signup.httpx.Client",
            side_effect=partial(httpx.Client, transport=httpx.MockTransport(send)),
        ),
        patch("flwr.cli.login.signup.sys") as signup_sys,
    ):
        signup_sys.stdin.isatty.return_value = input_text is not None
        signup_sys.stdout.isatty.return_value = input_text is not None
        result = CliRunner().invoke(app, ["login", "remote"], input=input_text)

    expected_requests = [
        ("GET", None),
        *[("POST", {"handle": handle}) for handle in (input_text or "").splitlines()],
    ]
    if retry_at is not None:
        expected_requests.insert(retry_at, expected_requests[retry_at - 1])
        control_client.RefreshAuthTokens.assert_called_once_with(
            RefreshAuthTokensRequest(refresh_token=credentials.refresh_token)
        )
    else:
        control_client.RefreshAuthTokens.assert_not_called()
    assert requests == expected_requests
    if expected_error:
        assert result.exit_code == 1
        assert expected_error in result.output
        auth_plugin.store_tokens.assert_not_called()
    else:
        assert result.exit_code == 0
        assert "Login successful" in result.output
        auth_plugin.store_tokens.assert_called_once_with(
            refreshed if retry_at is not None else credentials
        )
