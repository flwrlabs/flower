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
"""Tests for provider device verification formats in the Flower CLI."""

from unittest.mock import Mock, patch

import pytest

from flwr.cli.auth_plugin.auth_plugin import LoginError
from flwr.cli.auth_plugin.oidc_cli_plugin import OidcCliPlugin
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
)
from flwr.supercore.auth.typing import AccountAuthCredentials, AccountAuthLoginDetails


@pytest.mark.parametrize("complete_uri", ["https://identity.example/complete", ""])
def test_device_login_uses_provider_verification_format(
    complete_uri: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Display a split user code before opening the unmodified verification URL."""
    details = AccountAuthLoginDetails(
        authn_type="oidc",
        device_code="private-device-code",
        verification_uri_complete=complete_uri,
        expires_in=300,
        interval=5,
        verification_uri="https://identity.example/verify",
        user_code="ABCD-EFGH",
    )
    client = Mock()
    client.GetAuthTokens.return_value = GetAuthTokensResponse(
        access_token="access", refresh_token="refresh"
    )

    def check_browser_output(_url: str) -> None:
        output = capsys.readouterr().out
        assert ("ABCD-EFGH" in output) == (not complete_uri)
        assert "private-device-code" not in output

    with (
        patch(
            "flwr.cli.auth_plugin.oidc_cli_plugin.webbrowser.open",
            side_effect=check_browser_output,
        ) as browser,
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.time.sleep"),
    ):
        credentials = OidcCliPlugin.login(details, client)

    verification_uri = complete_uri or details.verification_uri
    browser.assert_called_once_with(verification_uri)
    output = capsys.readouterr().out
    assert verification_uri in output
    assert "ABCD-EFGH" not in output
    assert "private-device-code" not in output
    client.GetAuthTokens.assert_called_once_with(
        GetAuthTokensRequest(device_code="private-device-code")
    )
    assert credentials == AccountAuthCredentials("access", "refresh")


@pytest.mark.parametrize(
    ("verification_uri", "user_code"),
    [("", ""), ("https://identity.example/verify", ""), ("", "ABCD-EFGH")],
)
def test_device_login_rejects_incomplete_verification_details(
    verification_uri: str, user_code: str
) -> None:
    """Fail before opening a browser or polling when verification is impossible."""
    details = AccountAuthLoginDetails(
        authn_type="oidc",
        device_code="private-device-code",
        verification_uri_complete="",
        expires_in=300,
        interval=5,
        verification_uri=verification_uri,
        user_code=user_code,
    )
    client = Mock()

    with (
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.webbrowser.open") as browser,
        pytest.raises(LoginError, match="verification URL and code"),
    ):
        OidcCliPlugin.login(details, client)

    browser.assert_not_called()
    client.GetAuthTokens.assert_not_called()


def test_device_polling_keeps_increased_interval() -> None:
    """Accumulate slow-downs and keep the interval across pending responses."""
    details = AccountAuthLoginDetails("oidc", "device", "https://example.test", 120, 0)
    client = Mock()
    client.GetAuthTokens.side_effect = [
        GetAuthTokensResponse(),
        GetAuthTokensResponse(error="slow_down"),
        GetAuthTokensResponse(error="authorization_pending"),
        GetAuthTokensResponse(error="slow_down"),
        GetAuthTokensResponse(access_token="access", refresh_token="refresh"),
    ]
    with (
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.webbrowser.open"),
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.time.sleep") as sleep,
    ):
        assert OidcCliPlugin.login(details, client) == AccountAuthCredentials(
            "access", "refresh"
        )
    assert [c.args[0] for c in sleep.call_args_list] == [5, 5, 10, 10, 15]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        ("access_denied", "denied"),
        ("expired_token", "expired"),
        ("invalid_grant", "failed"),
    ],
)
def test_terminal_device_poll_stops_login(error: str, message: str) -> None:
    """Stop polling immediately on a terminal response."""
    details = AccountAuthLoginDetails("oidc", "device", "https://example.test", 120, 5)
    client = Mock()
    client.GetAuthTokens.return_value = GetAuthTokensResponse(error=error)
    with (
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.webbrowser.open"),
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.time.sleep"),
        pytest.raises(LoginError, match=message),
    ):
        OidcCliPlugin.login(details, client)
    client.GetAuthTokens.assert_called_once()


def test_backoff_respects_device_code_deadline() -> None:
    """Avoid polling again if the increased interval exceeds the time left."""
    details = AccountAuthLoginDetails("oidc", "device", "https://example.test", 7, 5)
    client = Mock()
    client.GetAuthTokens.return_value = GetAuthTokensResponse(error="slow_down")
    with (
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.webbrowser.open"),
        patch("flwr.cli.auth_plugin.oidc_cli_plugin.time") as clock,
        pytest.raises(LoginError, match="timed out"),
    ):
        clock.monotonic.side_effect = [0, 0, 5, 5, 7]
        OidcCliPlugin.login(details, client)
    assert [c.args[0] for c in clock.sleep.call_args_list] == [5, 2]
    client.GetAuthTokens.assert_called_once()
