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
"""Tests for CLI account authentication interceptors."""

from unittest.mock import Mock

import httpx
import pytest

from flwr.common.constant import ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListFederationsRequest,
    RefreshAuthTokensRequest,
)
from flwr.supercore.auth.typing import AccountAuthCredentials
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .auth_plugin import CliAuthPlugin
from .cli_account_auth_interceptor import CliAccountAuthHttpInterceptor


def test_http_interceptor_adds_bearer_token() -> None:
    """Send the stored access token in the Authorization header."""
    auth_plugin = Mock(spec=CliAuthPlugin)
    auth_plugin.write_tokens_to_metadata.return_value = [
        (ACCESS_TOKEN_KEY, "access-token")
    ]
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Control/ListFederations",
        message=ListFederationsRequest(),
        request=httpx.Request("POST", "https://control.example"),
    )
    response = httpx.Response(200)
    call_next = Mock(return_value=response)
    refresh_tokens = Mock()

    result = CliAccountAuthHttpInterceptor(auth_plugin, refresh_tokens).intercept(
        context, call_next
    )

    assert result is response
    assert context.request.headers["Authorization"] == "Bearer access-token"
    call_next.assert_called_once_with(context)
    refresh_tokens.assert_not_called()


@pytest.mark.parametrize("retry_status", [200, 401])
def test_http_interceptor_refreshes_expired_credentials(retry_status: int) -> None:
    """Refresh expired credentials, store them, and retry once."""
    auth_plugin = Mock(spec=CliAuthPlugin)
    auth_plugin.write_tokens_to_metadata.return_value = [
        (ACCESS_TOKEN_KEY, "old-access-token"),
        (REFRESH_TOKEN_KEY, "old-refresh-token"),
    ]
    credentials = AccountAuthCredentials(
        access_token="new-access-token",
        refresh_token="new-refresh-token",
    )
    refresh_tokens = Mock(return_value=credentials)
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Control/ListFederations",
        message=ListFederationsRequest(),
        request=httpx.Request("POST", "https://control.example"),
    )
    authorization_headers: list[str] = []

    def call_next(call_context: ProtobufRequestContext) -> httpx.Response:
        authorization_headers.append(call_context.request.headers["Authorization"])
        status_code = 401 if len(authorization_headers) == 1 else retry_status
        return httpx.Response(status_code)

    response = CliAccountAuthHttpInterceptor(
        auth_plugin, refresh_tokens
    ).intercept(context, call_next)

    assert response.status_code == retry_status
    assert authorization_headers == [
        "Bearer old-access-token",
        "Bearer new-access-token",
    ]
    refresh_tokens.assert_called_once_with("old-refresh-token")
    auth_plugin.store_tokens.assert_called_once_with(credentials)


def test_http_interceptor_does_not_refresh_the_refresh_request() -> None:
    """Do not recurse when the refresh endpoint returns unauthorized."""
    auth_plugin = Mock(spec=CliAuthPlugin)
    auth_plugin.write_tokens_to_metadata.return_value = [
        (ACCESS_TOKEN_KEY, "old-access-token"),
        (REFRESH_TOKEN_KEY, "old-refresh-token"),
    ]
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Control/RefreshAuthTokens",
        message=RefreshAuthTokensRequest(refresh_token="old-refresh-token"),
        request=httpx.Request("POST", "https://control.example"),
    )
    refresh_tokens = Mock()

    response = CliAccountAuthHttpInterceptor(
        auth_plugin, refresh_tokens
    ).intercept(context, Mock(return_value=httpx.Response(401)))

    assert response.status_code == 401
    refresh_tokens.assert_not_called()
