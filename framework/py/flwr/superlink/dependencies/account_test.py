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
"""Tests for the Control API account dependency."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request, Response

from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, FlowerError

from .account import AccountAccessDependency, get_account, get_authn_plugin


def _make_request() -> Request:  # type: ignore[type-arg]
    """Return a minimal request with authentication metadata."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"authorization", b"Bearer access-token")],
            "query_string": b"",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "scheme": "http",
        }
    )


def _make_app_request(app: FastAPI) -> Request:  # type: ignore[type-arg]
    """Return a minimal request bound to an application."""
    request = _make_request()
    request.scope["app"] = app
    return request


def test_account_access_dependency_returns_authorized_account() -> None:
    """AccountAccessDependency should return the account when tokens are valid."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid="aid", account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = True

    result = AccountAccessDependency(authn_plugin, authz_plugin)(
        _make_request(), Response()
    )

    assert result is account
    authn_plugin.validate_tokens_in_metadata.assert_called_once_with(
        [("authorization", "Bearer access-token")]
    )
    authn_plugin.refresh_tokens.assert_not_called()
    authz_plugin.authorize.assert_called_once_with(account)


def test_account_access_dependency_refreshes_tokens_and_sets_response_headers() -> None:
    """AccountAccessDependency returns an authorized account after token refresh."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid="aid", account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
    authn_plugin.refresh_tokens.return_value = (
        [("x-access-token", "new-token"), ("x-refresh-token", b"new-refresh")],
        account,
    )
    authz_plugin.authorize.return_value = True
    response = Response()

    result = AccountAccessDependency(authn_plugin, authz_plugin)(
        _make_request(), response
    )

    assert result is account
    assert response.headers.get("x-access-token") == "new-token"
    assert response.headers.get("x-refresh-token") == "new-refresh"
    authz_plugin.authorize.assert_called_once_with(account)


@pytest.mark.parametrize(
    ("valid_tokens", "tokens", "account", "detail"),
    [
        (
            True,
            None,
            None,
            "Tokens validated, but account info not found: authentication plugin "
            "returned no account.",
        ),
        (
            False,
            None,
            None,
            "Token refresh failed: authentication plugin returned no tokens.",
        ),
        (
            False,
            [("x-access-token", "new-token")],
            None,
            "Tokens refreshed, but account info not found: authentication plugin "
            "returned no account.",
        ),
    ],
)
def test_account_access_dependency_rejects_unauthenticated_requests(
    valid_tokens: bool,
    tokens: list[tuple[str, str]] | None,
    account: AccountInfo | None,
    detail: str,
) -> None:
    """AccountAccessDependency should reject absent or incomplete authentication."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (valid_tokens, account)
    authn_plugin.refresh_tokens.return_value = (tokens, account)

    with pytest.raises(FlowerError) as exc_info:
        AccountAccessDependency(authn_plugin, authz_plugin)(_make_request(), Response())

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED
    assert exc_info.value.message == detail
    authz_plugin.authorize.assert_not_called()


def test_account_access_dependency_rejects_unauthorized_account() -> None:
    """AccountAccessDependency should reject accounts denied by authorization."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid="aid", account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = False

    with pytest.raises(FlowerError) as exc_info:
        AccountAccessDependency(authn_plugin, authz_plugin)(_make_request(), Response())

    assert exc_info.value.code == ApiErrorCode.NO_PERMISSIONS
    assert exc_info.value.message == (
        "Account authorization failed for flwr_aid='aid', account_name='account'."
    )


def test_get_authn_plugin_returns_configured_plugin() -> None:
    """get_authn_plugin should return the configured authentication plugin."""
    app = FastAPI()
    authn_plugin = Mock()
    app.state.account_access_dep = AccountAccessDependency(authn_plugin, Mock())

    assert get_authn_plugin(_make_app_request(app)) is authn_plugin


def test_get_authn_plugin_raises_when_plugin_is_missing() -> None:
    """get_authn_plugin should fail clearly when the app is not configured."""
    with pytest.raises(FlowerError) as exc_info:
        get_authn_plugin(_make_app_request(FastAPI()))

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED
    assert exc_info.value.message == (
        "SuperLink authentication is not initialized: expected ControlAuthnPlugin, "
        "got None."
    )


def test_get_account_raises_when_dependency_is_missing() -> None:
    """get_account should fail clearly when the app is not configured."""
    with pytest.raises(FlowerError) as exc_info:
        get_account(_make_app_request(FastAPI()), Response())

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED
    assert (
        exc_info.value.message
        == "SuperLink account authentication is not initialized: expected "
        "AccountAccessDependency, got NoneType."
    )
