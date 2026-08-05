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

from .account import (
    AccountAccessDependency,
    get_account,
    get_authn_plugin,
    get_optional_user_authentication_service,
    get_user_authentication_service,
)


def _make_request() -> Request:
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


def _make_app_request(app: FastAPI) -> Request:
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


@pytest.mark.parametrize(
    ("valid_tokens", "account", "detail"),
    [
        (
            True,
            None,
            "Tokens validated, but account info not found: authentication plugin "
            "returned no account.",
        ),
        (
            False,
            None,
            "Authentication plugin rejected the request tokens.",
        ),
    ],
)
def test_account_access_dependency_rejects_unauthenticated_requests(
    valid_tokens: bool,
    account: AccountInfo | None,
    detail: str,
) -> None:
    """AccountAccessDependency should reject absent or incomplete authentication."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (valid_tokens, account)

    with pytest.raises(FlowerError) as exc_info:
        AccountAccessDependency(authn_plugin, authz_plugin)(_make_request(), Response())

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED
    assert exc_info.value.message == detail
    authn_plugin.refresh_tokens.assert_not_called()
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


def test_get_account_raises_when_authentication_middleware_did_not_run() -> None:
    """get_account should require the account saved by the middleware."""
    with pytest.raises(FlowerError) as exc_info:
        get_account(_make_app_request(FastAPI()))

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED
    assert (
        exc_info.value.message
        == "SuperLink account authentication is not initialized: expected an "
        "authenticated account, got NoneType."
    )


def test_user_authentication_service_dependencies() -> None:
    """Return an injected structural service and reject a missing one."""
    app = FastAPI()
    service = Mock()
    service.authenticate_user = Mock()
    service.authenticate_for_bootstrap = Mock()
    service.refresh_tokens = Mock()
    service.start_device_authorization = Mock()
    service.exchange_device_code = Mock()
    app.state.user_authentication_service = service
    request = _make_app_request(app)

    assert get_user_authentication_service(request) is service
    assert get_optional_user_authentication_service(request) is service

    missing_request = _make_app_request(FastAPI())
    assert get_optional_user_authentication_service(missing_request) is None
    with pytest.raises(FlowerError) as exc_info:
        get_user_authentication_service(missing_request)
    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED
