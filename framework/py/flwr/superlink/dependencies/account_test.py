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

from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    REFRESH_TOKEN_KEY,
)
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin

from .account import AccountAccessDependency, get_account, get_authn_plugin


def _make_request(
    authorization_headers: tuple[str, ...] = ("Bearer access-token",),
    extra_headers: tuple[tuple[bytes, bytes], ...] = (),
) -> Request:
    """Return a minimal request with authentication metadata."""
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [
                (b"authorization", value.encode()) for value in authorization_headers
            ]
            + list(extra_headers),
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


@pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
def test_account_access_dependency_returns_authorized_account(scheme: str) -> None:
    """AccountAccessDependency should return the account when tokens are valid."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    account = AccountInfo(flwr_aid="aid", account_name="account")
    authn_plugin.validate_tokens_in_metadata.return_value = (True, account)
    authz_plugin.authorize.return_value = True

    request = _make_request(
        (f"{scheme} access-token",),
        (
            (ACCESS_TOKEN_KEY.encode(), b"legacy-access-token"),
            (REFRESH_TOKEN_KEY.encode(), b"legacy-refresh-token"),
        ),
    )
    result = AccountAccessDependency(authn_plugin, authz_plugin)(request)

    assert result is account
    authn_plugin.validate_tokens_in_metadata.assert_called_once_with(
        [(ACCESS_TOKEN_KEY, "access-token")]
    )
    authn_plugin.refresh_tokens.assert_not_called()
    authz_plugin.authorize.assert_called_once_with(account)


def test_account_access_dependency_allows_missing_header_for_noop_plugin() -> None:
    """No-op authentication should not require HTTP credentials."""
    authn_plugin = NoOpControlAuthnPlugin(Path(), False)
    authz_plugin = Mock()
    authz_plugin.authorize.return_value = True

    result = AccountAccessDependency(authn_plugin, authz_plugin)(_make_request(()))

    assert result == AccountInfo(
        flwr_aid=NOOP_FLWR_AID,
        account_name=NOOP_ACCOUNT_NAME,
    )
    authz_plugin.authorize.assert_called_once_with(result)


@pytest.mark.parametrize(
    "authorization_headers",
    [
        (),
        ("Basic access-token",),
        ("Bearer",),
        ("Bearer ",),
        ("Bearer first-token", "Bearer second-token"),
    ],
)
def test_account_access_dependency_rejects_invalid_authorization_header(
    authorization_headers: tuple[str, ...],
) -> None:
    """AccountAccessDependency should require exactly one Bearer credential."""
    authn_plugin = Mock()
    authz_plugin = Mock()

    with pytest.raises(FlowerError) as exc_info:
        AccountAccessDependency(authn_plugin, authz_plugin)(
            _make_request(authorization_headers)
        )

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED
    authn_plugin.validate_tokens_in_metadata.assert_not_called()
    authn_plugin.refresh_tokens.assert_not_called()
    authz_plugin.authorize.assert_not_called()


def test_account_access_dependency_rejects_legacy_headers_without_bearer() -> None:
    """Legacy Control metadata headers are not an HTTP authentication contract."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    request = _make_request(
        (),
        (
            (ACCESS_TOKEN_KEY.encode(), b"legacy-access-token"),
            (REFRESH_TOKEN_KEY.encode(), b"legacy-refresh-token"),
        ),
    )

    with pytest.raises(FlowerError) as exc_info:
        AccountAccessDependency(authn_plugin, authz_plugin)(request)

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED
    authn_plugin.validate_tokens_in_metadata.assert_not_called()
    authn_plugin.refresh_tokens.assert_not_called()
    authz_plugin.authorize.assert_not_called()


@pytest.mark.parametrize(
    ("valid_token", "account", "detail"),
    [
        (
            False,
            None,
            "Access token validation failed.",
        ),
        (
            True,
            None,
            "Token validated, but account info not found: authentication plugin "
            "returned no account.",
        ),
    ],
)
def test_account_access_dependency_rejects_unauthenticated_requests(
    valid_token: bool,
    account: AccountInfo | None,
    detail: str,
) -> None:
    """AccountAccessDependency should reject absent or incomplete authentication."""
    authn_plugin = Mock()
    authz_plugin = Mock()
    authn_plugin.validate_tokens_in_metadata.return_value = (valid_token, account)

    with pytest.raises(FlowerError) as exc_info:
        AccountAccessDependency(authn_plugin, authz_plugin)(_make_request())

    assert exc_info.value.code == ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED
    assert exc_info.value.message == detail
    assert "access-token" not in exc_info.value.message
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
        AccountAccessDependency(authn_plugin, authz_plugin)(_make_request())

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
