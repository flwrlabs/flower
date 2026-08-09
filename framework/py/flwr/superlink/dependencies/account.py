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
"""FastAPI dependency for Control API account authentication."""

from fastapi import Request
from fastapi.security.utils import get_authorization_scheme_param

from flwr.common.constant import ACCESS_TOKEN_KEY
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.auth_plugin import (
    ControlAuthnPlugin,
    ControlAuthzPlugin,
    NoOpControlAuthnPlugin,
)


class AccountAccessDependency:
    """Authenticate and authorize a Control API request.

    Instances are FastAPI dependencies. For example::

        get_account = AccountAccessDependency(authn_plugin, authz_plugin)

        @router.get("/")
        def endpoint(account: Annotated[AccountInfo, Depends(get_account)]) -> None:
            ...
    """

    def __init__(
        self,
        authn_plugin: ControlAuthnPlugin,
        authz_plugin: ControlAuthzPlugin,
    ) -> None:
        self.authn_plugin = authn_plugin
        self.authz_plugin = authz_plugin

    def __call__(
        self,
        request: Request,
    ) -> AccountInfo:
        """Return the authenticated and authorized account for a request."""
        metadata: list[tuple[str, str]]
        if isinstance(self.authn_plugin, NoOpControlAuthnPlugin):
            # The no-op plugin means account authentication is disabled. Still call
            # it to obtain the synthetic account without requiring credentials.
            metadata = []
        else:
            authorization_headers = request.headers.getlist("authorization")
            if len(authorization_headers) != 1:
                raise FlowerError(
                    ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
                    "Expected exactly one Authorization header with a Bearer token.",
                )

            scheme, access_token = get_authorization_scheme_param(
                authorization_headers[0]
            )
            if scheme.lower() != "bearer" or not access_token:
                raise FlowerError(
                    ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
                    "Authorization header does not contain a Bearer token.",
                )
            metadata = [(ACCESS_TOKEN_KEY, access_token)]

        valid_token, account = self.authn_plugin.validate_tokens_in_metadata(metadata)
        if not valid_token:
            raise FlowerError(
                ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
                "Access token validation failed.",
            )

        return self._authorize(
            account=account,
            missing_account_detail="Token validated, but account info not found",
        )

    def _authorize(
        self,
        account: AccountInfo | None,
        missing_account_detail: str,
    ) -> AccountInfo:
        """Require account information and authorization."""
        if account is None:
            raise FlowerError(
                ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
                f"{missing_account_detail}: authentication plugin returned no account.",
            )
        if not self.authz_plugin.authorize(account):
            raise FlowerError(
                ApiErrorCode.NO_PERMISSIONS,
                "Account authorization failed for "
                f"flwr_aid={account.flwr_aid!r}, "
                f"account_name={account.account_name!r}.",
            )
        return account


def get_account(
    request: Request,
) -> AccountInfo:
    """Return the account authenticated by the Control API middleware.

    Control routes authenticate requests before dependency resolution and store
    the resulting account on the request state.
    """
    account = getattr(request.state, "account", None)
    if not isinstance(account, AccountInfo):
        raise FlowerError(
            ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED,
            "SuperLink account authentication is not initialized: expected an "
            f"authenticated account, got {type(account).__name__}.",
        )
    return account


def get_authn_plugin(
    request: Request,
) -> ControlAuthnPlugin:
    """Return the configured Control authentication plugin."""
    account_access = getattr(request.app.state, "account_access_dep", None)
    if not isinstance(account_access, AccountAccessDependency):
        raise FlowerError(
            ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED,
            "SuperLink authentication is not initialized: expected ControlAuthnPlugin, "
            "got None.",
        )
    return account_access.authn_plugin
