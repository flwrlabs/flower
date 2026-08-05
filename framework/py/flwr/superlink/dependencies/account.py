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

from fastapi import Request, Response

from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.auth.user_auth import UserAuthenticationService
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.superlink.auth_plugin import ControlAuthnPlugin, ControlAuthzPlugin


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
        _response: Response,
    ) -> AccountInfo:
        """Return the authenticated and authorized account for a request."""
        metadata = request.headers.items()
        valid_tokens, account = self.authn_plugin.validate_tokens_in_metadata(metadata)
        if not valid_tokens:
            raise FlowerError(
                ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
                "Authentication plugin rejected the request tokens.",
            )
        return self._authorize(
            account,
            "Tokens validated, but account info not found",
        )

    def authorize(self, account: AccountInfo) -> AccountInfo:
        """Apply the configured Control authorization plugin to an account."""
        return self._authorize(account, "Authenticated account info not found")

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


def get_user_authentication_service(request: Request) -> UserAuthenticationService:
    """Return the shared HTTP user-authentication service."""
    service = get_optional_user_authentication_service(request)
    if service is None:
        raise FlowerError(
            ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED,
            "Shared user authentication service is not initialized.",
        )
    return service


def get_optional_user_authentication_service(
    request: Request,
) -> UserAuthenticationService | None:
    """Return the shared service when an extension installed it."""
    service = getattr(request.app.state, "user_authentication_service", None)
    return service if isinstance(service, UserAuthenticationService) else None
