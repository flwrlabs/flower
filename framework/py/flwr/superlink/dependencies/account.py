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
from flwr.supercore.error import (
    ApiErrorCode,
    BearerAuthenticationError,
    FlowerError,
)
from flwr.superlink.auth_plugin import ControlAuthnPlugin


class AccountAccessDependency:
    """Authenticate a Control API request.

    Instances are FastAPI dependencies. For example::

        get_account = AccountAccessDependency(authn_plugin)

        @router.get("/")
        def endpoint(account: Annotated[AccountInfo, Depends(get_account)]) -> None:
            ...
    """

    def __init__(self, authn_plugin: ControlAuthnPlugin) -> None:
        self.authn_plugin = authn_plugin

    def __call__(
        self,
        request: Request,
    ) -> AccountInfo:
        """Return the authenticated account for a request."""
        authorization_headers = request.headers.getlist("authorization")
        if len(authorization_headers) > 1:
            raise BearerAuthenticationError()

        metadata: list[tuple[str, str]] = []
        if authorization_headers:
            scheme, access_token = get_authorization_scheme_param(
                authorization_headers[0]
            )
            if scheme.lower() != "bearer" or not access_token:
                raise BearerAuthenticationError()
            metadata = [(ACCESS_TOKEN_KEY, access_token)]

        valid_token, account = self.authn_plugin.validate_tokens_in_metadata(metadata)
        if not valid_token:
            raise BearerAuthenticationError()

        return self._require_account(account)

    def _require_account(self, account: AccountInfo | None) -> AccountInfo:
        """Require account information from the authentication plugin."""
        if account is None:
            raise BearerAuthenticationError()
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
