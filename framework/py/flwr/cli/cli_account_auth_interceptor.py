# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""CLI account authentication interceptors."""


from collections.abc import Callable

import httpx

from flwr.common.constant import ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY
from flwr.supercore.auth.typing import AccountAuthCredentials
from flwr.supercore.interceptors.http.utils import add_headers
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext
from flwr.supercore.utils import get_metadata_str

from .auth_plugin import CliAuthPlugin

_REFRESH_AUTH_TOKENS_METHOD = "/flwr.proto.Control/RefreshAuthTokens"


class CliAccountAuthHttpInterceptor:
    """Add CLI account authentication to protobuf-over-HTTP requests."""

    def __init__(
        self,
        auth_plugin: CliAuthPlugin,
        refresh_tokens: Callable[[str], AccountAuthCredentials],
    ) -> None:
        self.auth_plugin = auth_plugin
        self.refresh_tokens = refresh_tokens

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Add the access token, refreshing expired credentials once."""
        # Load the stored token pair and authenticate the initial request with the
        # access token. The refresh token is never sent as an HTTP header.
        metadata = self.auth_plugin.write_tokens_to_metadata([])
        if access_token := get_metadata_str(metadata, ACCESS_TOKEN_KEY):
            add_headers(
                context.request,
                {"Authorization": f"Bearer {access_token}"},
            )

        response = call_next(context)
        refresh_token = get_metadata_str(metadata, REFRESH_TOKEN_KEY)

        # Refresh only when the server identifies a bearer authentication failure.
        # Refresh requests themselves are excluded to prevent a failed refresh from
        # starting a recursive loop.
        if (
            response.status_code != httpx.codes.UNAUTHORIZED
            or response.headers.get("WWW-Authenticate") != "Bearer"
            or context.rpc_method == _REFRESH_AUTH_TOKENS_METHOD
            or refresh_token is None
        ):
            return response

        response.close()
        credentials = self.refresh_tokens(refresh_token)
        self.auth_plugin.store_tokens(credentials)

        # Retry the original request once with the new access token. Returning the
        # retry directly ensures a second 401 is propagated without another refresh.
        context.request.headers["Authorization"] = f"Bearer {credentials.access_token}"
        return call_next(context)
