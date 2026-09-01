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
from typing import Any

import grpc
import httpx

from flwr.common.constant import ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
    StreamRunEventsRequest,
)
from flwr.supercore.auth.typing import AccountAuthCredentials
from flwr.supercore.interceptors.http.utils import add_headers
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext
from flwr.supercore.utils import get_metadata_str

from .auth_plugin import CliAuthPlugin

Request = StartRunRequest | StopRunRequest | StreamLogsRequest | StreamRunEventsRequest

_REFRESH_AUTH_TOKENS_METHOD = "/flwr.proto.Control/RefreshAuthTokens"


class CliAccountAuthInterceptor(
    grpc.UnaryUnaryClientInterceptor, grpc.UnaryStreamClientInterceptor  # type: ignore
):
    """CLI interceptor for account authentication.

    This interceptor adds authentication tokens to gRPC metadata for CLI requests and
    handles token refresh from response metadata.
    """

    def __init__(self, auth_plugin: CliAuthPlugin):
        self.auth_plugin = auth_plugin

    def _authenticated_call(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Send and receive tokens via metadata.

        Parameters
        ----------
        continuation : Callable[[Any, Any], Any]
            The next interceptor or handler in the chain.
        client_call_details : grpc.ClientCallDetails
            Details of the RPC call as a NamedTuple.
        request : Request
            The RPC request object.

        Returns
        -------
        grpc.Call
            The RPC response.
        """
        new_metadata = self.auth_plugin.write_tokens_to_metadata(
            client_call_details.metadata or []
        )

        details = client_call_details._replace(metadata=new_metadata)

        response = continuation(details, request)
        if response.initial_metadata():
            credentials = self.auth_plugin.read_tokens_from_metadata(
                response.initial_metadata()
            )
            # The metadata contains tokens only if they have been refreshed
            if credentials is not None:
                self.auth_plugin.store_tokens(credentials)

        return response

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Intercept a unary-unary call for account authentication.

        This method intercepts a unary-unary RPC call initiated from the CLI and adds
        the required authentication tokens to the RPC metadata.
        """
        return self._authenticated_call(continuation, client_call_details, request)

    def intercept_unary_stream(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: Request,
    ) -> grpc.Call:
        """Intercept a unary-stream call for account authentication.

        This method intercepts a unary-stream RPC call initiated from the CLI and adds
        the required authentication tokens to the RPC metadata.
        """
        return self._authenticated_call(continuation, client_call_details, request)


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

        # Refresh only after an authentication failure. Refresh requests themselves
        # are excluded to prevent a failed refresh from starting a recursive loop.
        if (
            response.status_code != httpx.codes.UNAUTHORIZED
            or context.rpc_method == _REFRESH_AUTH_TOKENS_METHOD
            or refresh_token is None
        ):
            return response

        response.close()
        credentials = self.refresh_tokens(refresh_token)
        self.auth_plugin.store_tokens(credentials)

        # Retry the original request once with the new access token. Returning the
        # retry directly ensures a second 401 is propagated without another refresh.
        context.request.headers["Authorization"] = (
            f"Bearer {credentials.access_token}"
        )
        return call_next(context)
