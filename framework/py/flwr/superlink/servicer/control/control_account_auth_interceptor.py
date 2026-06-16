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
"""Flower Control API interceptor."""


import contextvars
from collections.abc import Callable
from typing import Any

import grpc

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    StartRunRequest,
    StartRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
)
from flwr.supercore.auth.typing import AccountInfo
from flwr.superlink.auth_plugin import ControlAuthnPlugin, ControlAuthzPlugin

Request = (
    StartRunRequest | StreamLogsRequest | GetLoginDetailsRequest | GetAuthTokensRequest
)

Response = (
    StartRunResponse
    | StreamLogsResponse
    | GetLoginDetailsResponse
    | GetAuthTokensResponse
)


shared_account_info: contextvars.ContextVar[AccountInfo | None] = (
    contextvars.ContextVar("account_info", default=None)
)


def get_current_account_info() -> AccountInfo:
    """Get the current account info from context, or return a default if not set."""
    account_info = shared_account_info.get()
    if account_info is None:
        return AccountInfo(flwr_aid=None, account_name=None)
    return account_info


class ControlAccountAuthInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Control API interceptor for account authentication."""

    def __init__(
        self,
        authn_plugin: ControlAuthnPlugin,
        authz_plugin: ControlAuthzPlugin,
    ):
        self.authn_plugin = authn_plugin
        self.authz_plugin = authz_plugin

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Flower server interceptor authentication logic.

        Intercept all unary-unary/unary-stream calls from users and authenticate users
        by validating auth metadata sent by the user. Continue RPC call if user is
        authenticated, else, terminate RPC call by setting context to abort.
        """
        # Only apply to Control service
        if not handler_call_details.method.startswith("/flwr.proto.Control/"):
            return continuation(handler_call_details)

        # One of the method handlers in
        # `flwr.superlink.servicer.control.ControlServicer`
        method_handler: grpc.RpcMethodHandler = continuation(handler_call_details)
        return self._generic_auth_unary_method_handler(method_handler)

    def _generic_auth_unary_method_handler(
        self, method_handler: grpc.RpcMethodHandler
    ) -> grpc.RpcMethodHandler:
        def _generic_method_handler(
            request: Request,
            context: grpc.ServicerContext,
        ) -> Response:
            call = method_handler.unary_unary or method_handler.unary_stream
            metadata = context.invocation_metadata()

            # Intercept GetLoginDetails and GetAuthTokens requests, and return
            # the response without authentication
            if isinstance(request, (GetLoginDetailsRequest | GetAuthTokensRequest)):
                return call(request, context)  # type: ignore

            # For other requests, check if the account is authenticated
            valid_tokens, account_info = self.authn_plugin.validate_tokens_in_metadata(
                metadata
            )
            if valid_tokens:
                if account_info is None:
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "Tokens validated, but account info not found",
                    )
                    raise grpc.RpcError()
                # Store account info in contextvars for authenticated accounts
                shared_account_info.set(account_info)
                # Check if the account is authorized
                if not self.authz_plugin.authorize(account_info):
                    context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        "❗️ Account not authorized. "
                        "Please contact the SuperLink administrator.",
                    )
                    raise grpc.RpcError()
                return call(request, context)  # type: ignore

            # If the account is not authenticated, refresh tokens
            tokens, account_info = self.authn_plugin.refresh_tokens(metadata)
            if tokens is not None:
                if account_info is None:
                    context.abort(
                        grpc.StatusCode.UNAUTHENTICATED,
                        "Tokens refreshed, but account info not found",
                    )
                    raise grpc.RpcError()
                # Store account info in contextvars for authenticated accounts
                shared_account_info.set(account_info)
                # Check if the account is authorized
                if not self.authz_plugin.authorize(account_info):
                    context.abort(
                        grpc.StatusCode.PERMISSION_DENIED,
                        "❗️ Account not authorized. "
                        "Please contact the SuperLink administrator.",
                    )
                    raise grpc.RpcError()

                context.send_initial_metadata(tokens)
                return call(request, context)  # type: ignore

            context.abort(grpc.StatusCode.UNAUTHENTICATED, "Access denied")
            raise grpc.RpcError()  # This line is unreachable

        if method_handler.unary_unary:
            message_handler = grpc.unary_unary_rpc_method_handler
        else:
            message_handler = grpc.unary_stream_rpc_method_handler
        return message_handler(
            _generic_method_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )
