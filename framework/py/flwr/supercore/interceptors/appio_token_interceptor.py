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
"""Token-based AppIo interceptors for short-term auth coverage."""


from __future__ import annotations

from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any, NoReturn, Protocol, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.supercore.auth import (
    CLIENTAPPIO_METHOD_AUTH_POLICY,
    SERVERAPPIO_METHOD_AUTH_POLICY,
    MethodTokenPolicy,
)
from flwr.supercore.utils import find_metadata_keys, get_metadata_str

APP_TOKEN_HEADER = "flwr-app-token"
AUTHENTICATION_FAILED_MESSAGE = "Authentication failed."


_current_task_id: ContextVar[int | None] = ContextVar("current_task_id", default=None)


class _TokenState(Protocol):
    """State methods required by token auth."""

    def get_run_id_by_token(self, token: str) -> int | None:
        """Return the run id associated with token, if it exists."""

    def verify_token(self, run_id: int, token: str) -> bool:
        """Return whether token is valid for run_id."""

    def get_task_id_by_token(self, token: str) -> int | None:
        """Return the task ID associated with the task token, if valid."""


def _abort_auth_denied(context: grpc.ServicerContext) -> NoReturn:
    context.abort(grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE)
    raise RuntimeError("Should not reach this point")


def _unauthenticated_terminator() -> grpc.RpcMethodHandler:
    def _terminate(_request: GrpcMessage, context: grpc.ServicerContext) -> GrpcMessage:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE)
        raise RuntimeError("Should not reach this point")

    return grpc.unary_unary_rpc_method_handler(_terminate)


class AppIoTokenClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Attach App token metadata to outbound unary RPCs."""

    def __init__(self, token: str) -> None:
        self._token = token

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Add/replace the App token metadata on outbound unary requests."""
        metadata = tuple(client_call_details.metadata or ())
        if find_metadata_keys(metadata, (APP_TOKEN_HEADER,)):
            raise RuntimeError(
                f"{APP_TOKEN_HEADER} already present in outbound metadata."
            )
        metadata += ((APP_TOKEN_HEADER, self._token),)
        details = client_call_details._replace(metadata=metadata)
        return continuation(details, request)


class AppIoTokenServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Validate AppIo tokens with per-method token policies."""

    def __init__(
        self,
        state_provider: Callable[[], _TokenState],
        method_auth_policy: Mapping[str, MethodTokenPolicy],
    ) -> None:
        self._state_provider = state_provider
        self._method_auth_policy = dict(method_auth_policy)

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Enforce per-method token policy for incoming unary RPC calls."""
        method = handler_call_details.method
        policy = self._method_auth_policy.get(method)
        if policy is None:
            return _unauthenticated_terminator()

        method_handler = continuation(handler_call_details)
        if method_handler is None:
            return _unauthenticated_terminator()

        # Future PR: lift mechanism-specific details into a shared auth abstraction.
        if not policy.requires_token:
            return method_handler

        if method_handler.unary_unary is None:
            return _unauthenticated_terminator()

        unary_handler = cast(
            Callable[[GrpcMessage, grpc.ServicerContext], GrpcMessage],
            method_handler.unary_unary,
        )
        token = get_metadata_str(
            handler_call_details.invocation_metadata,
            APP_TOKEN_HEADER,
        )

        def _authenticated_handler(
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            if token is None:
                _abort_auth_denied(context)

            state = self._state_provider()

            # Legacy: Validate both token->run lookup and run->token mapping.
            run_id = state.get_run_id_by_token(token)
            if run_id is not None and state.verify_token(run_id, token):
                return unary_handler(request, context)

            # Validate task token and set task context for downstream handlers
            task_id = state.get_task_id_by_token(token)
            if task_id is not None:
                ctx_token = _current_task_id.set(task_id)
                try:
                    return unary_handler(request, context)
                finally:
                    _current_task_id.reset(ctx_token)

            _abort_auth_denied(context)

        return grpc.unary_unary_rpc_method_handler(
            _authenticated_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )


def get_authenticated_task_id() -> int:
    """Return the task ID authenticated for the current RPC.

    The task ID is available only while handling an RPC authenticated with an AppIo task
    token.
    """
    ret = _current_task_id.get()
    if ret is None:
        raise RuntimeError(
            "No authenticated task ID in the current RPC context. "
            "This function must be called from a task-token-authenticated RPC handler."
        )
    return ret


def create_serverappio_token_auth_server_interceptor(
    state_provider: Callable[[], _TokenState],
) -> AppIoTokenServerInterceptor:
    """Create the default token interceptor for ServerAppIo."""
    return AppIoTokenServerInterceptor(
        state_provider=state_provider,
        method_auth_policy=SERVERAPPIO_METHOD_AUTH_POLICY,
    )


def create_clientappio_token_auth_server_interceptor(
    state_provider: Callable[[], _TokenState],
) -> AppIoTokenServerInterceptor:
    """Create the default token interceptor for ClientAppIo."""
    return AppIoTokenServerInterceptor(
        state_provider=state_provider,
        method_auth_policy=CLIENTAPPIO_METHOD_AUTH_POLICY,
    )
