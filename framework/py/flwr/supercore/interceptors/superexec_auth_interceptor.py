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
"""SuperExec HMAC metadata interceptors for AppIo services."""


from __future__ import annotations

import secrets
from collections.abc import Callable, Collection
from typing import Any, NoReturn, Protocol, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import now
from flwr.supercore.auth import (
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
    verify_superexec_signature,
)
from flwr.supercore.constant import (
    MAX_TIMESTAMP_DIFF_SECONDS,
    MIN_TIMESTAMP_DIFF_SECONDS,
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.utils import get_metadata_str

from .appio_token_interceptor import AUTHENTICATION_FAILED_MESSAGE

SERVERAPPIO_SUPEREXEC_METHODS: frozenset[str] = frozenset(
    {
        "/flwr.proto.ServerAppIo/ListAppsToLaunch",
        "/flwr.proto.ServerAppIo/RequestToken",
        "/flwr.proto.ServerAppIo/GetRun",
    }
)

CLIENTAPPIO_SUPEREXEC_METHODS: frozenset[str] = frozenset(
    {
        "/flwr.proto.ClientAppIo/ListAppsToLaunch",
        "/flwr.proto.ClientAppIo/RequestToken",
        "/flwr.proto.ClientAppIo/GetRun",
    }
)


class _NonceState(Protocol):
    """State methods required by SuperExec replay protection."""

    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Atomically reserve a nonce."""


def _abort_auth_denied(context: grpc.ServicerContext) -> NoReturn:
    context.abort(grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE)
    raise RuntimeError("Should not reach this point")


def _unauthenticated_terminator() -> grpc.RpcMethodHandler:
    def _terminate(_request: GrpcMessage, context: grpc.ServicerContext) -> GrpcMessage:
        context.abort(
            grpc.StatusCode.UNAUTHENTICATED,
            AUTHENTICATION_FAILED_MESSAGE,
        )
        raise RuntimeError("Should not reach this point")

    return grpc.unary_unary_rpc_method_handler(_terminate)


class SuperExecAuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Attach SuperExec HMAC metadata to outbound unary RPCs."""

    def __init__(
        self,
        *,
        master_secret: bytes,
        protected_methods: Collection[str],
    ) -> None:
        self._auth_secret = derive_auth_secret(master_secret)
        self._protected_methods = set(protected_methods)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Add SuperExec signature metadata on outbound unary requests."""
        method = client_call_details.method
        if method not in self._protected_methods:
            return continuation(client_call_details, request)

        timestamp = int(now().timestamp())
        nonce = secrets.token_hex(16)
        body_sha256 = compute_request_body_sha256(request)
        signature = compute_superexec_signature(
            auth_secret=self._auth_secret,
            method=method,
            timestamp=timestamp,
            nonce=nonce,
            body_sha256=body_sha256,
        )

        metadata = list(client_call_details.metadata or [])
        metadata.extend(
            [
                (SUPEREXEC_AUTH_TIMESTAMP_HEADER, str(timestamp)),
                (SUPEREXEC_AUTH_NONCE_HEADER, nonce),
                (SUPEREXEC_AUTH_BODY_SHA256_HEADER, body_sha256),
                (SUPEREXEC_AUTH_SIGNATURE_HEADER, signature),
            ]
        )

        details = client_call_details._replace(metadata=metadata)
        return continuation(details, request)


class SuperExecAuthServerInterceptor(grpc.ServerInterceptor):  # type: ignore
    """Verify SuperExec HMAC metadata on selected AppIo unary RPCs."""

    def __init__(
        self,
        *,
        state_provider: Callable[[], _NonceState],
        master_secret: bytes,
        protected_methods: Collection[str],
    ) -> None:
        self._state_provider = state_provider
        self._auth_secret = derive_auth_secret(master_secret)
        self._protected_methods = set(protected_methods)

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Enforce SuperExec metadata auth for configured unary RPC methods."""
        method = handler_call_details.method
        if method not in self._protected_methods:
            return continuation(handler_call_details)

        method_handler = continuation(handler_call_details)
        if method_handler is None or method_handler.unary_unary is None:
            return _unauthenticated_terminator()

        unary_handler = cast(
            Callable[[GrpcMessage, grpc.ServicerContext], GrpcMessage],
            method_handler.unary_unary,
        )
        metadata = tuple(handler_call_details.invocation_metadata or ())

        def _authenticated_handler(  # pylint: disable=R0914
            request: GrpcMessage,
            context: grpc.ServicerContext,
        ) -> GrpcMessage:
            ts_raw = get_metadata_str(metadata, SUPEREXEC_AUTH_TIMESTAMP_HEADER)
            nonce = get_metadata_str(metadata, SUPEREXEC_AUTH_NONCE_HEADER)
            body_sha256_header = get_metadata_str(
                metadata, SUPEREXEC_AUTH_BODY_SHA256_HEADER
            )
            signature = get_metadata_str(metadata, SUPEREXEC_AUTH_SIGNATURE_HEADER)
            if not ts_raw or not nonce or not body_sha256_header or not signature:
                _abort_auth_denied(context)

            try:
                timestamp = int(ts_raw)
            except (TypeError, ValueError):
                _abort_auth_denied(context)
            time_diff = now().timestamp() - timestamp
            is_timestamp_in_window = (
                MIN_TIMESTAMP_DIFF_SECONDS < time_diff < MAX_TIMESTAMP_DIFF_SECONDS
            )
            if not is_timestamp_in_window:
                _abort_auth_denied(context)

            body_sha256 = compute_request_body_sha256(request)
            if body_sha256 != body_sha256_header:
                _abort_auth_denied(context)

            expected_signature = compute_superexec_signature(
                auth_secret=self._auth_secret,
                method=method,
                timestamp=timestamp,
                nonce=nonce,
                body_sha256=body_sha256,
            )
            if not verify_superexec_signature(expected_signature, signature):
                _abort_auth_denied(context)

            namespace = f"superexec:{method}"
            expires_at = float(timestamp + MAX_TIMESTAMP_DIFF_SECONDS)
            if not self._state_provider().reserve_nonce(
                namespace=namespace,
                nonce=nonce,
                expires_at=expires_at,
            ):
                _abort_auth_denied(context)

            return unary_handler(request, context)

        return grpc.unary_unary_rpc_method_handler(
            _authenticated_handler,
            request_deserializer=method_handler.request_deserializer,
            response_serializer=method_handler.response_serializer,
        )


def create_serverappio_superexec_auth_server_interceptor(
    *,
    state_provider: Callable[[], _NonceState],
    master_secret: bytes,
) -> SuperExecAuthServerInterceptor:
    """Create SuperExec auth interceptor for ServerAppIo."""
    return SuperExecAuthServerInterceptor(
        state_provider=state_provider,
        master_secret=master_secret,
        protected_methods=SERVERAPPIO_SUPEREXEC_METHODS,
    )


def create_clientappio_superexec_auth_server_interceptor(
    *,
    state_provider: Callable[[], _NonceState],
    master_secret: bytes,
) -> SuperExecAuthServerInterceptor:
    """Create SuperExec auth interceptor for ClientAppIo."""
    return SuperExecAuthServerInterceptor(
        state_provider=state_provider,
        master_secret=master_secret,
        protected_methods=CLIENTAPPIO_SUPEREXEC_METHODS,
    )
