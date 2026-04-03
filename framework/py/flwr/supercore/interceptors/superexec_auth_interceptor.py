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
from collections.abc import Callable, Mapping
from typing import Any, NoReturn, Protocol, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import now
from flwr.supercore.auth import (
    CLIENTAPPIO_SUPEREXEC_AUTH_POLICY,
    MAX_TIMESTAMP_DIFF_SECONDS,
    MIN_TIMESTAMP_DIFF_SECONDS,
    SERVERAPPIO_SUPEREXEC_AUTH_POLICY,
    SUPEREXEC_AUTH_AUDIENCE_HEADER,
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_RUN_ID_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
    SuperExecMethodPolicy,
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_run_secret,
    extract_single_str_metadata,
    extract_superexec_run_id,
    verify_superexec_signature,
)

from .appio_token_interceptor import AUTHENTICATION_FAILED_MESSAGE


class _NonceState(Protocol):
    """State methods required by SuperExec replay protection."""

    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Atomically reserve a nonce."""


def _abort_auth_denied(context: grpc.ServicerContext) -> NoReturn:
    context.abort(grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE)
    raise RuntimeError("Should not reach this point")


def _unauthenticated_terminator() -> grpc.RpcMethodHandler:
    def _terminate(_request: GrpcMessage, context: grpc.ServicerContext) -> GrpcMessage:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE)
        raise RuntimeError("Should not reach this point")

    return grpc.unary_unary_rpc_method_handler(_terminate)


class SuperExecAuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Attach SuperExec HMAC metadata to outbound unary RPCs."""

    def __init__(
        self,
        *,
        master_secret: bytes,
        audience: str,
        method_auth_policy: Mapping[str, SuperExecMethodPolicy],
    ) -> None:
        self._master_secret = master_secret
        self._audience = audience
        self._method_auth_policy = dict(method_auth_policy)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Add SuperExec signature metadata on outbound unary requests."""
        method = client_call_details.method
        if method not in self._method_auth_policy:
            return continuation(client_call_details, request)

        run_id = extract_superexec_run_id(method, request, self._method_auth_policy)
        timestamp = int(now().timestamp())
        nonce = secrets.token_hex(16)
        body_sha256 = compute_request_body_sha256(request)
        run_secret = derive_run_secret(self._master_secret, run_id)
        signature = compute_superexec_signature(
            run_secret=run_secret,
            method=method,
            audience=self._audience,
            timestamp=timestamp,
            nonce=nonce,
            run_id=run_id,
            body_sha256=body_sha256,
        )

        metadata = list(client_call_details.metadata or [])
        auth_headers = {
            SUPEREXEC_AUTH_AUDIENCE_HEADER,
            SUPEREXEC_AUTH_TIMESTAMP_HEADER,
            SUPEREXEC_AUTH_NONCE_HEADER,
            SUPEREXEC_AUTH_RUN_ID_HEADER,
            SUPEREXEC_AUTH_BODY_SHA256_HEADER,
            SUPEREXEC_AUTH_SIGNATURE_HEADER,
        }
        metadata = [(key, value) for key, value in metadata if key not in auth_headers]
        metadata.extend(
            [
                (SUPEREXEC_AUTH_AUDIENCE_HEADER, self._audience),
                (SUPEREXEC_AUTH_TIMESTAMP_HEADER, str(timestamp)),
                (SUPEREXEC_AUTH_NONCE_HEADER, nonce),
                (SUPEREXEC_AUTH_RUN_ID_HEADER, run_id),
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
        expected_audience: str,
        method_auth_policy: Mapping[str, SuperExecMethodPolicy],
    ) -> None:
        self._state_provider = state_provider
        self._master_secret = master_secret
        self._expected_audience = expected_audience
        self._method_auth_policy = dict(method_auth_policy)

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        """Enforce SuperExec metadata auth for configured unary RPC methods."""
        method = handler_call_details.method
        if method not in self._method_auth_policy:
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
            audience = extract_single_str_metadata(
                metadata, SUPEREXEC_AUTH_AUDIENCE_HEADER
            )
            ts_raw = extract_single_str_metadata(
                metadata, SUPEREXEC_AUTH_TIMESTAMP_HEADER
            )
            nonce = extract_single_str_metadata(metadata, SUPEREXEC_AUTH_NONCE_HEADER)
            run_id_header = extract_single_str_metadata(
                metadata, SUPEREXEC_AUTH_RUN_ID_HEADER
            )
            body_sha256_header = extract_single_str_metadata(
                metadata, SUPEREXEC_AUTH_BODY_SHA256_HEADER
            )
            signature = extract_single_str_metadata(
                metadata, SUPEREXEC_AUTH_SIGNATURE_HEADER
            )
            if None in {
                audience,
                ts_raw,
                nonce,
                run_id_header,
                body_sha256_header,
                signature,
            }:
                _abort_auth_denied(context)

            if audience != self._expected_audience:
                _abort_auth_denied(context)

            try:
                timestamp = int(cast(str, ts_raw))
            except (TypeError, ValueError):
                _abort_auth_denied(context)
            time_diff = now().timestamp() - timestamp
            if not MIN_TIMESTAMP_DIFF_SECONDS < time_diff < MAX_TIMESTAMP_DIFF_SECONDS:
                _abort_auth_denied(context)

            try:
                run_id = extract_superexec_run_id(
                    method, request, self._method_auth_policy
                )
            except ValueError:
                _abort_auth_denied(context)
            if run_id != run_id_header:
                _abort_auth_denied(context)

            body_sha256 = compute_request_body_sha256(request)
            if body_sha256 != body_sha256_header:
                _abort_auth_denied(context)

            run_secret = derive_run_secret(self._master_secret, run_id)
            expected_signature = compute_superexec_signature(
                run_secret=run_secret,
                method=method,
                audience=audience,
                timestamp=timestamp,
                nonce=cast(str, nonce),
                run_id=run_id,
                body_sha256=body_sha256,
            )
            if not verify_superexec_signature(expected_signature, cast(str, signature)):
                _abort_auth_denied(context)

            namespace = f"superexec:{self._expected_audience}:{method}:{run_id}"
            expires_at = float(timestamp + MAX_TIMESTAMP_DIFF_SECONDS)
            if not self._state_provider().reserve_nonce(
                namespace=namespace,
                nonce=cast(str, nonce),
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
    expected_audience: str,
) -> SuperExecAuthServerInterceptor:
    """Create SuperExec auth interceptor for ServerAppIo."""
    return SuperExecAuthServerInterceptor(
        state_provider=state_provider,
        master_secret=master_secret,
        expected_audience=expected_audience,
        method_auth_policy=SERVERAPPIO_SUPEREXEC_AUTH_POLICY,
    )


def create_clientappio_superexec_auth_server_interceptor(
    *,
    state_provider: Callable[[], _NonceState],
    master_secret: bytes,
    expected_audience: str,
) -> SuperExecAuthServerInterceptor:
    """Create SuperExec auth interceptor for ClientAppIo."""
    return SuperExecAuthServerInterceptor(
        state_provider=state_provider,
        master_secret=master_secret,
        expected_audience=expected_audience,
        method_auth_policy=CLIENTAPPIO_SUPEREXEC_AUTH_POLICY,
    )
