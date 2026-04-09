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
"""Tests for SuperExec HMAC metadata interceptors."""


from collections import namedtuple
from typing import cast
from unittest import TestCase
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import now
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    RequestTokenRequest,
)
from flwr.supercore.auth import (
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
)
from flwr.supercore.constant import (
    MAX_TIMESTAMP_DIFF_SECONDS,
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.interceptors import (
    AUTHENTICATION_FAILED_MESSAGE,
    SuperExecAuthClientInterceptor,
    create_serverappio_superexec_auth_server_interceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    SERVERAPPIO_SUPEREXEC_METHODS as _SERVERAPPIO_SUPEREXEC_METHODS,
)

_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ["method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"],
)


class _HandlerCallDetails:
    def __init__(
        self,
        method: str,
        invocation_metadata: tuple[tuple[str, str | bytes], ...],
    ) -> None:
        self.method = method
        self.invocation_metadata = invocation_metadata


class _NonceState:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], float] = {}

    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Reserve nonce if it has not been seen before."""
        key = (namespace, nonce)
        if key in self._records:
            return False
        self._records[key] = expires_at
        return True


def _make_unary_handler() -> grpc.RpcMethodHandler:
    def _handler(_request: GrpcMessage, _context: grpc.ServicerContext) -> str:
        return "ok"

    return grpc.unary_unary_rpc_method_handler(_handler)


class TestSuperExecAuthClientInterceptor(TestCase):
    """Unit tests for SuperExecAuthClientInterceptor."""

    def test_attach_signed_metadata_for_superexec_method(self) -> None:
        """Protected SuperExec methods should receive signed metadata."""
        interceptor = SuperExecAuthClientInterceptor(
            master_secret=b"secret",
            protected_methods=_SERVERAPPIO_SUPEREXEC_METHODS,
        )
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/RequestToken",
            timeout=None,
            metadata=(),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> str:
            captured["metadata"] = list(client_call_details.metadata or [])
            return "ok"

        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=details,
            request=RequestTokenRequest(run_id=7),
        )
        self.assertEqual(response, "ok")
        md = dict(captured["metadata"])
        for header in (
            SUPEREXEC_AUTH_TIMESTAMP_HEADER,
            SUPEREXEC_AUTH_NONCE_HEADER,
            SUPEREXEC_AUTH_BODY_SHA256_HEADER,
            SUPEREXEC_AUTH_SIGNATURE_HEADER,
        ):
            self.assertIn(header, md)


class TestSuperExecAuthServerInterceptor(TestCase):
    """Unit tests for SuperExecAuthServerInterceptor."""

    def setUp(self) -> None:
        """Create the default server interceptor under test."""
        self._secret = b"secret"
        self._state = _NonceState()
        self._interceptor = create_serverappio_superexec_auth_server_interceptor(
            state_provider=lambda: self._state,
            master_secret=self._secret,
        )

    def _signed_metadata(  # pylint: disable=R0913
        self,
        *,
        method: str,
        request: GrpcMessage,
        body_sha256: str | None = None,
        signature_override: str | None = None,
        nonce: str = "nonce-1",
        timestamp: int | None = None,
    ) -> tuple[tuple[str, str], ...]:
        ts = int(now().timestamp()) if timestamp is None else timestamp
        body = (
            compute_request_body_sha256(request) if body_sha256 is None else body_sha256
        )
        auth_secret = derive_auth_secret(self._secret)
        signature = compute_superexec_signature(
            auth_secret=auth_secret,
            method=method,
            timestamp=ts,
            nonce=nonce,
            body_sha256=body,
        )
        return (
            (SUPEREXEC_AUTH_TIMESTAMP_HEADER, str(ts)),
            (SUPEREXEC_AUTH_NONCE_HEADER, nonce),
            (SUPEREXEC_AUTH_BODY_SHA256_HEADER, body),
            (
                SUPEREXEC_AUTH_SIGNATURE_HEADER,
                signature if signature_override is None else signature_override,
            ),
        )

    def test_valid_signed_request_is_allowed(self) -> None:
        """Valid metadata should allow a protected SuperExec RPC."""
        method = "/flwr.proto.ServerAppIo/RequestToken"
        request = RequestTokenRequest(run_id=9)
        intercepted = self._interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method=method,
                invocation_metadata=self._signed_metadata(
                    method=method,
                    request=request,
                ),
            ),
        )
        response = cast(str, intercepted.unary_unary(request, Mock()))
        self.assertEqual(response, "ok")

    def test_missing_metadata_is_denied(self) -> None:
        """Missing SuperExec metadata should be denied."""
        method = "/flwr.proto.ServerAppIo/ListAppsToLaunch"
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = self._interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(method=method, invocation_metadata=()),
        )
        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(ListAppsToLaunchRequest(), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_replayed_nonce_is_denied(self) -> None:
        """Reusing a nonce within the active window should be denied."""
        method = "/flwr.proto.ServerAppIo/ListAppsToLaunch"
        request = ListAppsToLaunchRequest()
        metadata = self._signed_metadata(
            method=method,
            request=request,
            nonce="nonce-replay",
        )
        intercepted = self._interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(method=method, invocation_metadata=metadata),
        )
        _ = cast(str, intercepted.unary_unary(request, Mock()))

        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(request, context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_invalid_body_hash_is_denied(self) -> None:
        """Body hash mismatch should be denied."""
        method = "/flwr.proto.ServerAppIo/RequestToken"
        request = RequestTokenRequest(run_id=9)
        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        intercepted = self._interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method=method,
                invocation_metadata=self._signed_metadata(
                    method=method,
                    request=request,
                    body_sha256="0" * 64,
                ),
            ),
        )
        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(request, context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_stale_timestamp_is_denied(self) -> None:
        """Requests outside timestamp tolerance should be denied."""
        method = "/flwr.proto.ServerAppIo/ListAppsToLaunch"
        request = ListAppsToLaunchRequest()
        old_ts = int(now().timestamp() - (MAX_TIMESTAMP_DIFF_SECONDS + 5))
        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        intercepted = self._interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method=method,
                invocation_metadata=self._signed_metadata(
                    method=method,
                    request=request,
                    timestamp=old_ts,
                ),
            ),
        )
        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(request, context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_non_superexec_methods_passthrough(self) -> None:
        """Methods outside SuperExec policy should pass through unchanged."""
        method = "/flwr.proto.ServerAppIo/GetNodes"
        intercepted = self._interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(method=method, invocation_metadata=()),
        )
        response = cast(
            str, intercepted.unary_unary(RequestTokenRequest(run_id=1), Mock())
        )
        self.assertEqual(response, "ok")
