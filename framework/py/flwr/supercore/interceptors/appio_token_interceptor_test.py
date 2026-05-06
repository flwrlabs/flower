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
"""Tests for short-term AppIo token interceptors."""


import inspect
from collections import namedtuple
from typing import cast
from unittest import TestCase
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ListAppsToLaunchRequest,
    PushAppMessagesRequest,
    PushAppOutputsRequest,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoServicer
from flwr.proto.message_pb2 import PushObjectRequest  # pylint: disable=E0611
from flwr.proto.serverappio_pb2 import GetNodesRequest  # pylint: disable=E0611
from flwr.proto.serverappio_pb2_grpc import ServerAppIoServicer
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.auth import (
    CLIENTAPPIO_METHOD_AUTH_POLICY,
    SERVERAPPIO_METHOD_AUTH_POLICY,
)
from flwr.supercore.interceptors import (
    APP_TOKEN_HEADER,
    AUTHENTICATION_FAILED_MESSAGE,
    AppIoTokenClientInterceptor,
    AppIoTokenServerInterceptor,
    create_clientappio_token_auth_server_interceptor,
    create_serverappio_token_auth_server_interceptor,
    get_authenticated_task,
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


class _TokenState:
    def __init__(self, token_to_run_id: dict[str, int]) -> None:
        self._token_to_run_id = token_to_run_id

    def get_run_id_by_token(self, token: str) -> int | None:
        """Return the run id for a token, if present."""
        return self._token_to_run_id.get(token)

    def verify_token(self, run_id: int, token: str) -> bool:
        """Return whether the token is bound to the given run id."""
        return self._token_to_run_id.get(token) == run_id

    def get_task_by_token(self, token: str) -> Task | None:  # pylint: disable=R1711
        """Return the task for a task token, if present."""
        _ = token
        return None  # make mypy happy


def _make_unary_handler() -> grpc.RpcMethodHandler:
    def _handler(_request: GrpcMessage, _context: grpc.ServicerContext) -> str:
        return "ok"

    return grpc.unary_unary_rpc_method_handler(_handler)


def _make_non_unary_handler() -> grpc.RpcMethodHandler:
    def _handler(
        _request: GrpcMessage, _context: grpc.ServicerContext
    ) -> list[GrpcMessage]:
        return []

    return grpc.unary_stream_rpc_method_handler(_handler)


class TestAppIoTokenClientInterceptor(TestCase):
    """Unit tests for AppIoTokenClientInterceptor."""

    def test_attach_app_token_header(self) -> None:
        """The interceptor should attach App token metadata."""
        interceptor = AppIoTokenClientInterceptor(token="new-token")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/GetNodes",
            timeout=None,
            metadata=(("x-test", "value"),),
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
            request=GetNodesRequest(run_id=1),
        )

        self.assertEqual(response, "ok")
        metadata = captured["metadata"]
        self.assertIn(("x-test", "value"), metadata)
        self.assertEqual(
            [item for item in metadata if item[0] == APP_TOKEN_HEADER],
            [(APP_TOKEN_HEADER, "new-token")],
        )

    def test_raise_if_app_token_header_already_present(self) -> None:
        """The interceptor should reject duplicate App token metadata."""
        interceptor = AppIoTokenClientInterceptor(token="new-token")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/GetNodes",
            timeout=None,
            metadata=(("x-test", "value"), (APP_TOKEN_HEADER, "old-token")),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )

        with self.assertRaises(RuntimeError):
            interceptor.intercept_unary_unary(
                continuation=Mock(),
                client_call_details=details,
                request=GetNodesRequest(run_id=1),
            )


class TestAppIoTokenServerInterceptor(TestCase):
    """Unit tests for AppIoTokenServerInterceptor."""

    def _new_interceptor(
        self, token_to_run_id: dict[str, int]
    ) -> AppIoTokenServerInterceptor:
        state = _TokenState(token_to_run_id)
        return create_serverappio_token_auth_server_interceptor(lambda: state)

    @staticmethod
    def _find_serverappio_method(*, requires_token: bool) -> str | None:
        methods = [
            method
            for method, policy in SERVERAPPIO_METHOD_AUTH_POLICY.items()
            if policy.requires_token is requires_token
        ]
        return sorted(methods)[0] if methods else None

    def test_no_auth_method_allows_call_without_token(self) -> None:
        """No-auth methods should pass through without metadata token."""
        interceptor = self._new_interceptor(token_to_run_id={})
        method = self._find_serverappio_method(requires_token=False)
        if method is None:
            self.skipTest("No no-auth ServerAppIo method found in policy table.")

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=(),
            ),
        )

        response = cast(str, intercepted.unary_unary(ListAppsToLaunchRequest(), Mock()))
        self.assertEqual(response, "ok")

    def test_missing_token_denied_for_protected_method(self) -> None:
        """Protected methods should deny requests without token input."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=(),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_invalid_token_denied_for_protected_method(self) -> None:
        """Protected methods should deny requests with invalid tokens."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "invalid"),),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_valid_token_passes_for_protected_method(self) -> None:
        """Protected methods should pass with a valid token."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )

        # Keep success-path test data aligned to avoid implying
        # cross-run use is expected.
        response = cast(str, intercepted.unary_unary(GetNodesRequest(run_id=7), Mock()))
        self.assertEqual(response, "ok")
        # Run-id mismatch deny coverage belongs to the
        # follow-up PR that enforces run binding.

    def test_valid_task_token_passes_and_sets_task_id(self) -> None:
        """Protected methods should pass with a valid task token."""
        state = Mock()
        state.get_run_id_by_token.return_value = None
        state.get_task_by_token.return_value = Mock(task_id=123)
        interceptor = create_serverappio_token_auth_server_interceptor(lambda: state)
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        captured_task = None

        def _handler(_request: GrpcMessage, _context: grpc.ServicerContext) -> str:
            nonlocal captured_task
            captured_task = get_authenticated_task()
            return "ok"

        intercepted = interceptor.intercept_service(
            lambda _: grpc.unary_unary_rpc_method_handler(_handler),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "task-token"),),
            ),
        )

        response = intercepted.unary_unary(GetNodesRequest(run_id=7), Mock())
        self.assertEqual(response, "ok")
        self.assertIsNotNone(captured_task)
        self.assertEqual(cast(Mock, captured_task).task_id, 123)
        state.get_run_id_by_token.assert_called_once_with("task-token")
        state.get_task_by_token.assert_called_once_with("task-token")

    def test_metadata_token_used_even_when_request_has_token(self) -> None:
        """Metadata token should be authoritative when both sources exist."""
        interceptor = self._new_interceptor(token_to_run_id={"metadata-token": 5})

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PushAppOutputs",
                invocation_metadata=((APP_TOKEN_HEADER, "metadata-token"),),
            ),
        )

        response = cast(
            str,
            intercepted.unary_unary(
                PushAppOutputsRequest(token="request-token", run_id=5), Mock()
            ),
        )
        self.assertEqual(response, "ok")

    def test_metadata_token_used_for_protected_method(self) -> None:
        """Metadata token should be used for protected methods."""
        interceptor = self._new_interceptor(token_to_run_id={"metadata-token": 5})

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PushMessages",
                invocation_metadata=((APP_TOKEN_HEADER, "metadata-token"),),
            ),
        )

        response = cast(
            str,
            intercepted.unary_unary(PushAppMessagesRequest(run_id=5), Mock()),
        )
        self.assertEqual(response, "ok")

    def test_request_token_without_metadata_is_denied(self) -> None:
        """Request-body token alone should not satisfy auth."""
        interceptor = self._new_interceptor(token_to_run_id={"request-token": 5})
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PushAppOutputs",
                invocation_metadata=(),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(
                PushAppOutputsRequest(token="request-token", run_id=5), context
            )
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_unknown_method_fails_closed(self) -> None:
        """Unknown methods should fail closed with UNAUTHENTICATED."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        continuation = Mock(return_value=_make_unary_handler())
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            continuation,
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/UnknownMethod",
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        continuation.assert_not_called()
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )

    def test_non_unary_handler_fails_closed_for_protected_method(self) -> None:
        """Protected methods with non-unary handlers should fail closed."""
        interceptor = self._new_interceptor(token_to_run_id={"valid": 7})
        method = self._find_serverappio_method(requires_token=True)
        if method is None:
            self.skipTest("No token-required ServerAppIo method found in policy table.")
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        intercepted = interceptor.intercept_service(
            lambda _: _make_non_unary_handler(),
            _HandlerCallDetails(
                method,
                invocation_metadata=((APP_TOKEN_HEADER, "valid"),),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=7), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.UNAUTHENTICATED, AUTHENTICATION_FAILED_MESSAGE
        )


class TestMethodPolicyMaps(TestCase):
    """Validate method auth policy map coverage and values."""

    NO_AUTH_BOOTSTRAP_METHODS = {
        "ListAppsToLaunch",
        "RequestToken",
        "PullPendingTasks",
        "ClaimTask",
        "GetRun",
    }

    @staticmethod
    def _serverappio_rpc_methods() -> set[str]:
        return {
            f"/flwr.proto.ServerAppIo/{name}"
            for name, ref in inspect.getmembers(ServerAppIoServicer)
            if inspect.isfunction(ref) and not name.startswith("_")
        }

    @staticmethod
    def _clientappio_rpc_methods() -> set[str]:
        return {
            f"/flwr.proto.ClientAppIo/{name}"
            for name, ref in inspect.getmembers(ClientAppIoServicer)
            if inspect.isfunction(ref) and not name.startswith("_")
        }

    def test_serverappio_policy_has_full_coverage(self) -> None:
        """ServerAppIo policy map should cover all RPC methods exactly."""
        expected_methods = self._serverappio_rpc_methods()
        self.assertEqual(set(SERVERAPPIO_METHOD_AUTH_POLICY), expected_methods)

    def test_only_expected_no_auth_methods_exist(self) -> None:
        """Only bootstrap methods should be marked no-auth in the policy table."""
        no_auth_methods = {
            method.rsplit("/", maxsplit=1)[-1]
            for method, policy in SERVERAPPIO_METHOD_AUTH_POLICY.items()
            if not policy.requires_token
        }
        self.assertEqual(no_auth_methods, self.NO_AUTH_BOOTSTRAP_METHODS)

    def test_clientappio_policy_has_full_coverage(self) -> None:
        """ClientAppIo policy map should cover all RPC methods exactly."""
        expected_methods = self._clientappio_rpc_methods()
        self.assertEqual(set(CLIENTAPPIO_METHOD_AUTH_POLICY), expected_methods)

    def test_clientappio_only_expected_no_auth_methods_exist(self) -> None:
        """ClientAppIo should only mark bootstrap methods as no-auth."""
        no_auth_methods = {
            method.rsplit("/", maxsplit=1)[-1]
            for method, policy in CLIENTAPPIO_METHOD_AUTH_POLICY.items()
            if not policy.requires_token
        }
        self.assertEqual(no_auth_methods, self.NO_AUTH_BOOTSTRAP_METHODS)


class TestFactoryFunctions(TestCase):
    """Validate interceptor factory behavior."""

    def test_serverappio_factory_uses_server_policy(self) -> None:
        """ServerAppIo factory should enforce ServerAppIo policy semantics."""
        state = _TokenState({"valid-token": 1})
        interceptor = create_serverappio_token_auth_server_interceptor(lambda: state)

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/GetNodes",
                invocation_metadata=((APP_TOKEN_HEADER, "valid-token"),),
            ),
        )

        response = cast(str, intercepted.unary_unary(GetNodesRequest(run_id=1), Mock()))
        self.assertEqual(response, "ok")

    def test_clientappio_factory_uses_client_policy(self) -> None:
        """ClientAppIo factory should enforce ClientAppIo policy semantics."""
        state = _TokenState({"valid-token": 1})
        interceptor = create_clientappio_token_auth_server_interceptor(lambda: state)

        intercepted = interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ClientAppIo/PushObject",
                invocation_metadata=((APP_TOKEN_HEADER, "valid-token"),),
            ),
        )

        response = cast(
            str,
            intercepted.unary_unary(
                PushObjectRequest(object_id="obj", object_content=b"x"),
                Mock(),
            ),
        )
        self.assertEqual(response, "ok")
