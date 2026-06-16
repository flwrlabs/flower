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
"""Flower Control API auth interceptor tests."""


import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import grpc
from google.protobuf.message import Message as GrpcMessage
from parameterized import parameterized

from flwr.common.dummy_grpc_handlers_test import (
    NoOpUnaryStreamHandler,
    NoOpUnaryUnaryHandler,
    get_noop_unary_stream_handler,
    get_noop_unary_unary_handler,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetLoginDetailsRequest,
    ListRunsRequest,
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
)
from flwr.supercore.auth.typing import AccountInfo

from .control_account_auth_interceptor import (
    ControlAccountAuthInterceptor,
    get_current_account_info,
    shared_account_info,
)


class TestControlAccountAuthInterceptor(unittest.TestCase):
    """Test the ControlAccountAuthInterceptor authentication logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Set a known default for shared_account_info and store the token.
        self.default_account_info = AccountInfo(flwr_aid=None, account_name=None)
        self.token = shared_account_info.set(self.default_account_info)
        self.expected_account_info = AccountInfo(
            flwr_aid="flwr_aid", account_name="account_name"
        )

    def tearDown(self) -> None:
        """Reset shared_account_info to its previous state to prevent state leakage."""
        shared_account_info.reset(self.token)

    @parameterized.expand(
        [
            (GetAuthTokensRequest()),
            (GetLoginDetailsRequest(),),
        ]
    )  # type: ignore
    def test_unary_unary_authentication_successful(self, request: GrpcMessage) -> None:
        """Test unary-unary RPC call successful for a login request.

        Occurs for requests that are GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_authn_plugin = MagicMock()
        dummy_authz_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating invalid tokens
        dummy_authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
        # Set up validate account authorization to return True. The return value is
        # irrelevant because no account authentication is required for requests of type
        # GetLoginDetailsRequest and GetAuthTokensRequest.
        dummy_authz_plugin.authorize.return_value = True
        interceptor = ControlAccountAuthInterceptor(
            authn_plugin=dummy_authn_plugin, authz_plugin=dummy_authz_plugin
        )
        intercepted_handler = interceptor.intercept_service(
            get_noop_unary_unary_handler, handler_call_details
        )

        # Execute
        response = intercepted_handler.unary_unary(dummy_request, dummy_context)

        # Assert response is as expected
        self.assertEqual(response, "dummy_response")
        # Assert `shared_account_info` is not set
        account_info_from_context = get_current_account_info()
        self.assertIsNone(account_info_from_context.flwr_aid)
        self.assertIsNone(account_info_from_context.account_name)

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_unary_rpc_authentication_unsuccessful(self, request: GrpcMessage) -> None:
        """Test unary-unary/stream RPC call not successful when authentication fails.

        Occurs for requests that are not GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_authn_plugin = MagicMock()
        dummy_authz_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating invalid tokens
        dummy_authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
        dummy_authn_plugin.refresh_tokens.return_value = (None, None)
        # Set up `authorize` to return True. The return value is
        # irrelevant because the authentication will fail and the authorization
        # plugin will not be called.
        dummy_authz_plugin.authorize.return_value = True
        interceptor = ControlAccountAuthInterceptor(
            authn_plugin=dummy_authn_plugin, authz_plugin=dummy_authz_plugin
        )
        continuation: (
            Callable[[Any], NoOpUnaryUnaryHandler]
            | Callable[[Any], NoOpUnaryStreamHandler]
        ) = get_noop_unary_unary_handler
        # Set up unary-stream case for StreamLogsRequest
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler
        intercepted_handler = interceptor.intercept_service(
            continuation, handler_call_details
        )

        # Execute & Assert: interceptor should abort with UNAUTHENTICATED
        if isinstance(request, StreamLogsRequest):
            with self.assertRaises(grpc.RpcError):
                _ = intercepted_handler.unary_stream(dummy_request, dummy_context)
        else:
            with self.assertRaises(grpc.RpcError):
                intercepted_handler.unary_unary(dummy_request, dummy_context)

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_unary_validate_tokens_successful(self, request: GrpcMessage) -> None:
        """Test unary-unary/stream RPC call is successful when token is valid.

        Occurs for requests that are not GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_authn_plugin = MagicMock()
        dummy_authz_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating valid tokens
        dummy_authn_plugin.validate_tokens_in_metadata.return_value = (
            True,
            self.expected_account_info,
        )
        # Set up `authorize` to return True. The return value must
        # be True because the authorization plugin is expected to be called after a
        # successful token validation.
        dummy_authz_plugin.authorize.return_value = True
        interceptor = ControlAccountAuthInterceptor(
            authn_plugin=dummy_authn_plugin, authz_plugin=dummy_authz_plugin
        )
        continuation: (
            Callable[[Any], NoOpUnaryUnaryHandler]
            | Callable[[Any], NoOpUnaryStreamHandler]
        ) = get_noop_unary_unary_handler
        # Set up unary-stream case for StreamLogsRequest
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler
        intercepted_handler = interceptor.intercept_service(
            continuation, handler_call_details
        )

        # Execute & Assert
        if isinstance(request, StreamLogsRequest):
            response = intercepted_handler.unary_stream(dummy_request, dummy_context)
            # Assert response is as expected
            self.assertEqual(list(response), ["stream response 1", "stream response 2"])
        else:
            response = intercepted_handler.unary_unary(dummy_request, dummy_context)
            # Assert response is as expected
            self.assertEqual(response, "dummy_response")

        # Assert `shared_account_info` is set
        account_info_from_context = get_current_account_info()
        self.assertEqual(
            account_info_from_context.flwr_aid, self.expected_account_info.flwr_aid
        )
        self.assertEqual(
            account_info_from_context.account_name,
            self.expected_account_info.account_name,
        )

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_unary_refresh_tokens_successful(self, request: GrpcMessage) -> None:
        """Test unary-unary/stream RPC call is successful when tokens are refreshed.

        Occurs for requests that are not GetLoginDetailsRequest or GetAuthTokensRequest.
        """
        # Prepare
        dummy_request = request
        dummy_context = MagicMock()
        dummy_authn_plugin = MagicMock()
        dummy_authz_plugin = MagicMock()
        handler_call_details = MagicMock()

        # Set up validate_tokens_in_metadata to return a tuple indicating invalid tokens
        dummy_authn_plugin.validate_tokens_in_metadata.return_value = (False, None)
        # Set up refresh tokens
        expected_refresh_tokens_value = [("new-token", "value")]
        dummy_authn_plugin.refresh_tokens.return_value = (
            expected_refresh_tokens_value,
            self.default_account_info,
        )
        # Set up `authorize` to return True. The return value must be True
        # because the authorization plugin is expected to be called after a successful
        # token refresh.
        dummy_authz_plugin.authorize.return_value = True

        interceptor = ControlAccountAuthInterceptor(
            authn_plugin=dummy_authn_plugin, authz_plugin=dummy_authz_plugin
        )
        continuation: (
            Callable[[Any], NoOpUnaryUnaryHandler]
            | Callable[[Any], NoOpUnaryStreamHandler]
        ) = get_noop_unary_unary_handler
        # Set up unary-stream case for StreamLogsRequest
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler
        intercepted_handler = interceptor.intercept_service(
            continuation, handler_call_details
        )

        # Execute & Assert
        if isinstance(request, StreamLogsRequest):
            response_iterator = intercepted_handler.unary_stream(
                dummy_request, dummy_context
            )
            responses = list(response_iterator)
            # Assert responses are as expected
            self.assertEqual(responses, ["stream response 1", "stream response 2"])
        else:
            response = intercepted_handler.unary_unary(dummy_request, dummy_context)
            # Assert response is as expected and initial metadata was sent
            self.assertEqual(response, "dummy_response")
        # Assert refresh tokens were sent in initial metadata
        dummy_context.send_initial_metadata.assert_called_once_with(
            expected_refresh_tokens_value
        )


class TestExecUserAuthInterceptorAuthorization(unittest.TestCase):
    """Test the ExecUserAuthInterceptor authorization logic."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Reset the shared AccountInfo before each test
        self.default_token = shared_account_info.set(
            AccountInfo(flwr_aid=None, account_name=None)
        )
        self.expected_account_info = AccountInfo(
            flwr_aid="flwr_aid", account_name="account_name"
        )

        # A dummy authorization plugin
        self.authz_plugin = MagicMock()

        # A dummy authentication plugin that always validates tokens
        self.authn_plugin = MagicMock()
        self.authn_plugin.validate_tokens_in_metadata.return_value = (
            True,
            self.expected_account_info,
        )

    def tearDown(self) -> None:
        """Reset shared_account_info."""
        shared_account_info.reset(self.default_token)

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_authorization_successful(self, request: GrpcMessage) -> None:
        """Test RPC calls successful when authorization is approved.

        When AuthZ plugin approves, the RPC calls should succeed.
        """
        dummy_context = MagicMock()
        handler_call_details = MagicMock()

        # Authorization approves
        self.authz_plugin.authorize.return_value = True

        interceptor = ControlAccountAuthInterceptor(
            authn_plugin=self.authn_plugin, authz_plugin=self.authz_plugin
        )

        # Pick correct continuation for unary vs stream
        continuation: (
            Callable[[Any], NoOpUnaryUnaryHandler]
            | Callable[[Any], NoOpUnaryStreamHandler]
        ) = get_noop_unary_unary_handler
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler

        intercepted = interceptor.intercept_service(continuation, handler_call_details)

        # Execute & Assert
        if isinstance(request, StreamLogsRequest):
            result = list(intercepted.unary_stream(request, dummy_context))
            self.assertEqual(result, ["stream response 1", "stream response 2"])
        else:
            result = intercepted.unary_unary(request, dummy_context)
            self.assertEqual(result, "dummy_response")
        # Authz plugin should have been called once
        self.authz_plugin.authorize.assert_called_once_with(self.expected_account_info)

    @parameterized.expand(
        [
            (ListRunsRequest()),
            (StartRunRequest()),
            (StopRunRequest()),
            (StreamLogsRequest()),
        ]
    )  # type: ignore
    def test_authorization_failure(self, request: GrpcMessage) -> None:
        """Test RPC calls not successful when authorization fails.

        When AuthZ plugin denies, the calls should be aborted with PERMISSION_DENIED.
        """
        dummy_context = MagicMock()
        handler_call_details = MagicMock()

        # Authorization denies
        self.authz_plugin.authorize.return_value = False

        interceptor = ControlAccountAuthInterceptor(
            authn_plugin=self.authn_plugin, authz_plugin=self.authz_plugin
        )

        continuation: (
            Callable[[Any], NoOpUnaryUnaryHandler]
            | Callable[[Any], NoOpUnaryStreamHandler]
        ) = get_noop_unary_unary_handler
        if isinstance(request, StreamLogsRequest):
            continuation = get_noop_unary_stream_handler

        intercepted = interceptor.intercept_service(continuation, handler_call_details)

        # Execute & Assert
        if isinstance(request, StreamLogsRequest):
            with self.assertRaises(grpc.RpcError):
                _ = intercepted.unary_stream(request, dummy_context)
        else:
            with self.assertRaises(grpc.RpcError):
                _ = intercepted.unary_unary(request, dummy_context)

        # Ensure abort was called with PERMISSION_DENIED
        dummy_context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED,
            "❗️ Account not authorized. Please contact the SuperLink administrator.",
        )
