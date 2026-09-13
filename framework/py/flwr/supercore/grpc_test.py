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
"""Tests for gRPC."""


import unittest
from unittest.mock import MagicMock, patch

import grpc

from flwr.supercore.grpc import GRPC_MAX_MESSAGE_LENGTH, create_channel

from .grpc import valid_certificates


def test_valid_certificates_when_correct() -> None:
    """Test is validation function works correctly when passed valid list."""
    # Prepare
    certificates = (b"a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)

    # Assert
    assert is_valid


def test_valid_certificates_when_wrong() -> None:
    """Test is validation function works correctly when passed invalid list."""
    # Prepare
    certificates = ("not_a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)  # type: ignore

    # Assert
    assert not is_valid


class TestCreateChannel(unittest.TestCase):  # pylint: disable=R0902
    """Test the `create_channel` function."""

    def setUp(self) -> None:
        """Initialize."""
        self.insecure_patcher = patch.object(grpc, "insecure_channel")
        self.mock_insecure_channel = self.insecure_patcher.start()

        self.secure_patcher = patch.object(grpc, "secure_channel")
        self.mock_secure_channel = self.secure_patcher.start()

        self.ssl_patcher = patch.object(grpc, "ssl_channel_credentials")
        self.mock_ssl_channel_credentials = self.ssl_patcher.start()

        self.intercept_patcher = patch.object(grpc, "intercept_channel")
        self.mock_intercept_channel = self.intercept_patcher.start()

    def tearDown(self) -> None:
        """Cleanup."""
        self.insecure_patcher.stop()
        self.secure_patcher.stop()
        self.ssl_patcher.stop()
        self.intercept_patcher.stop()

    def test_insecure_channel_creation(self) -> None:
        """Test insecure channel created successfully."""
        server_address = "localhost:50051"

        # Setup - Configure the insecure channel mock to return a dummy channel
        self.mock_insecure_channel.return_value = "fake_insecure_channel"

        # Execute - Call create_channel in insecure mode
        channel = create_channel(server_address, insecure=True)

        # Assert that insecure_channel was called with the expected options
        self.mock_insecure_channel.assert_called_once_with(
            server_address,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        # Assert that secure-related functions were not called
        self.mock_ssl_channel_credentials.assert_not_called()
        self.mock_secure_channel.assert_not_called()
        self.mock_intercept_channel.assert_not_called()

        # Assert the returned channel is our dummy insecure channel
        self.assertEqual(channel, "fake_insecure_channel")

    def test_secure_channel_creation(self) -> None:
        """Test secure channel created successfully."""
        # Setup
        server_address = "localhost:50051"
        dummy_root_cert = b"dummy_root_cert"

        # Configure secure mocks with return values
        self.mock_ssl_channel_credentials.return_value = "dummy_credentials"
        self.mock_secure_channel.return_value = "fake_secure_channel"

        # Execute - Call create_channel in secure mode
        channel = create_channel(
            server_address,
            insecure=False,
            root_certificates=dummy_root_cert,
        )

        # Assert
        # Verify that ssl_channel_credentials was called with the dummy certificate
        self.mock_ssl_channel_credentials.assert_called_once_with(dummy_root_cert)
        # Verify that secure_channel was called with the appropriate arguments
        self.mock_secure_channel.assert_called_once_with(
            server_address,
            "dummy_credentials",
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        self.mock_intercept_channel.assert_not_called()

        # Assert that the returned channel is what secure_channel returned
        self.assertEqual(channel, "fake_secure_channel")

    def test_secure_channel_creation_with_interceptors(self) -> None:
        """Test secure channel created successfully with interceptors."""
        # Setup
        server_address = "localhost:50051"
        dummy_root_cert = b"dummy_root_cert"
        dummy_interceptor = MagicMock(name="dummy_interceptor")

        # Set return values for secure creation
        self.mock_ssl_channel_credentials.return_value = "dummy_credentials"
        self.mock_secure_channel.return_value = "fake_secure_channel"
        self.mock_intercept_channel.return_value = "intercepted_channel"

        # Execute - Call create_channel with an interceptor
        channel = create_channel(
            server_address,
            insecure=False,
            root_certificates=dummy_root_cert,
            interceptors=[dummy_interceptor],
        )

        # Assert - Verify that ssl_channel_credentials was called correctly
        self.mock_ssl_channel_credentials.assert_called_once_with(dummy_root_cert)
        # Verify that secure_channel was invoked with the expected options
        self.mock_secure_channel.assert_called_once_with(
            server_address,
            "dummy_credentials",
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        # Verify that intercept_channel wrapped the channel
        self.mock_intercept_channel.assert_called_once_with(
            "fake_secure_channel", dummy_interceptor
        )
        # Assert that the returned channel is the one returned by intercept_channel
        self.assertEqual(channel, "intercepted_channel")

    def test_secure_channel_credentials_failure(self) -> None:
        """Test misconfigured secure channel raises ValueError."""
        # Setup
        server_address = "localhost:50051"
        dummy_root_cert = b"dummy_root_cert"

        # Configure ssl_channel_credentials to raise an exception
        self.mock_ssl_channel_credentials.side_effect = Exception("SSL creation failed")

        # Execute & Assert
        with self.assertRaises(ValueError):
            create_channel(
                server_address, insecure=False, root_certificates=dummy_root_cert
            )
