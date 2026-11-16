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
"""Tests for FlowerServiceServicer."""


import unittest
import uuid
from unittest.mock import MagicMock, call

from flwr.compat.server.grpc_bidi.flower_service_servicer import (
    FlowerServiceServicer,
    register_client_proxy,
)
from flwr.compat.server.grpc_bidi.grpc_bridge import InsWrapper, ResWrapper
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    ServerMessage,
)

CLIENT_MESSAGE = ClientMessage()
SERVER_MESSAGE = ServerMessage()

CID: str = uuid.uuid4().hex


class FlowerServiceServicerTestCase(unittest.TestCase):
    """Test suite for class FlowerServiceServicer and helper functions."""

    # pylint: disable=too-many-instance-attributes

    def setUp(self) -> None:
        """Create mocks for tests."""
        # Mock for the gRPC context argument
        self.context_mock = MagicMock()

        # Define client_messages to be processed by FlowerServiceServicer instance
        self.client_messages = [CLIENT_MESSAGE for _ in range(5)]
        self.res_wrappers = [
            ResWrapper(client_message=msg) for msg in self.client_messages
        ]
        self.client_messages_iterator = iter(self.client_messages)

        # Define corresponding responses from bridge
        self.ins_wrappers = [
            InsWrapper(server_message=SERVER_MESSAGE, timeout=None)
            for _ in self.client_messages
        ]
        self.ins_wrapper_iterator = iter(self.ins_wrappers)

        # Mock for GrpcBridge
        self.grpc_bridge_mock = MagicMock()
        self.grpc_bridge_mock.ins_wrapper_iterator.return_value = (
            self.ins_wrapper_iterator
        )

        self.grpc_bridge_factory_mock = MagicMock()
        self.grpc_bridge_factory_mock.return_value = self.grpc_bridge_mock

        # Create a GrpcClientProxy mock which we will use to test if correct
        # methods where called and client_messages are getting passed to it
        self.grpc_client_proxy_mock = MagicMock()
        self.grpc_client_proxy_mock.cid = CID

        self.client_proxy_factory_mock = MagicMock()
        self.client_proxy_factory_mock.return_value = self.grpc_client_proxy_mock

        self.client_manager_mock = MagicMock()

    def test_register_client_proxy(self) -> None:
        """Test register_client_proxy function."""
        # Prepare
        self.client_manager_mock.register.return_value = True

        # Execute
        register_client_proxy(
            client_manager=self.client_manager_mock,
            client_proxy=self.grpc_client_proxy_mock,
            context=self.context_mock,
        )

        # Assert
        self.context_mock.add_callback.assert_called_once()

        # call_args contains the arguments each wrapped in a unittest.mock.call object
        # which holds the args in wrapped a tuple. We therefore we need to take [0][0]
        rpc_termination_callback = self.context_mock.add_callback.call_args[0][0]
        rpc_termination_callback()

        self.client_manager_mock.register.assert_called_once_with(
            self.grpc_client_proxy_mock
        )
        self.client_manager_mock.unregister.assert_called_once_with(
            self.grpc_client_proxy_mock
        )

    def test_join(self) -> None:
        """Test Join method of FlowerServiceServicer."""
        # Prepare

        # Create a instance of FlowerServiceServicer
        servicer = FlowerServiceServicer(
            client_manager=self.client_manager_mock,
            grpc_bridge_factory=self.grpc_bridge_factory_mock,
            grpc_client_proxy_factory=self.client_proxy_factory_mock,
        )

        # Execute
        server_message_iterator = servicer.Join(
            self.client_messages_iterator, self.context_mock
        )

        # Assert
        num_server_messages = 0

        for _ in server_message_iterator:
            num_server_messages += 1

        assert len(self.client_messages) == num_server_messages
        assert self.grpc_client_proxy_mock.cid == CID

        # Check if the client was registered with the client_manager
        self.client_manager_mock.register.assert_called_once_with(
            self.grpc_client_proxy_mock
        )

        self.grpc_bridge_mock.set_res_wrapper.assert_has_calls(
            [call(res_wrapper=message) for message in self.res_wrappers]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
