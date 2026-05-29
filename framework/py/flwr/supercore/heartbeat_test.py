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
"""Tests for heartbeat sender."""


import time
import unittest
from unittest.mock import Mock

import grpc

from flwr.common.constant import HEARTBEAT_CALL_TIMEOUT
from flwr.proto.appio_pb2 import SendTaskHeartbeatResponse  # pylint:disable=E0611

from .heartbeat import HeartbeatSender, make_task_heartbeat_fn_grpc


# pylint: disable=protected-access
class TestHeartbeatSender(unittest.TestCase):
    """Test the HeartbeatSender class."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.mock_heartbeat_fn = Mock(return_value=True)
        self.heartbeat_sender = HeartbeatSender(self.mock_heartbeat_fn)

    def test_start_the_thread(self) -> None:
        """Test that the thread is started and is alive after calling start()."""
        self.heartbeat_sender.start()
        self.assertTrue(self.heartbeat_sender._thread.is_alive())
        self.assertTrue(self.heartbeat_sender.is_running)
        self.heartbeat_sender.stop()  # Clean up

    def test_stop_the_thread(self) -> None:
        """Test that the thread is stopped and not alive after calling stop()."""
        self.heartbeat_sender.start()
        self.assertTrue(self.heartbeat_sender._thread.is_alive())
        self.assertTrue(self.heartbeat_sender.is_running)

        self.heartbeat_sender.stop()
        self.assertFalse(self.heartbeat_sender._thread.is_alive())
        self.assertTrue(self.heartbeat_sender._stop_event.is_set())
        self.assertFalse(self.heartbeat_sender.is_running)

    def test_heartbeat_function_called(self) -> None:
        """Test that the heartbeat function is called."""
        # Execute
        self.heartbeat_sender.start()
        time.sleep(0.1)

        # Assert
        self.mock_heartbeat_fn.assert_called_once()

    def test_stop_interrupts_wait(self) -> None:
        """Test that stop() interrupts any ongoing wait."""
        # Prepare
        self.heartbeat_sender.start()
        time.sleep(0.1)  # Allow some time for heartbeats to be sent
        current = time.time()

        # Execute
        self.heartbeat_sender.stop()

        # Assert
        self.assertLess(time.time() - current, 0.2)
        self.mock_heartbeat_fn.assert_called_once()
        self.assertFalse(self.heartbeat_sender._thread.is_alive())

    def test_heartbeat_fail_and_retry(self) -> None:
        """Test that the heartbeat function is retried on failure."""
        # Prepare
        self.mock_heartbeat_fn.side_effect = [False, False, True]
        self.heartbeat_sender._retry_invoker.wait_function = lambda _: None

        # Execute
        self.heartbeat_sender.start()
        time.sleep(0.1)
        self.heartbeat_sender.stop()

        # Assert
        self.assertEqual(self.mock_heartbeat_fn.call_count, 3)

    def test_thread_is_daemon(self) -> None:
        """Test that the thread is a daemon thread."""
        self.assertTrue(self.heartbeat_sender._thread.daemon)


class TestMakeTaskHeartbeatFnGrpc(unittest.TestCase):
    """Test task heartbeat gRPC function factory."""

    def test_send_task_heartbeat_uses_call_timeout(self) -> None:
        """Test that task heartbeats use a bounded gRPC call deadline."""
        # Prepare
        stub = Mock()
        stub.SendTaskHeartbeat.return_value = SendTaskHeartbeatResponse(success=True)
        heartbeat_fn = make_task_heartbeat_fn_grpc(stub)

        # Execute
        success = heartbeat_fn()

        # Assert
        self.assertTrue(success)
        stub.SendTaskHeartbeat.assert_called_once()
        _, kwargs = stub.SendTaskHeartbeat.call_args
        self.assertEqual(kwargs["timeout"], HEARTBEAT_CALL_TIMEOUT)

    def test_send_task_heartbeat_deadline_exceeded_returns_false(self) -> None:
        """Test that a heartbeat deadline returns false for retry."""

        class DeadlineExceededRpcError(grpc.RpcError):  # type: ignore[misc]
            """gRPC error with DEADLINE_EXCEEDED status."""

            def code(self) -> grpc.StatusCode:
                """Return gRPC status code."""
                return grpc.StatusCode.DEADLINE_EXCEEDED

        # Prepare
        stub = Mock()
        stub.SendTaskHeartbeat.side_effect = DeadlineExceededRpcError()
        heartbeat_fn = make_task_heartbeat_fn_grpc(stub)

        # Execute & assert
        self.assertFalse(heartbeat_fn())
