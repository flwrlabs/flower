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


import threading
import time
import unittest
from unittest.mock import Mock, patch

from flwr.common.constant import (
    HEARTBEAT_BASE_MULTIPLIER,
    HEARTBEAT_CALL_TIMEOUT,
    HEARTBEAT_DEFAULT_INTERVAL,
    TASK_WORKER_CALL_TIMEOUT,
)
from flwr.proto.runtime_pb2 import SendTaskHeartbeatResponse  # pylint: disable=E0611

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

    def test_stop_respects_timeout(self) -> None:
        """Test that stop returns when the heartbeat function is blocked."""
        heartbeat_started = threading.Event()
        release_heartbeat = threading.Event()

        def heartbeat_fn() -> bool:
            heartbeat_started.set()
            release_heartbeat.wait()
            return True

        sender = HeartbeatSender(heartbeat_fn)
        sender.start()
        self.assertTrue(heartbeat_started.wait(timeout=1.0))

        started_at = time.monotonic()
        sender.stop(timeout=0.01)

        self.assertLess(time.monotonic() - started_at, 0.2)
        self.assertTrue(sender._thread.is_alive())

        release_heartbeat.set()
        sender._thread.join(timeout=1.0)
        self.assertFalse(sender._thread.is_alive())

    def test_stop_after_sender_thread_exits(self) -> None:
        """Test that stop tolerates a sender thread that has already exited."""
        sender: HeartbeatSender

        def heartbeat_fn() -> bool:
            sender._stop_event.set()
            return True

        sender = HeartbeatSender(heartbeat_fn)
        sender.start()
        sender._thread.join(timeout=1.0)
        self.assertFalse(sender._thread.is_alive())

        sender.stop(timeout=0.01)

    def test_grpc_heartbeat_uses_timeout(self) -> None:
        """Test that gRPC heartbeats have a deadline."""
        stub = Mock()
        stub.SendTaskHeartbeat.return_value = SendTaskHeartbeatResponse(success=True)

        heartbeat_fn = make_task_heartbeat_fn_grpc(stub)
        self.assertTrue(heartbeat_fn())

        self.assertEqual(
            stub.SendTaskHeartbeat.call_args.kwargs["timeout"],
            TASK_WORKER_CALL_TIMEOUT,
        )

    def test_heartbeat_interval_accounts_for_rpc_timeout(self) -> None:
        """Test that a shorter RPC deadline does not increase heartbeat load."""
        self.assertEqual(self.heartbeat_sender._call_timeout, HEARTBEAT_CALL_TIMEOUT)
        sender = HeartbeatSender(
            Mock(return_value=True), call_timeout=TASK_WORKER_CALL_TIMEOUT
        )

        def stop_after_wait(_timeout: float) -> bool:
            sender._stop_event.set()
            return True

        with (
            patch("flwr.supercore.heartbeat.random.uniform", return_value=0.0),
            patch.object(
                sender._stop_event, "wait", side_effect=stop_after_wait
            ) as mock_wait,
        ):
            sender._run()

        expected_interval = (
            HEARTBEAT_DEFAULT_INTERVAL - TASK_WORKER_CALL_TIMEOUT
        ) * HEARTBEAT_BASE_MULTIPLIER
        mock_wait.assert_called_once_with(expected_interval)

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
