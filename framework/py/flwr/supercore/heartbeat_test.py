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


import signal
import time
import unittest
from unittest.mock import Mock, patch

import httpx
import pytest

from flwr.proto.runtime_pb2 import SendTaskHeartbeatResponse  # pylint: disable=E0611

from .heartbeat import HeartbeatSender, make_task_heartbeat_fn_http


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    """Create an HTTP status error for a Runtime request."""
    request = httpx.Request("POST", "http://runtime.example")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        "Runtime request failed", request=request, response=response
    )


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


def test_http_heartbeat_returns_true_on_success() -> None:
    """HTTP heartbeat should report a successful Runtime response."""
    stub = Mock()
    stub.SendTaskHeartbeat.return_value = SendTaskHeartbeatResponse(success=True)

    assert make_task_heartbeat_fn_http(stub)() is True


def test_http_heartbeat_returns_false_on_transport_error() -> None:
    """HTTP heartbeat should report transport errors as retryable failures."""
    stub = Mock()
    stub.SendTaskHeartbeat.side_effect = httpx.ConnectError("connection failed")

    assert make_task_heartbeat_fn_http(stub)() is False


@pytest.mark.parametrize("status_code", [503, 504])
def test_http_heartbeat_returns_false_on_transient_status(status_code: int) -> None:
    """HTTP heartbeat should report transient statuses as retryable failures."""
    stub = Mock()
    stub.SendTaskHeartbeat.side_effect = _http_status_error(status_code)

    assert make_task_heartbeat_fn_http(stub)() is False


def test_http_heartbeat_raises_non_transient_status_error() -> None:
    """HTTP heartbeat should preserve non-transient status errors."""
    stub = Mock()
    error = _http_status_error(500)
    stub.SendTaskHeartbeat.side_effect = error

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        make_task_heartbeat_fn_http(stub)()

    assert exc_info.value is error


def test_http_heartbeat_raises_sigint_when_rejected() -> None:
    """HTTP heartbeat should trigger graceful shutdown when rejected."""
    stub = Mock()
    stub.SendTaskHeartbeat.return_value = SendTaskHeartbeatResponse(success=False)

    with patch("flwr.supercore.heartbeat.signal.raise_signal") as raise_signal:
        result = make_task_heartbeat_fn_http(stub)()

    raise_signal.assert_called_once_with(signal.SIGINT)
    assert result is True
