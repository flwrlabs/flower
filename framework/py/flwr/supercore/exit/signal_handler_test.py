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
"""Tests for signal handler utils."""


import os
import signal
import unittest
from unittest.mock import Mock, patch

from flwr.common.telemetry import EventType

from .exit_handler import (
    add_exit_handler,
    registered_exit_handlers,
    trigger_exit_handlers,
)
from .signal_handler import register_signal_handlers


class TestExitHandlers(unittest.TestCase):
    """Tests for exit handler utils."""

    def setUp(self) -> None:
        """Clear all exit handlers before each test."""
        registered_exit_handlers.clear()

    @patch("flwr.supercore.exit.signal_handler.flwr_exit")
    def test_register_exit_handlers(self, mock_flwr_exit: Mock) -> None:
        """Test register_exit_handlers."""
        # Prepare
        handlers = [Mock(), Mock(), Mock()]
        register_signal_handlers(EventType.PING, exit_handlers=handlers[:-1])  # type: ignore
        add_exit_handler(handlers[-1])

        # Execute
        os.kill(os.getpid(), signal.SIGTERM)
        # This should be called in `flwr_exit`, but we patched it above
        trigger_exit_handlers()

        # Assert
        mock_flwr_exit.assert_called_once()
        for handler in handlers:
            handler.assert_called_once()

    def test_add_exit_handler(self) -> None:
        """Test add_exit_handler."""
        # Prepare
        handler = Mock()

        # Execute
        add_exit_handler(handler)

        # Assert
        self.assertIn(handler, registered_exit_handlers)
