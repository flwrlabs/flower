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
"""Tests for simulation runtime wiring."""


import unittest
from queue import Queue
from unittest.mock import Mock, patch

from .app import run_simulation_process


class TestRunSimulationProcess(unittest.TestCase):
    """Tests for `run_simulation_process`."""

    @patch("flwr.simulation.app.flwr_exit")
    @patch("flwr.simulation.app.register_signal_handlers")
    @patch("flwr.simulation.app.SimulationIoConnection")
    def test_run_simulation_process_passes_token_to_connection(
        self,
        mock_connection_cls: Mock,
        _mock_register_signal_handlers: Mock,
        mock_flwr_exit: Mock,
    ) -> None:
        """`run_simulation_process` should pass token into SimulationIoConnection."""
        mock_conn = Mock()
        mock_conn.configure_mock(
            **{"_stub.PullAppInputs.side_effect": RuntimeError("boom")}
        )
        mock_connection_cls.return_value = mock_conn

        run_simulation_process(
            serverappio_api_address="127.0.0.1:9091",
            log_queue=Queue(),
            token="test-token",
        )

        mock_connection_cls.assert_called_once_with(
            serverappio_api_address="127.0.0.1:9091",
            root_certificates=None,
            token="test-token",
        )
        mock_flwr_exit.assert_called_once()
