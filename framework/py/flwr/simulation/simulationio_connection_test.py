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
"""Tests for SimulationIoConnection wiring."""


import unittest
from unittest.mock import Mock, patch

from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)

from .simulationio_connection import SimulationIoConnection


class TestSimulationIoConnection(unittest.TestCase):
    """Tests for `SimulationIoConnection`."""

    @patch("flwr.simulation.simulationio_connection.wrap_stub")
    @patch("flwr.simulation.simulationio_connection.ServerAppIoStub")
    @patch("flwr.simulation.simulationio_connection.create_channel")
    def test_connect_adds_client_interceptors(
        self,
        mock_create_channel: Mock,
        _mock_serverappio_stub: Mock,
        _mock_wrap_stub: Mock,
    ) -> None:
        """`_connect` should pass version and token interceptors to create_channel."""
        mock_create_channel.return_value = Mock()
        conn = SimulationIoConnection(token="test-token")

        conn._connect()  # pylint: disable=protected-access

        kwargs = mock_create_channel.call_args.kwargs
        interceptors = kwargs["interceptors"]
        self.assertIsNotNone(interceptors)
        assert interceptors is not None
        self.assertEqual(len(interceptors), 2)
        self.assertIsInstance(interceptors[0], RuntimeVersionClientInterceptor)
        self.assertIsInstance(interceptors[1], AppIoTokenClientInterceptor)

    def test_init_requires_token(self) -> None:
        """`SimulationIoConnection` should require token values."""
        with self.assertRaises(TypeError):
            # pylint: disable-next=missing-kwoa
            SimulationIoConnection()  # type: ignore[call-arg]

    def test_init_rejects_empty_token(self) -> None:
        """`SimulationIoConnection` should reject empty token values."""
        with self.assertRaises(ValueError):
            SimulationIoConnection(token="")
