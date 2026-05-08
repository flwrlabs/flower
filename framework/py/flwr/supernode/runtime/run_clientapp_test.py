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
"""Tests for ClientApp runtime wiring."""


import unittest
from unittest.mock import patch

import grpc

from flwr.common.exit import ExitCode
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)

from .run_clientapp import run_clientapp


class TestRunClientApp(unittest.TestCase):
    """Tests for `run_clientapp`."""

    def test_run_clientapp_adds_client_interceptors(self) -> None:
        """`run_clientapp` should add client interceptors to gRPC channel creation."""
        with patch(
            "flwr.supernode.runtime.run_clientapp.create_channel",
            side_effect=RuntimeError,
        ) as mock_create_channel:
            with self.assertRaises(RuntimeError):
                run_clientapp("127.0.0.1:9094", insecure=True, token="test-token")

        kwargs = mock_create_channel.call_args.kwargs
        interceptors = kwargs["interceptors"]
        self.assertIsNotNone(interceptors)
        assert interceptors is not None
        self.assertEqual(len(interceptors), 2)
        self.assertIsInstance(interceptors[0], RuntimeVersionClientInterceptor)
        self.assertIsInstance(interceptors[1], AppIoTokenClientInterceptor)
        # pylint: disable-next=protected-access
        self.assertEqual(interceptors[0]._metadata.component_name, "flwr-clientapp")

    def test_run_clientapp_exits_nonzero_on_grpc_error(self) -> None:
        """`run_clientapp` should not report success after AppIO gRPC failures."""
        with (
            patch("flwr.supernode.runtime.run_clientapp.create_channel") as channel,
            patch("flwr.supernode.runtime.run_clientapp.HeartbeatSender"),
            patch(
                "flwr.supernode.runtime.run_clientapp.pull_appinputs",
                side_effect=grpc.RpcError,
            ),
            patch("flwr.supernode.runtime.run_clientapp.flwr_exit") as flwr_exit,
        ):
            flwr_exit.side_effect = RuntimeError
            with self.assertRaises(RuntimeError):
                run_clientapp("127.0.0.1:9094", insecure=True, token="test-token")

        channel.return_value.subscribe.assert_called_once()
        flwr_exit.assert_called_once()
        self.assertEqual(
            flwr_exit.call_args.kwargs["code"],
            ExitCode.CLIENTAPP_COMMUNICATION_ERROR,
        )
