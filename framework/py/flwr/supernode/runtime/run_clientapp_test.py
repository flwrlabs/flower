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
from unittest.mock import Mock, patch

import httpx

from flwr.common.serde import fab_to_proto
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    PullAppMessagesResponse,
    PullTaskInputResponse,
)
from flwr.supercore.exit import ExitCode
from flwr.supercore.fab import Fab
from flwr.supercore.interceptors import (
    RuntimeTokenHttpInterceptor,
    RuntimeVersionHttpInterceptor,
)

from .run_clientapp import pull_task_input, run_clientapp


class TestRunClientApp(unittest.TestCase):
    """Tests for `run_clientapp`."""

    def test_pull_task_input_raises_when_no_message_received(self) -> None:
        """`pull_task_input` should reject an empty PullMessages response."""
        stub = Mock()
        stub.PullTaskInput.return_value = PullTaskInputResponse(
            context=ProtoContext(run_id=61016, node_id=123),
            run=ProtoRun(run_id=61016),
            fab=fab_to_proto(Fab("hash", b"content", {})),
        )
        stub.PullMessages.return_value = PullAppMessagesResponse()

        with self.assertRaisesRegex(
            RuntimeError, "No messages received from Runtime API"
        ):
            pull_task_input(stub)

    def test_run_clientapp_adds_client_interceptors(self) -> None:
        """`run_clientapp` should add interceptors to the Runtime HTTP client."""
        with patch(
            "flwr.supernode.runtime.run_clientapp.RuntimeHttpClient.from_server_address",
            side_effect=RuntimeError,
        ) as from_server_address:
            with self.assertRaises(RuntimeError):
                run_clientapp("127.0.0.1:9094", insecure=True, token="test-token")

        kwargs = from_server_address.call_args.kwargs
        interceptors = kwargs["interceptors"]
        self.assertIsNotNone(interceptors)
        assert interceptors is not None
        self.assertEqual(len(interceptors), 2)
        self.assertIsInstance(interceptors[0], RuntimeVersionHttpInterceptor)
        self.assertIsInstance(interceptors[1], RuntimeTokenHttpInterceptor)
        # pylint: disable-next=protected-access
        self.assertEqual(interceptors[0]._metadata.component_name, "flwr-clientapp")

    def test_run_clientapp_exits_nonzero_on_http_error(self) -> None:
        """`run_clientapp` should not report success after Runtime API failures."""
        with (
            patch(
                "flwr.supernode.runtime.run_clientapp.RuntimeHttpClient.from_server_address"
            ),
            patch("flwr.supernode.runtime.run_clientapp.HeartbeatSender"),
            patch(
                "flwr.supernode.runtime.run_clientapp.pull_task_input",
                side_effect=httpx.ConnectError("Connection failed"),
            ),
            patch("flwr.supernode.runtime.run_clientapp.flwr_exit") as flwr_exit,
        ):
            flwr_exit.side_effect = RuntimeError
            with self.assertRaises(RuntimeError):
                run_clientapp("127.0.0.1:9094", insecure=True, token="test-token")

        flwr_exit.assert_called_once()
        self.assertEqual(
            flwr_exit.call_args.kwargs["code"], ExitCode.TASK_PROC_EXCEPTION
        )
