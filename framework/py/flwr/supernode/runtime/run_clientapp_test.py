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

from flwr.app import Context
from flwr.app.message import make_message
from flwr.common.constant import SubStatus
from flwr.common.serde import context_to_proto, fab_to_proto, message_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    PullAppMessagesResponse,
    PullTaskInputResponse,
    PushAppMessagesResponse,
    PushTaskOutputResponse,
)
from flwr.supercore.exit import ExitCode
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.interceptors import (
    RuntimeTokenHttpInterceptor,
    RuntimeVersionHttpInterceptor,
)

from .run_clientapp import (
    pull_task_input,
    push_message,
    push_task_output,
    run_clientapp,
)


class TestRunClientApp(unittest.TestCase):
    """Tests for `run_clientapp`."""

    def setUp(self) -> None:
        """Initialize record fixtures."""
        self.maker = RecordMaker()

    def test_pull_task_input(self) -> None:
        """Test pulling messages from SuperNode."""
        stub = Mock()
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(3, 2, 1),
        )
        fab = Fab("abc123#$%", b"\xf3\xf5\xf8\x98", {"ab12#$%": "abc123#$%"})
        stub.PullTaskInput.return_value = PullTaskInputResponse(
            context=ProtoContext(node_id=123),
            run=ProtoRun(run_id=61016, fab_id="mock/mock", fab_version="v1.0.0"),
            fab=fab_to_proto(fab),
        )
        stub.PullMessages.return_value = PullAppMessagesResponse(
            messages_list=[message_to_proto(message)],
            message_object_trees=[get_object_tree(message)],
        )
        all_objects = get_all_nested_objects(message)
        all_objects[message.object_id] = message
        stub.PullObject.side_effect = lambda request: Mock(
            object_found=True,
            object_available=True,
            object_content=all_objects[request.object_id].deflate(),
        )

        pulled_message, context, run, pulled_fab = pull_task_input(stub)

        stub.PullTaskInput.assert_called_once()
        self.assertEqual(len(pulled_message.content.array_records), 3)
        self.assertEqual(len(pulled_message.content.metric_records), 2)
        self.assertEqual(len(pulled_message.content.config_records), 1)
        self.assertEqual(context.node_id, 123)
        self.assertEqual(run.run_id, 61016)
        self.assertEqual(run.fab_id, "mock/mock")
        self.assertEqual(run.fab_version, "v1.0.0")
        self.assertEqual(pulled_fab, fab)

    def test_push_task_output(self) -> None:
        """Test pushing messages and task output to SuperNode."""
        stub = Mock()
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(2, 2, 1),
        )
        context = Context(
            run_id=1,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recorddict(2, 2, 1),
            run_config={"runconfig1": 6.1},
        )
        object_tree = get_object_tree(message)
        object_ids = [tree.object_id for tree in iterate_object_tree(object_tree)]
        stub.PushMessages.return_value = PushAppMessagesResponse(
            message_ids=[message.object_id], objects_to_push=object_ids
        )
        stub.PushTaskOutput.return_value = PushTaskOutputResponse()

        push_message(stub, message, context)
        push_task_output(stub, context, SubStatus.COMPLETED, "completed")

        stub.PushTaskOutput.assert_called_once()
        stub.PushMessages.assert_called_once()
        self.assertCountEqual(
            [call.args[0].object_id for call in stub.PushObject.call_args_list],
            object_ids,
        )
        request = stub.PushTaskOutput.call_args.args[0]
        self.assertEqual(request.context, context_to_proto(context))
        self.assertEqual(request.sub_status, SubStatus.COMPLETED)
        self.assertEqual(request.details, "completed")

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
