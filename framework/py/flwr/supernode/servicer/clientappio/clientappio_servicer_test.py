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
"""Test the ClientAppIo API servicer."""


import unittest
from unittest.mock import Mock, patch

import grpc
from parameterized import parameterized

from flwr.app import Context
from flwr.app.message import make_message
from flwr.common.constant import SubStatus
from flwr.common.serde import context_to_proto, fab_to_proto, message_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.proto.appio_pb2 import (  # pylint:disable=E0611
    GetNodesRequest,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
    SendTaskHeartbeatRequest,
    SendTaskHeartbeatResponse,
)
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint:disable=E0611
from flwr.proto.message_pb2 import (  # pylint:disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint:disable=E0611
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.run import Run
from flwr.supernode.runtime.run_clientapp import (
    pull_task_input,
    push_message,
    push_task_output,
)

from .clientappio_servicer import ClientAppIoServicer


class TestClientAppIoServicer(unittest.TestCase):
    """Tests for `ClientAppIoServicer` class."""

    def setUp(self) -> None:
        """Initialize."""
        self.maker = RecordMaker()
        self.mock_stub = Mock()
        self.mock_state = Mock()
        mock_state_factory = Mock()
        mock_state_factory.state.return_value = self.mock_state
        self.servicer = ClientAppIoServicer(mock_state_factory, Mock())

    def test_pull_task_input(self) -> None:
        """Test pulling messages from SuperNode."""
        # Prepare
        mock_message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(3, 2, 1),
        )
        mock_fab = Fab(
            hash_str="abc123#$%",
            content=b"\xf3\xf5\xf8\x98",
            verifications={"ab12#$%": "abc123#$%"},
        )
        mock_response = PullTaskInputResponse(
            context=ProtoContext(node_id=123),
            run=ProtoRun(run_id=61016, fab_id="mock/mock", fab_version="v1.0.0"),
            fab=fab_to_proto(mock_fab),
        )
        self.mock_stub.PullMessages.return_value = PullAppMessagesResponse(
            messages_list=[message_to_proto(mock_message)],
            message_object_trees=[get_object_tree(mock_message)],
        )
        # Create series of responses for PullObject
        # Adding responses for objects in a post-order traversal of object tree order
        all_objects = get_all_nested_objects(mock_message)
        all_objects[mock_message.object_id] = mock_message

        # Get the object tree and iterate in the correct order
        def pull_object_side_effect(request: PullObjectRequest) -> PullObjectResponse:
            obj_id = request.object_id
            return PullObjectResponse(
                object_found=True,
                object_available=True,
                object_content=all_objects[obj_id].deflate(),
            )

        self.mock_stub.PullObject.side_effect = pull_object_side_effect
        self.mock_stub.PullTaskInput.return_value = mock_response

        # Execute
        message, context, run, fab = pull_task_input(self.mock_stub)

        # Assert
        self.mock_stub.PullTaskInput.assert_called_once()
        self.assertEqual(len(message.content.array_records), 3)
        self.assertEqual(len(message.content.metric_records), 2)
        self.assertEqual(len(message.content.config_records), 1)
        self.assertEqual(context.node_id, 123)
        self.assertEqual(run.run_id, 61016)
        self.assertEqual(run.fab_id, "mock/mock")
        self.assertEqual(run.fab_version, "v1.0.0")
        self.assertEqual(fab.hash_str, mock_fab.hash_str)
        self.assertEqual(fab.content, mock_fab.content)

    def test_push_task_output(self) -> None:
        """Test pushing messages to SuperNode."""
        # Prepare: Create Message and context
        sub_status = SubStatus.COMPLETED
        details = "ClientApp execution completed successfully"
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

        # Prepare: Mock PushTaskOutput RPC call
        mock_response = PushTaskOutputResponse()
        self.mock_stub.PushTaskOutput.return_value = mock_response

        # Prepare: Mock PushMessages RPC call
        object_tree = get_object_tree(message)
        all_obj_ids = [tree.object_id for tree in iterate_object_tree(object_tree)]
        self.mock_stub.PushMessages.return_value = PushAppMessagesResponse(
            message_ids=[message.object_id],
            objects_to_push=all_obj_ids,
        )

        # Prepare: Mock PushObject RPC calls
        pushed_obj_ids = set()

        def mock_push_object(request: PushObjectRequest) -> PushObjectResponse:
            """Mock PushObject RPC call."""
            pushed_obj_ids.add(request.object_id)
            return PushObjectResponse(stored=True)

        self.mock_stub.PushObject.side_effect = mock_push_object

        # Execute
        push_message(self.mock_stub, message, context)
        push_task_output(
            stub=self.mock_stub,
            context=context,
            sub_status=sub_status,
            details=details,
        )

        # Assert
        self.mock_stub.PushTaskOutput.assert_called_once()
        self.mock_stub.PushMessages.assert_called_once()
        self.assertSetEqual(pushed_obj_ids, set(all_obj_ids))
        push_outputs_request = self.mock_stub.PushTaskOutput.call_args.args[0]
        self.assertEqual(push_outputs_request.sub_status, sub_status)
        self.assertEqual(push_outputs_request.details, details)

    def test_servicer_pull_task_input_activates_task(self) -> None:
        """PullTaskInput should activate the authenticated task."""
        run_id = 61016
        task_id = 123
        request = PullTaskInputRequest()

        run = Run.create_empty(run_id=run_id)
        run.fab_id = "mock/mock"
        run.fab_version = "v1.0.0"
        run.fab_hash = "fab-hash"
        run.series_id = 777

        app_context = Context(
            run_id=run_id,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recorddict(1, 1, 1),
            run_config={"runconfig1": 6.1},
            series_id=run.series_id,
        )
        fab = Fab(
            hash_str="fab-hash",
            content=b"fab-content",
            verifications={"sig": "value"},
        )

        self.mock_state.get_run_series_context.return_value = app_context
        self.mock_state.get_run.return_value = run
        self.mock_state.get_fab.return_value = fab

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(task_id=task_id, run_id=run_id),
        ):
            response = self.servicer.PullTaskInput(request, Mock())

        self.assertIsInstance(response, PullTaskInputResponse)
        self.mock_state.get_run_series_context.assert_called_once_with(run.series_id)
        self.mock_state.activate_task.assert_called_once_with(task_id=task_id)

    def test_servicer_push_task_output_finishes_task(self) -> None:
        """PushTaskOutput should finish the authenticated task."""
        run_id = 61016
        task_id = 123
        run = Run.create_empty(run_id=run_id)
        run.series_id = 777
        app_context = Context(
            run_id=run_id,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recorddict(1, 1, 1),
            run_config={"runconfig1": 6.1},
            series_id=run.series_id,
        )
        request = PushTaskOutputRequest(
            context=context_to_proto(app_context),
            sub_status=SubStatus.COMPLETED,
        )
        self.mock_state.get_run.return_value = run

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(task_id=task_id, run_id=run_id),
        ):
            response = self.servicer.PushTaskOutput(request, Mock())

        self.assertIsInstance(response, PushTaskOutputResponse)
        self.mock_state.set_run_series_context.assert_called_once()
        args, _ = self.mock_state.set_run_series_context.call_args
        self.assertEqual(args[0], run.series_id)
        self.mock_state.finish_task.assert_called_once()
        finish_task_kwargs = self.mock_state.finish_task.call_args.kwargs
        self.assertEqual(finish_task_kwargs["task_id"], task_id)
        self.assertEqual(finish_task_kwargs["sub_status"], request.sub_status)

    def test_servicer_pull_messages_aborts_when_no_message_found(self) -> None:
        """PullMessages should abort cleanly when no message is available."""
        run_id = 61016
        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        self.mock_state.get_messages.return_value = []

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=run_id),
        ):
            with self.assertRaises(grpc.RpcError):
                self.servicer.PullMessages(PullAppMessagesRequest(), context)

        context.abort.assert_called_once_with(
            grpc.StatusCode.NOT_FOUND,
            f"No message found for run {run_id} in NodeState.",
        )
        self.mock_state.record_message_processing_start.assert_not_called()

    @parameterized.expand([(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)])  # type: ignore
    def test_servicer_push_messages_rejects_invalid_message_count(
        self, message_count: int, object_tree_count: int
    ) -> None:
        """PushMessages should reject anything other than one message/tree."""
        run_id = 61016
        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(1, 1, 1),
        )
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)] * message_count,
            message_object_trees=[get_object_tree(message)] * object_tree_count,
        )

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=run_id),
        ):
            with self.assertRaises(grpc.RpcError):
                self.servicer.PushMessages(request, context)

        context.abort.assert_called_once_with(
            grpc.StatusCode.INVALID_ARGUMENT,
            "ClientAppIo.PushMessages expects exactly one message and one object tree.",
        )
        self.mock_state.record_message_processing_end.assert_not_called()
        self.mock_state.store_message_and_object_tree.assert_not_called()

    def test_servicer_push_messages_stores_message_and_object_tree(self) -> None:
        """PushMessages should store the message and preregister its object tree."""
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(1, 1, 1),
        )
        object_tree = get_object_tree(message)
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)],
            message_object_trees=[object_tree],
        )
        self.mock_state.store_message_and_object_tree.return_value = (
            True,
            ["object-id"],
        )
        self.mock_state.start_session.return_value = "session-id"

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=message.metadata.run_id),
        ):
            response = self.servicer.PushMessages(request, Mock())

        self.mock_state.record_message_processing_end.assert_called_once_with(
            message_id=message.metadata.reply_to_message_id
        )
        self.mock_state.start_session.assert_called_once_with(message.metadata.run_id)
        self.mock_state.store_message_and_object_tree.assert_called_once()
        stored_message, stored_tree, session_id = (
            self.mock_state.store_message_and_object_tree.call_args.args
        )
        self.assertEqual(
            stored_message.metadata.message_id, message.metadata.message_id
        )
        self.assertEqual(stored_tree, object_tree)
        self.assertEqual(session_id, "session-id")
        self.assertEqual(list(response.objects_to_push), ["object-id"])
        self.assertEqual(response.session_id, "session-id")

    def test_push_object_uses_state(self) -> None:
        """PushObject should delegate session validation and storage to state."""
        request = PushObjectRequest(
            run_id=456,
            session_id="session-id",
            object_id="object-id",
            object_content=b"content",
        )
        self.mock_state.store_object.return_value = True

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=123),
        ):
            response = self.servicer.PushObject(request, Mock())

        self.mock_state.store_object.assert_called_once_with(
            123, "session-id", "object-id", b"content"
        )
        self.assertTrue(response.stored)

    def test_pull_object_uses_state(self) -> None:
        """PullObject should delegate retrieval and expiry cleanup to state."""
        request = PullObjectRequest(run_id=456, object_id="object-id")
        self.mock_state.get_object.return_value = b"content"

        with patch(
            "flwr.supernode.servicer.clientappio.clientappio_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=123),
        ):
            response = self.servicer.PullObject(request, Mock())

        self.mock_state.get_object.assert_called_once_with(123, "object-id")
        self.assertTrue(response.object_found)
        self.assertTrue(response.object_available)
        self.assertEqual(response.object_content, b"content")

    def test_get_nodes_unimplemented(self) -> None:
        """GetNodes should be unavailable on ClientAppIo."""
        context = Mock()
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            self.servicer.GetNodes(GetNodesRequest(), context)

        context.abort.assert_called_once_with(
            grpc.StatusCode.UNIMPLEMENTED,
            "GetNodes is not available on ClientAppIo.",
        )

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_send_task_heartbeat(self, success: bool) -> None:
        """Test sending a task heartbeat."""
        # Prepare
        task_id = 123
        request = SendTaskHeartbeatRequest()
        self.mock_state.acknowledge_task_heartbeat.return_value = success

        # Execute
        with patch(
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Mock(task_id=task_id),
        ):
            response = self.servicer.SendTaskHeartbeat(request, Mock())

        # Assert
        self.assertIsInstance(response, SendTaskHeartbeatResponse)
        self.assertEqual(response.success, success)
        self.mock_state.acknowledge_task_heartbeat.assert_called_once_with(task_id)
