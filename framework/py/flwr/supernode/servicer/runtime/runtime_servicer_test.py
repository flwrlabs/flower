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
"""Test the SuperNode Runtime API servicer implementation."""


import unittest
from unittest.mock import Mock, patch

from parameterized import parameterized

from flwr.app import Context
from flwr.app.message import make_message
from flwr.common.constant import SubStatus
from flwr.common.serde import context_to_proto, fab_to_proto, message_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.proto.control_pb2 import StartAutomationRequest  # pylint:disable=E0611
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint:disable=E0611
from flwr.proto.message_pb2 import (  # pylint:disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint:disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint:disable=E0611
    GetConnectorRequest,
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
from flwr.supercore.error import ApiErrorCode, FlowerError
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

from .runtime_servicer import SuperNodeRuntimeServicer


class TestSuperNodeRuntimeServicer(unittest.TestCase):
    """Tests for `SuperNodeRuntimeServicer` class."""

    def setUp(self) -> None:
        """Initialize."""
        self.maker = RecordMaker()
        self.mock_stub = Mock()
        self.mock_state = Mock()
        mock_state_factory = Mock()
        mock_state_factory.state.return_value = self.mock_state
        self.servicer = SuperNodeRuntimeServicer(mock_state_factory, Mock())

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
            client=self.mock_stub,
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
            "flwr.supernode.servicer.runtime.runtime_servicer."
            "get_authenticated_task",
            return_value=Mock(task_id=task_id, run_id=run_id),
        ):
            response = self.servicer.PullTaskInput(request, Mock())

        self.assertIsInstance(response, PullTaskInputResponse)
        self.mock_state.get_run_series_context.assert_called_once_with(run.series_id)
        self.mock_state.activate_task.assert_called_once_with(task_id=task_id)

    @parameterized.expand(  # type: ignore
        [
            (
                "run_not_found",
                (False, False, False, False),
                ApiErrorCode.RUN_ID_NOT_FOUND,
            ),
            (
                "context_not_found",
                (True, False, False, False),
                ApiErrorCode.RUNTIME_RUN_SERIES_CONTEXT_NOT_FOUND,
            ),
            (
                "fab_not_found",
                (True, True, False, False),
                ApiErrorCode.RUNTIME_FAB_NOT_FOUND,
            ),
            (
                "task_start_failed",
                (True, True, True, False),
                ApiErrorCode.RUNTIME_TASK_START_FAILED,
            ),
        ]
    )
    def test_servicer_pull_task_input_raises_flower_error(
        self,
        _name: str,
        conditions: tuple[bool, bool, bool, bool],
        expected_code: ApiErrorCode,
    ) -> None:
        """PullTaskInput should raise the relevant FlowerError."""
        run_exists, context_exists, fab_exists, task_started = conditions
        run_id = 61016
        task_id = 123
        run = Run.create_empty(run_id=run_id)
        run.fab_hash = "fab-hash"
        run.series_id = 777
        self.mock_state.get_run.return_value = run if run_exists else None
        self.mock_state.get_run_series_context.return_value = (
            Mock() if context_exists else None
        )
        self.mock_state.get_fab.return_value = (
            Fab(hash_str="fab-hash", content=b"fab-content", verifications={})
            if fab_exists
            else None
        )
        self.mock_state.activate_task.return_value = task_started

        with (
            patch(
                "flwr.supernode.servicer.runtime.runtime_servicer."
                "get_authenticated_task",
                return_value=Mock(task_id=task_id, run_id=run_id),
            ),
            self.assertRaises(FlowerError) as error,
        ):
            self.servicer.PullTaskInput(PullTaskInputRequest(), Mock())

        self.assertEqual(error.exception.code, expected_code)

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
            "flwr.supernode.servicer.runtime.runtime_servicer."
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

    def test_servicer_pull_messages_returns_empty_response(self) -> None:
        """PullMessages should return an empty response when no message is available."""
        run_id = 61016
        context = Mock()
        self.mock_state.get_messages.return_value = []

        with patch(
            "flwr.supernode.servicer.runtime.runtime_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=run_id),
        ):
            response = self.servicer.PullMessages(PullAppMessagesRequest(), context)

        self.assertEqual(list(response.messages_list), [])
        self.assertEqual(list(response.message_object_trees), [])
        context.abort.assert_not_called()
        self.mock_state.record_message_processing_start.assert_not_called()

    @parameterized.expand([(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)])  # type: ignore
    def test_servicer_push_messages_rejects_invalid_message_count(
        self, message_count: int, object_tree_count: int
    ) -> None:
        """PushMessages should reject anything other than one message/tree."""
        run_id = 61016
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(1, 1, 1),
        )
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)] * message_count,
            message_object_trees=[get_object_tree(message)] * object_tree_count,
        )

        with (
            patch(
                "flwr.supernode.servicer.runtime.runtime_servicer."
                "get_authenticated_task",
                return_value=Mock(run_id=run_id),
            ),
            self.assertRaises(FlowerError) as error,
        ):
            self.servicer.PushMessages(request, Mock())

        self.assertEqual(
            error.exception.code, ApiErrorCode.RUNTIME_INVALID_MESSAGE_COUNT
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
            "flwr.supernode.servicer.runtime.runtime_servicer."
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

    def test_servicer_push_messages_rejects_mismatched_run_id(self) -> None:
        """PushMessages should reject messages from another run."""
        message = make_message(
            metadata=self.maker.metadata(),
            content=self.maker.recorddict(1, 1, 1),
        )
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)],
            message_object_trees=[get_object_tree(message)],
        )

        with (
            patch(
                "flwr.supernode.servicer.runtime.runtime_servicer."
                "get_authenticated_task",
                return_value=Mock(run_id=message.metadata.run_id + 1),
            ),
            self.assertRaises(FlowerError) as error,
        ):
            self.servicer.PushMessages(request, Mock())

        self.assertEqual(
            error.exception.code, ApiErrorCode.RUNTIME_MESSAGE_RUN_ID_MISMATCH
        )
        self.mock_state.store_message_and_object_tree.assert_not_called()

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
            "flwr.supernode.servicer.runtime.runtime_servicer."
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
            "flwr.supernode.servicer.runtime.runtime_servicer."
            "get_authenticated_task",
            return_value=Mock(run_id=123),
        ):
            response = self.servicer.PullObject(request, Mock())

        self.mock_state.get_object.assert_called_once_with(123, "object-id")
        self.assertTrue(response.object_found)
        self.assertTrue(response.object_available)
        self.assertEqual(response.object_content, b"content")

    @parameterized.expand(  # type: ignore
        [
            (
                "start_automation",
                "StartAutomation",
                StartAutomationRequest(),
                ApiErrorCode.RUNTIME_AUTOMATION_CREATION_NOT_ALLOWED,
            ),
            (
                "get_connector",
                "GetConnector",
                GetConnectorRequest(),
                ApiErrorCode.RUNTIME_CONNECTOR_CREDENTIALS_NOT_AVAILABLE,
            ),
            (
                "get_nodes",
                "GetNodes",
                GetNodesRequest(),
                ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE,
            ),
        ]
    )
    def test_server_side_endpoint_permission_denied(
        self,
        _case_name: str,
        method_name: str,
        request: object,
        expected_code: ApiErrorCode,
    ) -> None:
        """Server-side endpoints should be unavailable to ClientApp tasks."""
        with self.assertRaises(FlowerError) as error:
            getattr(self.servicer, method_name)(request, Mock())

        self.assertEqual(error.exception.code, expected_code)

    @parameterized.expand([(True,), (False,)])  # type: ignore
    def test_send_task_heartbeat(self, success: bool) -> None:
        """Test sending a task heartbeat."""
        # Prepare
        task_id = 123
        request = SendTaskHeartbeatRequest()
        self.mock_state.acknowledge_task_heartbeat.return_value = success

        # Execute
        with patch(
            "flwr.supercore.servicer.runtime.runtime_servicer.get_authenticated_task",
            return_value=Mock(task_id=task_id),
        ):
            response = self.servicer.SendTaskHeartbeat(request, Mock())

        # Assert
        self.assertIsInstance(response, SendTaskHeartbeatResponse)
        self.assertEqual(response.success, success)
        self.mock_state.acknowledge_task_heartbeat.assert_called_once_with(task_id)
