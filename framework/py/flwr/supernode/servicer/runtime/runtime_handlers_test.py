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
"""Tests for SuperNode Runtime API handlers."""

import unittest
from unittest.mock import Mock

from parameterized import parameterized

from flwr.app import Context
from flwr.app.message import make_message
from flwr.common.constant import SubStatus
from flwr.common.serde import context_to_proto, message_to_proto
from flwr.common.serde_test import RecordMaker
from flwr.proto.control_pb2 import StartAutomationRequest  # pylint:disable=E0611
from flwr.proto.message_pb2 import (  # pylint:disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
)
from flwr.proto.runtime_pb2 import (  # pylint:disable=E0611
    GetConnectorRequest,
    GetNodesRequest,
    PullAppMessagesRequest,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint:disable=E0611
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import get_object_tree
from flwr.supercore.run import Run

from . import runtime_handlers


class TestSuperNodeRuntimeHandlers(unittest.TestCase):
    """Tests for SuperNode Runtime handlers."""

    def setUp(self) -> None:
        """Initialize test state."""
        self.maker = RecordMaker()
        self.state = Mock()

    def test_pull_task_input_activates_task(self) -> None:
        """PullTaskInput should activate the authenticated task."""
        run = Run.create_empty(run_id=61016)
        run.fab_id = "mock/mock"
        run.fab_version = "v1.0.0"
        run.fab_hash = "fab-hash"
        run.series_id = 777
        app_context = Context(
            run_id=run.run_id,
            node_id=1,
            node_config={"nodeconfig1": 4.2},
            state=self.maker.recorddict(1, 1, 1),
            run_config={"runconfig1": 6.1},
            series_id=run.series_id,
        )
        fab = Fab("fab-hash", b"fab-content", {"sig": "value"})
        task = Task(task_id=123, run_id=run.run_id)
        self.state.get_run.return_value = run
        self.state.get_run_series_context.return_value = app_context
        self.state.get_fab.return_value = fab
        self.state.activate_task.return_value = True

        response = runtime_handlers.pull_task_input(
            PullTaskInputRequest(), self.state, task
        )

        self.assertIsInstance(response, PullTaskInputResponse)
        self.state.get_run_series_context.assert_called_once_with(run.series_id)
        self.state.activate_task.assert_called_once_with(task_id=task.task_id)

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
    def test_pull_task_input_raises_flower_error(
        self,
        _name: str,
        conditions: tuple[bool, bool, bool, bool],
        expected_code: ApiErrorCode,
    ) -> None:
        """PullTaskInput should raise the relevant FlowerError."""
        run_exists, context_exists, fab_exists, task_started = conditions
        run = Run.create_empty(run_id=61016)
        run.fab_hash = "fab-hash"
        run.series_id = 777
        task = Task(task_id=123, run_id=run.run_id)
        self.state.get_run.return_value = run if run_exists else None
        self.state.get_run_series_context.return_value = (
            Mock() if context_exists else None
        )
        self.state.get_fab.return_value = (
            Fab("fab-hash", b"fab-content", {}) if fab_exists else None
        )
        self.state.activate_task.return_value = task_started

        with self.assertRaises(FlowerError) as error:
            runtime_handlers.pull_task_input(PullTaskInputRequest(), self.state, task)

        self.assertEqual(error.exception.code, expected_code)

    def test_push_task_output_finishes_task(self) -> None:
        """PushTaskOutput should finish the authenticated task."""
        run = Run.create_empty(run_id=61016)
        run.series_id = 777
        task = Task(task_id=123, run_id=run.run_id)
        app_context = Context(
            run_id=run.run_id,
            node_id=1,
            node_config={},
            state=self.maker.recorddict(1, 1, 1),
            run_config={},
            series_id=run.series_id,
        )
        request = PushTaskOutputRequest(
            context=context_to_proto(app_context), sub_status=SubStatus.COMPLETED
        )
        self.state.finish_task.return_value = True
        self.state.get_run.return_value = run

        response = runtime_handlers.push_task_output(request, self.state, task)

        self.assertIsInstance(response, PushTaskOutputResponse)
        self.state.set_run_series_context.assert_called_once()
        self.assertEqual(
            self.state.set_run_series_context.call_args.args[0], run.series_id
        )
        self.state.finish_task.assert_called_once_with(
            task_id=task.task_id,
            sub_status=request.sub_status,
            details=request.details,
        )

    def test_pull_messages_returns_empty_response(self) -> None:
        """PullMessages should return an empty response when none is available."""
        self.state.get_messages.return_value = []
        task = Task(run_id=61016)

        response = runtime_handlers.pull_messages(
            PullAppMessagesRequest(), self.state, task
        )

        self.assertEqual(list(response.messages_list), [])
        self.assertEqual(list(response.message_object_trees), [])
        self.state.record_message_processing_start.assert_not_called()

    @parameterized.expand([(0, 0), (1, 0), (0, 1), (2, 1), (1, 2)])  # type: ignore
    def test_push_messages_rejects_invalid_message_count(
        self, message_count: int, object_tree_count: int
    ) -> None:
        """PushMessages should reject anything other than one message/tree."""
        message = make_message(
            metadata=self.maker.metadata(), content=self.maker.recorddict(1, 1, 1)
        )
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)] * message_count,
            message_object_trees=[get_object_tree(message)] * object_tree_count,
        )

        with self.assertRaises(FlowerError) as error:
            runtime_handlers.push_messages(
                request, self.state, Task(run_id=message.metadata.run_id)
            )

        self.assertEqual(
            error.exception.code, ApiErrorCode.RUNTIME_INVALID_MESSAGE_COUNT
        )
        self.state.record_message_processing_end.assert_not_called()
        self.state.store_message_and_object_tree.assert_not_called()

    def test_push_messages_stores_message_and_object_tree(self) -> None:
        """PushMessages should store the message and preregister its object tree."""
        message = make_message(
            metadata=self.maker.metadata(), content=self.maker.recorddict(1, 1, 1)
        )
        object_tree = get_object_tree(message)
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)],
            message_object_trees=[object_tree],
        )
        self.state.store_message_and_object_tree.return_value = (True, ["object-id"])
        self.state.start_session.return_value = "session-id"

        response = runtime_handlers.push_messages(
            request, self.state, Task(run_id=message.metadata.run_id)
        )

        self.state.record_message_processing_end.assert_called_once_with(
            message_id=message.metadata.reply_to_message_id
        )
        self.state.start_session.assert_called_once_with(message.metadata.run_id)
        stored_message, stored_tree, session_id = (
            self.state.store_message_and_object_tree.call_args.args
        )
        self.assertEqual(
            stored_message.metadata.message_id, message.metadata.message_id
        )
        self.assertEqual(stored_tree, object_tree)
        self.assertEqual(session_id, "session-id")
        self.assertEqual(list(response.objects_to_push), ["object-id"])
        self.assertEqual(response.session_id, "session-id")

    def test_push_messages_rejects_mismatched_run_id(self) -> None:
        """PushMessages should reject messages from another run."""
        message = make_message(
            metadata=self.maker.metadata(), content=self.maker.recorddict(1, 1, 1)
        )
        request = PushAppMessagesRequest(
            messages_list=[message_to_proto(message)],
            message_object_trees=[get_object_tree(message)],
        )

        with self.assertRaises(FlowerError) as error:
            runtime_handlers.push_messages(
                request, self.state, Task(run_id=message.metadata.run_id + 1)
            )

        self.assertEqual(
            error.exception.code, ApiErrorCode.RUNTIME_MESSAGE_RUN_ID_MISMATCH
        )
        self.state.store_message_and_object_tree.assert_not_called()

    def test_push_object_uses_state(self) -> None:
        """PushObject should delegate session validation and storage to state."""
        request = PushObjectRequest(
            run_id=456,
            session_id="session-id",
            object_id="object-id",
            object_content=b"content",
        )
        self.state.store_object.return_value = True

        response = runtime_handlers.push_object(request, self.state, Task(run_id=123))

        self.state.store_object.assert_called_once_with(
            123, "session-id", "object-id", b"content"
        )
        self.assertTrue(response.stored)

    def test_pull_object_uses_state(self) -> None:
        """PullObject should delegate retrieval to state."""
        self.state.get_object.return_value = b"content"

        response = runtime_handlers.pull_object(
            PullObjectRequest(run_id=456, object_id="object-id"),
            self.state,
            Task(run_id=123),
        )

        self.state.get_object.assert_called_once_with(123, "object-id")
        self.assertEqual(
            response,
            PullObjectResponse(
                object_found=True, object_available=True, object_content=b"content"
            ),
        )

    @parameterized.expand(  # type: ignore
        [
            (
                "start_automation",
                runtime_handlers.start_automation,
                StartAutomationRequest(),
                ApiErrorCode.RUNTIME_AUTOMATION_CREATION_NOT_ALLOWED,
            ),
            (
                "get_connector",
                runtime_handlers.get_connector,
                GetConnectorRequest(),
                ApiErrorCode.RUNTIME_CONNECTOR_CREDENTIALS_NOT_AVAILABLE,
            ),
            (
                "get_nodes",
                runtime_handlers.get_nodes,
                GetNodesRequest(),
                ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE,
            ),
        ]
    )
    def test_server_side_endpoint_permission_denied(
        self, _name: str, handler: Mock, request: object, expected_code: ApiErrorCode
    ) -> None:
        """Server-side endpoints should be unavailable to ClientApp tasks."""
        with self.assertRaises(FlowerError) as error:
            handler(request)

        self.assertEqual(error.exception.code, expected_code)
