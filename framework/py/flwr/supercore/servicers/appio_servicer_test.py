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
"""AppIoServicer tests."""


import unittest
from unittest.mock import Mock, patch

import grpc

from flwr.common.constant import Status
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    CreateTaskRequest,
    PullPendingTasksRequest,
    SendTaskHeartbeatRequest,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.task_pb2 import Task, TaskStatus  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TYPES_ALLOWED_TO_CREATE_TASKS, TaskType

from .appio_servicer import AppIoServicer


class _TestAppIoServicer(AppIoServicer):
    """Concrete AppIoServicer for tests."""

    def __init__(self, state: Mock) -> None:
        self._state = state

    def state(self) -> Mock:
        """Return mocked CoreState."""
        return self._state


class TestAppIoServicer(unittest.TestCase):
    """Tests for shared AppIoServicer task RPCs."""

    def setUp(self) -> None:
        """Set up test fixture."""
        self.state = Mock()
        self.servicer = _TestAppIoServicer(self.state)

    def test_pull_pending_tasks_returns_pending_tasks(self) -> None:
        """PullPendingTasks should return pending tasks from state."""
        # Prepare
        task = Task(
            task_id=123,
            run_id=456,
            status=TaskStatus(status=Status.PENDING, sub_status="", details=""),
        )
        self.state.get_tasks.return_value = [task]

        # Execute
        response = self.servicer.PullPendingTasks(PullPendingTasksRequest(), Mock())

        # Assert
        self.state.get_tasks.assert_called_once_with(
            statuses=[Status.PENDING], order_by="pending_at", ascending=True
        )
        self.assertEqual(len(response.tasks), 1)
        self.assertEqual(response.tasks[0].task_id, 123)

    def test_claim_task_returns_token_when_claim_succeeds(self) -> None:
        """ClaimTask should return the token from state."""
        # Prepare
        self.state.claim_task.return_value = "task-token"

        # Execute
        response = self.servicer.ClaimTask(ClaimTaskRequest(task_id=123), Mock())

        # Assert
        self.state.claim_task.assert_called_once_with(123)
        self.assertEqual(response.token, "task-token")

    def test_claim_task_returns_empty_token_when_claim_fails(self) -> None:
        """ClaimTask should return an empty token if the claim fails."""
        # Prepare
        self.state.claim_task.return_value = None

        # Execute
        response = self.servicer.ClaimTask(ClaimTaskRequest(task_id=123), Mock())

        # Assert
        self.state.claim_task.assert_called_once_with(123)
        self.assertFalse(response.HasField("token"))

    def test_send_task_heartbeat_acknowledges_authenticated_task(self) -> None:
        """SendTaskHeartbeat should use the authenticated task ID."""
        # Prepare
        self.state.acknowledge_task_heartbeat.return_value = True

        # Execute
        with patch(
            "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
            return_value=Mock(task_id=123),
        ):
            response = self.servicer.SendTaskHeartbeat(
                SendTaskHeartbeatRequest(), Mock()
            )

        # Assert
        self.state.acknowledge_task_heartbeat.assert_called_once_with(123)
        self.assertTrue(response.success)

    def test_create_task_returns_task_id(self) -> None:
        """CreateTask should create a task for the authenticated run."""
        # Prepare
        self.state.create_task.return_value = 456
        request = CreateTaskRequest(
            type=TaskType.MODEL,
            model_ref="models/abc",
        )

        # Execute
        with patch(
            "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
            return_value=Mock(run_id=123, type=TaskType.SERVER_APP),
        ):
            response = self.servicer.CreateTask(request, Mock())

        # Assert
        self.state.create_task.assert_called_once_with(
            task_type=TaskType.MODEL,
            run_id=123,
            fab_hash=None,
            model_ref="models/abc",
            connector_ref=None,
        )
        self.assertEqual(response.task_id, 456)

    def test_create_task_allows_app_task_types_to_request_creation(self) -> None:
        """CreateTask should allow app tasks to request task creation."""
        # Prepare
        self.state.create_task.return_value = 456
        request = CreateTaskRequest(type=TaskType.MODEL, model_ref="model")

        for requesting_task_type in TASK_TYPES_ALLOWED_TO_CREATE_TASKS:
            self.state.create_task.reset_mock()

            with self.subTest(requesting_task_type=requesting_task_type):
                # Execute
                with patch(
                    "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
                    return_value=Mock(run_id=123, type=requesting_task_type),
                ):
                    response = self.servicer.CreateTask(request, Mock())

                # Assert
                self.state.create_task.assert_called_once_with(
                    task_type=TaskType.MODEL,
                    run_id=123,
                    fab_hash=None,
                    model_ref="model",
                    connector_ref=None,
                )
                self.assertEqual(response.task_id, 456)

    def test_create_task_propagates_state_error(self) -> None:
        """CreateTask should let state-layer run validation errors propagate."""
        # Prepare
        self.state.create_task.side_effect = RuntimeError(
            "Run 123 not found. create_task requires an existing run."
        )

        # Execute
        with (
            patch(
                "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
                return_value=Mock(run_id=123, type=TaskType.SERVER_APP),
            ),
            self.assertRaises(RuntimeError) as err,
        ):
            self.servicer.CreateTask(
                CreateTaskRequest(type=TaskType.MODEL, model_ref="model"),
                Mock(),
            )

        # Assert
        self.assertEqual(
            str(err.exception),
            "Run 123 not found. create_task requires an existing run.",
        )
        self.state.create_task.assert_called_once_with(
            task_type=TaskType.MODEL,
            run_id=123,
            fab_hash=None,
            model_ref="model",
            connector_ref=None,
        )

    def test_create_task_aborts_if_required_field_is_missing(self) -> None:
        """CreateTask should validate task-type-specific required fields."""
        # Prepare
        test_cases = [
            (
                CreateTaskRequest(type=TaskType.SERVER_APP),
                "Task type 'flwr-serverapp' requires fab_hash.",
            ),
            (
                CreateTaskRequest(type=TaskType.MODEL),
                "Task type 'flwr-model' requires model_ref.",
            ),
            (
                CreateTaskRequest(type=TaskType.CONNECTOR),
                "Task type 'flwr-connector' requires connector_ref.",
            ),
        ]

        for request, detail in test_cases:
            context = Mock(spec=grpc.ServicerContext)
            context.abort.side_effect = grpc.RpcError()

            with self.subTest(task_type=request.type):
                with (
                    patch(
                        "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
                        return_value=Mock(run_id=123, type=TaskType.SERVER_APP),
                    ),
                    self.assertRaises(grpc.RpcError),
                ):
                    self.servicer.CreateTask(request, context)

                context.abort.assert_called_once_with(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    detail,
                )
                self.state.create_task.assert_not_called()

    def test_create_task_aborts_if_state_creation_fails(self) -> None:
        """CreateTask should surface task creation failures as INTERNAL."""
        # Prepare
        self.state.create_task.return_value = None
        context = Mock(spec=grpc.ServicerContext)
        context.abort.side_effect = grpc.RpcError()
        request = CreateTaskRequest(
            type=TaskType.MODEL,
            model_ref="models/abc",
        )

        # Execute
        with (
            patch(
                "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
                return_value=Mock(run_id=123, type=TaskType.SERVER_APP),
            ),
            self.assertRaises(grpc.RpcError),
        ):
            self.servicer.CreateTask(request, context)

        # Assert
        context.abort.assert_called_once_with(
            grpc.StatusCode.INTERNAL,
            "Failed to create task",
        )

    def test_create_task_aborts_if_requesting_task_type_is_not_allowed(self) -> None:
        """CreateTask should reject task creation requests from non-app task types."""
        # Prepare
        disallowed_requesting_task_types = (
            set(TaskType) - TASK_TYPES_ALLOWED_TO_CREATE_TASKS
        ) | {"unknown"}

        for requesting_task_type in disallowed_requesting_task_types:
            context = Mock(spec=grpc.ServicerContext)
            context.abort.side_effect = grpc.RpcError()
            self.state.create_task.reset_mock()

            with self.subTest(requesting_task_type=requesting_task_type):
                # Execute
                with (
                    patch(
                        "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
                        return_value=Mock(run_id=123, type=requesting_task_type),
                    ),
                    self.assertRaises(grpc.RpcError),
                ):
                    self.servicer.CreateTask(
                        CreateTaskRequest(type=TaskType.MODEL, model_ref="model"),
                        context,
                    )

                # Assert
                context.abort.assert_called_once_with(
                    grpc.StatusCode.PERMISSION_DENIED,
                    f"Task type '{requesting_task_type}' is not allowed to "
                    "create tasks.",
                )
                self.state.create_task.assert_not_called()

    def test_push_logs_merges_logs_and_stores_them(self) -> None:
        """PushLogs should concatenate fragments and store them via state."""
        # Execute
        with patch(
            "flwr.supercore.servicers.appio_servicer.get_authenticated_task",
            return_value=Mock(task_id=123),
        ):
            response = self.servicer.PushLogs(
                PushLogsRequest(run_id=123, logs=["hello", " ", "world"]),
                Mock(),
            )

        # Assert
        self.state.add_task_log.assert_called_once_with(123, "hello world")
        self.assertIsInstance(response, PushLogsResponse)
