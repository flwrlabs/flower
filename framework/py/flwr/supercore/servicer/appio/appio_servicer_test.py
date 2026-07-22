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
from logging import ERROR
from unittest.mock import Mock, patch

import grpc

from flwr.common.constant import SUPERLINK_NODE_ID, Status
from flwr.common.serde import message_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    CreateTaskRequest,
    CreateTaskResponse,
    PullPendingTasksRequest,
    PullTaskMessageRequest,
    PushTaskEventsRequest,
    PushTaskEventsResponse,
    PushTaskMessageRequest,
    RecordTaskUsageRequest,
    SendTaskHeartbeatRequest,
)
from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.task_pb2 import (  # pylint: disable=E0611
    Task,
    TaskEvent,
    TaskStatus,
    TaskUsage,
)
from flwr.supercore.constant import TASK_TYPES_ALLOWED_TO_CREATE_TASKS, TaskType
from flwr.supercore.corestate.utils_test import create_task_message
from flwr.supercore.task_process.connector import registry as connector_registry

from .appio_servicer import AppIoServicer


class _TestAppIoServicer(AppIoServicer):
    """Concrete AppIoServicer for tests."""

    def __init__(self, state: Mock) -> None:
        self._state = state

    def state(self) -> Mock:
        """Return mocked CoreState."""
        return self._state


class TestAppIoServicer(unittest.TestCase):  # pylint: disable=R0904
    """Tests for shared AppIoServicer task RPCs."""

    def setUp(self) -> None:
        """Set up test fixture."""
        self.state = Mock()
        self.state.get_tasks.return_value = []
        self.servicer = _TestAppIoServicer(self.state)

    def _create_connector_task(
        self, connector_ref: str, context: grpc.ServicerContext | None = None
    ) -> CreateTaskResponse:
        """Create a connector task as an authenticated AgentApp task."""
        with patch(
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Mock(task_id=789, run_id=123, type=TaskType.AGENT_APP),
        ):
            return self.servicer.CreateTask(
                CreateTaskRequest(
                    type=TaskType.CONNECTOR,
                    connector_ref=connector_ref,
                ),
                context if context is not None else Mock(),
            )

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
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
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
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Mock(task_id=789, run_id=123, type=TaskType.SERVER_APP),
        ):
            response = self.servicer.CreateTask(request, Mock())

        # Assert
        self.state.create_task.assert_called_once_with(
            task_type=TaskType.MODEL,
            run_id=123,
            fab_hash=None,
            model_ref="models/abc",
            connector_ref=None,
            requesting_task_id=789,
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
                    "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
                    return_value=Mock(
                        task_id=789, run_id=123, type=requesting_task_type
                    ),
                ):
                    response = self.servicer.CreateTask(request, Mock())

                # Assert
                self.state.create_task.assert_called_once_with(
                    task_type=TaskType.MODEL,
                    run_id=123,
                    fab_hash=None,
                    model_ref="model",
                    connector_ref=None,
                    requesting_task_id=789,
                )
                self.assertEqual(response.task_id, 456)

    def test_create_task_allows_bound_oauth_connector(self) -> None:
        """CreateTask should allow an OAuth connector bound to the run."""
        self.state.get_run_connector_refs.return_value = ["notion"]
        self.state.create_task.return_value = 456

        with patch.object(
            connector_registry,
            "OAUTH_CONNECTOR_PROVIDERS",
            (Mock(connector_ref="notion"),),
        ):
            response = self._create_connector_task("notion")

        self.state.get_run_connector_refs.assert_called_once_with(run_id=123)
        self.assertEqual(
            self.state.create_task.call_args.kwargs["connector_ref"], "notion"
        )
        self.assertEqual(response.task_id, 456)

    def test_create_task_rejects_unbound_oauth_connector(self) -> None:
        """CreateTask should reject OAuth credentials unavailable to the run."""
        self.state.get_run_connector_refs.return_value = []
        context = Mock(spec=grpc.ServicerContext)
        context.abort.side_effect = grpc.RpcError()

        with (
            patch.object(
                connector_registry,
                "OAUTH_CONNECTOR_PROVIDERS",
                (Mock(connector_ref="notion"),),
            ),
            self.assertRaises(grpc.RpcError),
        ):
            self._create_connector_task("notion", context)

        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED,
            "Connector is not available to this run.",
        )
        self.state.create_task.assert_not_called()

    def test_create_task_rejects_unknown_connector(self) -> None:
        """CreateTask should reject names absent from the connector registry."""
        context = Mock(spec=grpc.ServicerContext)
        context.abort.side_effect = grpc.RpcError()

        with self.assertRaises(grpc.RpcError):
            self._create_connector_task("unknown", context)

        context.abort.assert_called_once_with(
            grpc.StatusCode.NOT_FOUND,
            "Unsupported OAuth connector 'unknown'.",
        )
        self.state.get_run_connector_refs.assert_not_called()
        self.state.create_task.assert_not_called()

    def test_create_task_preserves_builtin_connector_access(self) -> None:
        """CreateTask should keep credential-free built-in connectors available."""
        self.state.create_task.return_value = 456

        response = self._create_connector_task("web_search")

        self.state.get_run_connector_refs.assert_not_called()
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
                "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
                return_value=Mock(task_id=789, run_id=123, type=TaskType.SERVER_APP),
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
            requesting_task_id=789,
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
                        "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
                        return_value=Mock(
                            task_id=789, run_id=123, type=TaskType.SERVER_APP
                        ),
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
                "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
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

    def test_push_task_message_stores_message_for_authenticated_task(self) -> None:
        """PushTaskMessage should store metadata matching the authenticated task."""
        # Prepare
        self.state.store_task_message.return_value = True
        message = create_task_message(
            run_id=789,
            src_task_id=123,
            dst_task_id=456,
        )
        request = PushTaskMessageRequest(message=message_to_proto(message))

        # Execute
        with patch(
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Task(task_id=123, run_id=789),
        ):
            response = self.servicer.PushTaskMessage(request, Mock())

        # Assert
        self.state.store_task_message.assert_called_once()
        stored_message = self.state.store_task_message.call_args.args[0]
        self.assertEqual(response.message_id, stored_message.metadata.message_id)
        self.assertEqual(stored_message.metadata.run_id, 789)
        self.assertEqual(stored_message.metadata.src_node_id, SUPERLINK_NODE_ID)
        self.assertEqual(stored_message.metadata.dst_node_id, SUPERLINK_NODE_ID)
        self.assertEqual(stored_message.metadata.src_task_id, 123)
        self.assertEqual(stored_message.metadata.dst_task_id, 456)

    def test_push_task_message_aborts_when_state_rejects_message(self) -> None:
        """PushTaskMessage should abort when CoreState cannot store the message."""
        # Prepare
        self.state.store_task_message.return_value = False
        message = create_task_message(run_id=789, src_task_id=123, dst_task_id=456)
        request = PushTaskMessageRequest(message=message_to_proto(message))
        context = Mock(spec=grpc.ServicerContext)
        context.abort.side_effect = grpc.RpcError()

        # Execute
        with (
            patch(
                "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
                return_value=Task(task_id=123, run_id=789),
            ),
            self.assertRaises(grpc.RpcError),
        ):
            self.servicer.PushTaskMessage(request, context)

        # Assert
        context.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Task message could not be stored.",
        )

    def test_push_task_events_derives_authenticated_task_identity(self) -> None:
        """PushTaskEvents should derive run and task IDs from task auth."""
        # Prepare
        self.state.store_task_events.return_value = True
        request = PushTaskEventsRequest(
            events=[
                TaskEvent(
                    id=999,
                    timestamp="client-ts",
                    run_id=111,
                    task_id=222,
                    event=" response.created ",
                    data='{"payload":"preserved"}',
                )
            ]
        )

        # Execute
        with patch(
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Task(task_id=123, run_id=789),
        ):
            response = self.servicer.PushTaskEvents(request, Mock())

        # Assert
        self.state.store_task_events.assert_called_once()
        stored_events = self.state.store_task_events.call_args.args[0]
        self.assertIsInstance(response, PushTaskEventsResponse)
        self.assertEqual(len(stored_events), 1)
        self.assertIs(stored_events[0], request.events[0])
        self.assertEqual(stored_events[0].id, 999)
        self.assertEqual(stored_events[0].timestamp, "client-ts")
        self.assertEqual(stored_events[0].run_id, 789)
        self.assertEqual(stored_events[0].task_id, 123)
        self.assertEqual(stored_events[0].event, " response.created ")
        self.assertEqual(stored_events[0].data, '{"payload":"preserved"}')

    def test_push_task_events_logs_when_state_rejects_events(self) -> None:
        """PushTaskEvents should log when CoreState cannot store events."""
        # Prepare
        self.state.store_task_events.return_value = False
        request = PushTaskEventsRequest(
            events=[
                TaskEvent(event="response.created", data="{"),
            ]
        )
        context = Mock(spec=grpc.ServicerContext)

        # Execute
        with (
            patch(
                "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
                return_value=Task(task_id=123, run_id=789),
            ),
            patch("flwr.supercore.servicer.appio.appio_servicer.log") as log_mock,
        ):
            response = self.servicer.PushTaskEvents(request, context)

        # Assert
        self.assertIsInstance(response, PushTaskEventsResponse)
        context.abort.assert_not_called()
        log_mock.assert_any_call(
            ERROR,
            "Task events could not be stored for task %d of run %d.",
            123,
            789,
        )

    def test_record_task_usage_stores_usage_for_authenticated_metered_task(
        self,
    ) -> None:
        """RecordTaskUsage should store usage for model and connector tasks."""
        usage = TaskUsage(
            usage_type="model_inference",
            provider="openai/gpt-test",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
        )
        request = RecordTaskUsageRequest(task_usage=usage)

        with patch(
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Task(task_id=123, run_id=789, type=TaskType.MODEL),
        ):
            response = self.servicer.RecordTaskUsage(request, Mock())

        self.state.add_task_usage.assert_called_once_with(123, usage)
        self.assertIsNotNone(response)

    def test_pull_task_message_uses_authenticated_task_destination(self) -> None:
        """PullTaskMessage should query messages for the authenticated task."""
        # Prepare
        message = create_task_message(
            src_task_id=123,
            dst_task_id=321,
            run_id=789,
        )
        message.metadata.__dict__["_message_id"] = "message-id"
        self.state.get_task_message.return_value = [message]

        # Execute
        with patch(
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Task(task_id=321, run_id=789),
        ):
            response = self.servicer.PullTaskMessage(
                PullTaskMessageRequest(limit=5),
                Mock(),
            )

        # Assert
        self.state.get_task_message.assert_called_once_with(
            dst_task_ids=[321],
            limit=5,
            order_by="created_at",
        )
        self.assertEqual(len(response.messages), 1)
        self.assertEqual(response.messages[0].metadata.src_task_id, 123)
        self.assertEqual(response.messages[0].metadata.dst_task_id, 321)
        self.assertEqual(response.messages[0].metadata.src_node_id, SUPERLINK_NODE_ID)
        self.assertEqual(response.messages[0].metadata.dst_node_id, SUPERLINK_NODE_ID)

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
                        "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
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
            "flwr.supercore.servicer.appio.appio_servicer.get_authenticated_task",
            return_value=Mock(task_id=123),
        ):
            response = self.servicer.PushLogs(
                PushLogsRequest(run_id=123, logs=["hello", " ", "world"]),
                Mock(),
            )

        # Assert
        self.state.add_task_log.assert_called_once_with(123, "hello world")
        self.assertIsInstance(response, PushLogsResponse)
