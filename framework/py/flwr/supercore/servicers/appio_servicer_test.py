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

from flwr.common.constant import Status
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    PullPendingTasksRequest,
    SendTaskHeartbeatRequest,
)
from flwr.proto.task_pb2 import Task, TaskStatus  # pylint: disable=E0611

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
