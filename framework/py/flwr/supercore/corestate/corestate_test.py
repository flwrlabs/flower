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
"""Tests all CoreState implementations have to conform to."""


# pylint: disable=too-many-lines
import unittest
from contextlib import ExitStack
from datetime import datetime, timedelta
from typing import Any, cast
from unittest.mock import patch

from parameterized import parameterized

from flwr.common.constant import (
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    SUPERLINK_NODE_ID,
    Status,
    SubStatus,
)
from flwr.proto.control_pb2 import Automation  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.task_pb2 import (  # pylint: disable=E0611
    TaskEvent,
    TaskStatus,
    TaskUsage,
)
from flwr.supercore.constant import AutomationStatus, TaskType
from flwr.supercore.date import now
from flwr.supercore.typing import ConnectorRecord

from . import CoreState
from .utils_test import create_task_message


class StateTest(unittest.TestCase):  # pylint: disable=R0904
    """Test all CoreState implementations."""

    # This is to True in each child class
    __test__ = False

    def state_factory(self) -> CoreState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def task_run_id(self, _state: CoreState) -> int:
        """Return the run ID to use in task-related tests.

        Subclasses can override this hook when task creation requires an existing run
        record instead of an arbitrary placeholder ID.
        """
        return 42

    def other_task_run_id(self, _state: CoreState) -> int:
        """Return a second run ID for task tests that need multiple runs.

        Subclasses can override this hook when task creation requires existing run
        records instead of arbitrary placeholder IDs.
        """
        return 123

    def _patch_task_log_datetime_now(self, *timestamps: datetime) -> ExitStack:
        """Patch the shared datetime source used for task-log timestamps."""
        stack = ExitStack()
        mock_datetime = stack.enter_context(
            patch("flwr.supercore.date.datetime.datetime")
        )
        mock_datetime.now.side_effect = timestamps
        return stack

    def test_connector_upsert_get_and_delete(self) -> None:
        """A connector can be created, updated, retrieved, and deleted."""
        state = self.state_factory()

        self.assertTrue(
            state.upsert_connector(
                flwr_aid="account-a",
                connector_ref="calendar",
                credentials_json='{"token":"first"}',
                config_json='{"calendar":"primary"}',
            )
        )
        self.assertEqual(
            state.get_connector(flwr_aid="account-a", connector_ref="calendar"),
            ConnectorRecord(
                flwr_aid="account-a",
                connector_ref="calendar",
                credentials_json='{"token":"first"}',
                config_json='{"calendar":"primary"}',
            ),
        )
        self.assertTrue(
            state.upsert_connector(
                flwr_aid="account-a",
                connector_ref="calendar",
                credentials_json='{"token":"updated"}',
                config_json='{"calendar":"work"}',
            )
        )
        updated = state.get_connector(flwr_aid="account-a", connector_ref="calendar")
        assert updated is not None
        self.assertEqual(updated.credentials_json, '{"token":"updated"}')
        self.assertEqual(updated.config_json, '{"calendar":"work"}')

        self.assertTrue(
            state.delete_connector(flwr_aid="account-a", connector_ref="calendar")
        )
        self.assertIsNone(
            state.get_connector(flwr_aid="account-a", connector_ref="calendar")
        )

    def test_connector_oauth_session_lifecycle(self) -> None:
        """An OAuth session can be created, retrieved, and completed once."""
        state = self.state_factory()
        expires_at = now() + timedelta(minutes=10)

        session = state.create_connector_oauth_session(
            oauth_session_id="session-1",
            flwr_aid="account-a",
            connector_ref="calendar",
            state="oauth-state",
            redirect_uri="https://example.test/callback",
            pkce_verifier=None,
            expires_at=expires_at,
        )
        assert session is not None
        self.assertEqual(session.expires_at, expires_at.isoformat())
        self.assertIsNone(session.completed_at)
        self.assertEqual(
            state.get_connector_oauth_session(
                oauth_session_id="session-1", flwr_aid="account-a"
            ),
            session,
        )
        self.assertTrue(
            state.complete_connector_oauth_session(
                oauth_session_id="session-1", flwr_aid="account-a"
            )
        )
        completed = state.get_connector_oauth_session(
            oauth_session_id="session-1", flwr_aid="account-a"
        )
        assert completed is not None
        self.assertIsNotNone(completed.completed_at)
        self.assertFalse(
            state.complete_connector_oauth_session(
                oauth_session_id="session-1", flwr_aid="account-a"
            )
        )

    def store_automation(  # pylint: disable=too-many-arguments
        self,
        state: CoreState,
        *,
        series_id: int,
        federation_id: str = "@me/fed-a",
        flwr_aid: str = "aid-a",
        next_run_at: str | None = None,
        fixed_interval: int | None = None,
        max_runs: int | None = 1,
    ) -> Automation:
        """Store a minimal automation."""
        return state.store_automation(
            federation_id=federation_id,
            flwr_aid=flwr_aid,
            fab_id=None,
            fab_version=None,
            fab_hash=None,
            override_config={},
            federation_config=None,
            primary_task_type=TaskType.SERVER_APP,
            series_id=series_id,
            next_run_at=next_run_at or now().isoformat(),
            fixed_interval=fixed_interval,
            max_runs=max_runs,
        )

    def test_preregister_object_tree(self) -> None:
        """Preregistering an object tree returns its missing objects."""
        state = self.state_factory()
        object_id = "a" * 64
        object_tree = ObjectTree(object_id=object_id)
        run_id = self.task_run_id(state)

        missing_objects = state.preregister_object_tree(
            object_tree, state.start_session(run_id)
        )
        replacement_missing_objects = state.preregister_object_tree(
            object_tree, state.start_session(run_id)
        )

        self.assertEqual(missing_objects, [object_id])
        self.assertEqual(replacement_missing_objects, [object_id])

    def test_store_run_in_series_creates_id(self) -> None:
        """Storing a run in a run series should create a nonzero ID."""
        state = self.state_factory()

        series_id = state.store_run_in_series(
            run_id=123, federation_id="@me/fed-a", series_id=None
        )

        self.assertIsNotNone(series_id)
        assert series_id is not None
        self.assertGreater(series_id, 0)

    def test_store_run_in_series_returns_none_for_unknown_id(self) -> None:
        """Unknown caller-provided run series IDs return None."""
        state = self.state_factory()

        with self.assertLogs("flwr", level="ERROR") as logs:
            series_id = state.store_run_in_series(
                run_id=123,
                federation_id="@me/fed-a",
                series_id=123,
            )

        self.assertIsNone(series_id)
        self.assertIn("Run series 123 not found", logs.output[0])

    def test_store_run_in_series_returns_none_for_duplicate_run_id(self) -> None:
        """Storing the same run ID twice should return None."""
        state = self.state_factory()
        series_id = state.store_run_in_series(
            run_id=123, federation_id="@me/fed-a", series_id=None
        )
        assert series_id is not None

        stored = state.store_run_in_series(
            run_id=123,
            federation_id="@me/fed-a",
            series_id=series_id,
        )

        self.assertIsNone(stored)

    def test_get_run_series_filters_by_series_ids_and_federation_ids(self) -> None:
        """RunSeries lookup should filter by series IDs and federation IDs."""
        state = self.state_factory()
        series_id_a = state.store_run_in_series(
            run_id=123, federation_id="@me/fed-a", series_id=None
        )
        series_id_b = state.store_run_in_series(
            run_id=456, federation_id="@me/fed-b", series_id=None
        )
        series_id_c = state.store_run_in_series(
            run_id=789, federation_id="@me/fed-a", series_id=None
        )
        assert series_id_a is not None
        assert series_id_b is not None
        assert series_id_c is not None

        fed_a_series = state.get_run_series(federation_ids=["@me/fed-a"])
        self.assertSetEqual(
            {entry.series_id for entry in fed_a_series},
            {series_id_a, series_id_c},
        )

        id_filtered_series = state.get_run_series(series_ids=[series_id_b])
        self.assertEqual(
            [entry.series_id for entry in id_filtered_series],
            [series_id_b],
        )

        combined_series = state.get_run_series(
            series_ids=[series_id_a, series_id_b],
            federation_ids=["@me/fed-a"],
        )
        self.assertEqual([entry.series_id for entry in combined_series], [series_id_a])

        self.assertEqual(state.get_run_series(series_ids=[]), [])
        self.assertEqual(state.get_run_series(federation_ids=[]), [])

    def test_store_list_and_stop_automation(self) -> None:
        """Automation storage should support list, due filtering, and stop."""
        state = self.state_factory()
        current = now()
        due_at = (current - timedelta(seconds=60)).isoformat()

        due = self.store_automation(
            state,
            series_id=1,
            next_run_at=due_at,
            fixed_interval=60,
        )
        future = self.store_automation(
            state,
            series_id=2,
            next_run_at=(current + timedelta(seconds=60)).isoformat(),
        )
        _ = self.store_automation(
            state,
            series_id=3,
            federation_id="@me/fed-b",
            next_run_at=(current - timedelta(seconds=30)).isoformat(),
        )

        self.assertEqual(due.next_run_at, due_at)

        listed = state.list_automations(federation="@me/fed-a", order_by="updated_at")
        self.assertSetEqual(
            {automation.automation_id for automation in listed},
            {due.automation_id, future.automation_id},
        )

        due_list = state.list_automations(
            federation="@me/fed-a",
            statuses=["active"],
            due_before=current,
            order_by="next_run_at",
            limit=10,
        )
        self.assertEqual(
            [automation.automation_id for automation in due_list], [due.automation_id]
        )
        self.assertEqual(due_list[0].remaining_runs, 1)

        self.assertTrue(state.stop_automation(due.automation_id))
        self.assertFalse(state.stop_automation(due.automation_id))

        stopped = state.list_automations(
            federation="@me/fed-a",
            statuses=[AutomationStatus.STOPPED],
            order_by="updated_at",
        )
        self.assertEqual(
            [automation.automation_id for automation in stopped], [due.automation_id]
        )
        self.assertEqual(stopped[0].next_run_at, due_at)

    def test_store_automation_preserves_series_id_without_validation(self) -> None:
        """Automation storage should preserve caller-provided series IDs."""
        state = self.state_factory()
        series_id = 123

        automation = self.store_automation(
            state, federation_id="@me/fed-b", series_id=series_id
        )

        self.assertEqual(automation.series_id, series_id)
        self.assertEqual(automation.federation, "@me/fed-b")

    def test_create_and_get_task(self) -> None:
        """Test creating and retrieving a task."""
        state = self.state_factory()
        run_id = self.task_run_id(state)

        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=run_id,
            fab_hash=None,
            model_ref="model://test",
            connector_ref=None,
        )
        assert task_id is not None
        tasks = state.get_tasks(task_ids=[task_id])

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task.task_id, task_id)
        self.assertEqual(task.type, TaskType.MODEL)
        self.assertEqual(task.run_id, run_id)
        self.assertEqual(
            task.status,
            TaskStatus(status=Status.PENDING, sub_status="", details=""),
        )
        self.assertEqual(task.model_ref, "model://test")
        self.assertFalse(task.HasField("fab_hash"))
        self.assertFalse(task.HasField("connector_ref"))
        self.assertTrue(task.pending_at)
        self.assertEqual(task.starting_at, "")
        self.assertEqual(task.running_at, "")
        self.assertEqual(task.finished_at, "")

    def test_create_task_rejects_finished_requesting_task(self) -> None:
        """Task creation should fail if the requesting task is already finished."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        requesting_task_id = state.create_task(
            task_type=TaskType.SERVER_APP,
            run_id=run_id,
        )
        assert requesting_task_id is not None
        self.assertTrue(state.finish_task(requesting_task_id, SubStatus.STOPPED, ""))

        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=run_id,
            model_ref="model://test",
            requesting_task_id=requesting_task_id,
        )

        self.assertIsNone(task_id)

    def test_get_tasks_missing_returns_empty(self) -> None:
        """Missing tasks should return an empty sequence."""
        state = self.state_factory()
        self.assertEqual(state.get_tasks(task_ids=[123]), [])

    def test_get_tasks_run_id_matches(self) -> None:
        """Run ID filters should match only tasks from the requested runs."""
        state = self.state_factory()
        run_id_1 = self.task_run_id(state)
        run_id_2 = self.other_task_run_id(state)
        task_id_1 = state.create_task(task_type=TaskType.MODEL, run_id=run_id_1)
        task_id_2 = state.create_task(task_type=TaskType.MODEL, run_id=run_id_2)
        task_id_3 = state.create_task(task_type=TaskType.MODEL, run_id=run_id_1)
        assert task_id_1 and task_id_2 and task_id_3

        tasks = state.get_tasks(run_ids=[run_id_1])
        task_ids = {task.task_id for task in tasks}

        self.assertTrue({task_id_1, task_id_3}.issubset(task_ids))
        self.assertNotIn(task_id_2, task_ids)
        self.assertTrue(all(task.run_id == run_id_1 for task in tasks))

    def test_get_tasks_single_status_matches(self) -> None:
        """A single-item status sequence should match pending tasks."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        assert task_id

        tasks = state.get_tasks(statuses=[Status.PENDING])
        task_ids = {task.task_id for task in tasks}

        self.assertIn(task_id, task_ids)
        for task in tasks:
            self.assertEqual(task.status.status, Status.PENDING)

    def test_get_tasks_negative_limit_raises(self) -> None:
        """Negative limits should be rejected consistently."""
        state = self.state_factory()

        with self.assertRaises(AssertionError):
            _ = state.get_tasks(limit=-1)

    def test_get_tasks_invalid_order_by_raises(self) -> None:
        """Unsupported order_by values should be rejected consistently."""
        state = self.state_factory()

        with self.assertRaises(AssertionError):
            _ = state.get_tasks(order_by=cast(Any, "foo"))

    def test_get_task_message_invalid_order_by_raises(self) -> None:
        """Unsupported task-message order_by values should be rejected."""
        state = self.state_factory()

        with self.assertRaises(AssertionError):
            _ = state.get_task_message(order_by=cast(Any, "foo"))

    def test_get_task_returns_copy(self) -> None:
        """Retrieved task should be a defensive copy."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        task_id = state.create_task(
            task_type=TaskType.SERVER_APP,
            run_id=run_id,
            fab_hash="fab-hash",
            model_ref=None,
            connector_ref=None,
        )
        assert task_id is not None

        tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        task.fab_hash = "mutated"

        reloaded_tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(reloaded_tasks), 1)
        reloaded = reloaded_tasks[0]
        self.assertEqual(reloaded.fab_hash, "fab-hash")

    def test_add_and_get_task_usage(self) -> None:
        """Task usage should round-trip and filter by task ID."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=self.task_run_id(state),
        )
        assert task_id is not None

        state.add_task_usage(
            task_id,
            TaskUsage(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                usage_type="token",
            ),
        )
        state.add_task_usage(task_id, TaskUsage(input_tokens=999, usage_type="token"))

        usages = state.get_task_usage(task_ids=[task_id])

        self.assertEqual(len(usages), 2)
        usage = usages[0]
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.usage_type, "token")
        self.assertEqual(usages[1].input_tokens, 999)

    def test_add_and_get_task_log(self) -> None:
        """Adding and retrieving task logs should preserve concatenation order."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=self.task_run_id(state),
        )
        assert task_id is not None
        log_entry_1 = "Log entry 1"
        log_entry_2 = "Log entry 2"
        fixed_now = now()
        timestamp = (fixed_now - timedelta(microseconds=1)).timestamp()

        with self._patch_task_log_datetime_now(
            fixed_now,
            fixed_now + timedelta(microseconds=1),
        ):
            state.add_task_log(task_id, log_entry_1)
            state.add_task_log(task_id, log_entry_2)

        # Reading from before the first log should return both entries and the
        # timestamp of the newest returned entry.
        retrieved_logs, latest = state.get_task_log(task_id, after_timestamp=timestamp)

        assert latest > timestamp
        assert log_entry_1 + log_entry_2 == retrieved_logs

    def test_get_task_log_after_timestamp(self) -> None:
        """Retrieving task logs after a specific timestamp should filter old logs."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=self.task_run_id(state),
        )
        assert task_id is not None
        log_entry_1 = "Log entry 1"
        log_entry_2 = "Log entry 2"
        fixed_now = now()
        timestamp = (fixed_now + timedelta(microseconds=1)).timestamp()

        with self._patch_task_log_datetime_now(
            fixed_now,
            fixed_now + timedelta(microseconds=2),
        ):
            state.add_task_log(task_id, log_entry_1)
            state.add_task_log(task_id, log_entry_2)

        # A timestamp between the two entries should filter out only the older
        # log and advance the checkpoint to the returned entry.
        retrieved_logs, latest = state.get_task_log(task_id, after_timestamp=timestamp)

        assert latest > timestamp
        assert log_entry_1 not in retrieved_logs
        assert log_entry_2 == retrieved_logs

    def test_get_task_log_after_timestamp_no_logs(self) -> None:
        """Retrieving task logs after the last entry should return an empty result."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=self.task_run_id(state),
        )
        assert task_id is not None
        fixed_now = now()
        with self._patch_task_log_datetime_now(fixed_now):
            state.add_task_log(task_id, "Log entry")
        timestamp = (fixed_now + timedelta(microseconds=1)).timestamp()

        # Polling after the latest known entry should return no logs and no new
        # checkpoint.
        retrieved_logs, latest = state.get_task_log(task_id, after_timestamp=timestamp)

        assert latest == 0
        assert retrieved_logs == ""

    def test_get_task_log_does_not_repeat_logs_at_checkpoint_timestamp(self) -> None:
        """Polling with the last returned timestamp should not repeat old logs."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=self.task_run_id(state),
        )
        assert task_id is not None
        fixed_now = now()

        with self._patch_task_log_datetime_now(fixed_now):
            state.add_task_log(task_id, "Log entry 1")
        retrieved_logs, latest = state.get_task_log(task_id, after_timestamp=None)

        assert retrieved_logs == "Log entry 1"
        assert latest == fixed_now.timestamp()

        # Reusing the returned timestamp as the next checkpoint must not replay
        # the log that produced that checkpoint.
        next_logs, next_latest = state.get_task_log(task_id, after_timestamp=latest)

        assert next_logs == ""
        assert next_latest == 0

    def test_claim_task_transitions_pending_to_starting(self) -> None:
        """Claiming a task should create a token and move it to starting."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL, run_id=self.task_run_id(state)
        )
        assert task_id is not None

        # Claim should persist token ownership and move the task to STARTING.
        token = state.claim_task(task_id)

        self.assertIsNotNone(token)
        assert token is not None
        assert (task := state.get_task_by_token(token))
        self.assertEqual(task.task_id, task_id)
        tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].status.status, Status.STARTING)
        self.assertTrue(tasks[0].starting_at)
        self.assertEqual(tasks[0].running_at, "")
        self.assertEqual(tasks[0].finished_at, "")

    def test_claim_task_rejects_missing_claimed_and_non_pending(self) -> None:
        """Only existing pending unclaimed tasks should be claimable."""
        state = self.state_factory()
        run_id = self.task_run_id(state)

        # Missing tasks cannot be claimed.
        self.assertIsNone(state.claim_task(61016))

        claimed_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        finished_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        assert claimed_task_id is not None and finished_task_id is not None

        # Claiming is single-owner and cannot be repeated.
        self.assertIsNotNone(state.claim_task(claimed_task_id))
        self.assertIsNone(state.claim_task(claimed_task_id))

        # Finished tasks are not claimable.
        self.assertTrue(state.finish_task(finished_task_id, SubStatus.FAILED, "done"))
        self.assertIsNone(state.claim_task(finished_task_id))

    def test_activate_task_transitions_starting_to_running(self) -> None:
        """Only starting tasks should transition to running."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL, run_id=self.task_run_id(state)
        )
        assert task_id is not None

        # Task does not exist, so it cannot be activated.
        self.assertFalse(state.activate_task(61016))
        # Task exists but is pending, so it must be claimed before activation.
        self.assertFalse(state.activate_task(task_id))
        # Claiming the task returns a token.
        self.assertIsNotNone(state.claim_task(task_id))
        # The task is in starting status, so it can be activated.
        self.assertTrue(state.activate_task(task_id))
        # The task is already in running status, so it cannot be activated again.
        self.assertFalse(state.activate_task(task_id))

        tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].status.status, Status.RUNNING)
        self.assertTrue(tasks[0].running_at)
        self.assertEqual(tasks[0].finished_at, "")

    @parameterized.expand(  # type: ignore
        [
            (SubStatus.FAILED, False),
            (SubStatus.STOPPED, False),
            (SubStatus.COMPLETED, True),
        ]
    )
    def test_finish_task_transitions_unfinished_task_to_finished(
        self, sub_status: str, requires_running: bool
    ) -> None:
        """Finishing a task should store the terminal status details."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.MODEL, run_id=self.task_run_id(state)
        )
        assert task_id is not None

        # Task does not exist.
        self.assertFalse(state.finish_task(61016, SubStatus.FAILED, "missing"))

        if requires_running:
            # FINISHED:COMPLETED is only valid once the task is RUNNING.
            self.assertFalse(state.finish_task(task_id, sub_status, "boom"))
            self.assertIsNotNone(state.claim_task(task_id))
            self.assertTrue(state.activate_task(task_id))

        # Valid unfinished task transition should succeed.
        self.assertTrue(state.finish_task(task_id, sub_status, "boom"))
        # Task is already finished, so it cannot be finished again.
        self.assertFalse(state.finish_task(task_id, SubStatus.FAILED, "again"))
        # Finished tasks cannot be claimed.
        self.assertIsNone(state.claim_task(task_id))

        tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(
            task.status,
            TaskStatus(
                status=Status.FINISHED,
                sub_status=sub_status,
                details="boom",
            ),
        )
        self.assertTrue(task.finished_at)

    def test_task_heartbeat_extends_token_expiration(self) -> None:
        """Task heartbeat should keep a claimed task token valid."""
        state = self.state_factory()
        fixed_now = now()
        run_id = self.task_run_id(state)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
            assert task_id is not None
            token = state.claim_task(task_id)
            assert token is not None

            # Heartbeat extends only existing claimed task leases.
            self.assertFalse(state.acknowledge_task_heartbeat(61016))
            self.assertTrue(state.acknowledge_task_heartbeat(task_id))

            # The heartbeat extension should keep the token valid past its
            # initial claim deadline.
            mock_dt.now.return_value = fixed_now + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
            )
            assert (task := state.get_task_by_token(token))
            self.assertEqual(task.task_id, task_id)

            # Once the extended deadline passes, the token no longer resolves.
            mock_dt.now.return_value = fixed_now + timedelta(
                seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL + 1
            )
            self.assertIsNone(state.get_task_by_token(token))
            self.assertFalse(state.acknowledge_task_heartbeat(task_id))

    def test_expired_starting_task_token_revives_task_to_pending(self) -> None:
        """Expired STARTING task claims should make tasks pending again."""
        # Prepare: create and claim a model task.
        state = self.state_factory()
        fixed_now = now()
        run_id = self.task_run_id(state)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
            assert task_id is not None
            pending_at = state.get_tasks(task_ids=[task_id])[0].pending_at

            token = state.claim_task(task_id)
            assert token is not None

            # Execute: advance past claim expiry and trigger cleanup.
            mock_dt.now.return_value = fixed_now + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
            )
            self.assertIsNone(state.get_task_by_token(token))
            self.assertFalse(state.acknowledge_task_heartbeat(task_id))

        # Assert: task is pending again and can be claimed with a fresh token.
        tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(
            tasks[0].status,
            TaskStatus(status=Status.PENDING, sub_status="", details=""),
        )
        self.assertEqual(tasks[0].pending_at, pending_at)
        self.assertEqual(tasks[0].starting_at, "")
        self.assertEqual(tasks[0].running_at, "")
        self.assertEqual(tasks[0].finished_at, "")
        self.assertIsNone(state.get_task_by_token(token))
        new_token = state.claim_task(task_id)
        self.assertNotEqual(new_token, token)
        assert new_token is not None
        new_task = state.get_task_by_token(new_token)
        self.assertIsNotNone(new_task)
        assert new_task is not None
        self.assertEqual(new_task.task_id, task_id)

    def test_expired_running_task_token_transitions_task_to_finished_failed(
        self,
    ) -> None:
        """Expired RUNNING task claims should transition tasks to FINISHED:FAILED."""
        state = self.state_factory()
        fixed_now = now()
        active_until = fixed_now + timedelta(
            seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
        )
        run_id = self.task_run_id(state)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
            assert task_id is not None

            token = state.claim_task(task_id)
            assert token is not None
            self.assertTrue(state.activate_task(task_id))

            mock_dt.now.return_value = active_until + timedelta(seconds=1)
            self.assertIsNone(state.get_task_by_token(token))
            self.assertFalse(state.acknowledge_task_heartbeat(task_id))

        tasks = state.get_tasks(task_ids=[task_id])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(
            tasks[0].status,
            TaskStatus(
                status=Status.FINISHED,
                sub_status=SubStatus.FAILED,
                details="No heartbeat received from the task",
            ),
        )
        self.assertTrue(tasks[0].finished_at)
        self.assertEqual(datetime.fromisoformat(tasks[0].finished_at), active_until)

    def test_activate_task_extends_token_expiration(self) -> None:
        """Activating a task should give it the regular heartbeat grace period."""
        state = self.state_factory()
        fixed_now = now()
        run_id = self.task_run_id(state)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
            assert task_id is not None
            token = state.claim_task(task_id)
            assert token is not None
            self.assertTrue(state.activate_task(task_id))

            mock_dt.now.return_value = fixed_now + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
            )
            task = state.get_task_by_token(token)
            self.assertIsNotNone(task)
            assert task is not None
            self.assertEqual(task.task_id, task_id)

    def test_get_tasks_expires_stale_task_tokens(self) -> None:
        """Reading tasks should expire stale claimed task tokens first."""
        state = self.state_factory()
        fixed_now = now()
        run_id = self.task_run_id(state)

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
            assert task_id is not None
            assert (token := state.claim_task(task_id))

            mock_dt.now.return_value = fixed_now + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
            )
            tasks = state.get_tasks(task_ids=[task_id])

        self.assertIsNone(state.get_task_by_token(token))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(
            tasks[0].status,
            TaskStatus(status=Status.PENDING, sub_status="", details=""),
        )
        self.assertEqual(tasks[0].starting_at, "")
        self.assertEqual(tasks[0].finished_at, "")

    def test_expired_starting_task_token_does_not_call_expiry_hook(self) -> None:
        """Revived STARTING tasks should not be passed to expiry hooks."""
        state = self.state_factory()
        fixed_now = now()
        run_id = self.task_run_id(state)

        with patch.object(  # pylint: disable=protected-access
            state, "_on_task_tokens_expired"
        ) as on_expired:
            with patch("datetime.datetime") as mock_dt:
                mock_dt.now.return_value = fixed_now
                task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
                assert task_id is not None
                assert state.claim_task(task_id) is not None

                mock_dt.now.return_value = fixed_now + timedelta(
                    seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
                )
                state.get_tasks(task_ids=[task_id])

            on_expired.assert_not_called()

    def test_get_task_by_token_returns_none_for_unknown_token(self) -> None:
        """Unknown task tokens should not resolve to a task."""
        state = self.state_factory()

        self.assertIsNone(state.get_task_by_token("missing-token"))

    def test_store_and_get_task_message(self) -> None:
        """Task Messages should round-trip, filter, and be delivered once."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        src_task_id = state.create_task(task_type=TaskType.AGENT_APP, run_id=run_id)
        dst_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        other_dst_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        assert (
            src_task_id is not None
            and dst_task_id is not None
            and other_dst_task_id is not None
        )

        message = create_task_message(
            src_task_id=src_task_id,
            dst_task_id=dst_task_id,
            run_id=run_id,
        )
        expired = create_task_message(
            src_task_id,
            dst_task_id,
            run_id,
            created_at=now().timestamp() - 2.0,
            ttl=1.0,
        )
        other_destination = create_task_message(src_task_id, other_dst_task_id, run_id)

        self.assertTrue(state.store_task_message(message))
        self.assertFalse(state.store_task_message(expired))
        self.assertTrue(state.store_task_message(other_destination))
        pulled = state.get_task_message(dst_task_ids=[dst_task_id])
        pulled_again = state.get_task_message(dst_task_ids=[dst_task_id])
        pulled_other = state.get_task_message(dst_task_ids=[other_dst_task_id])
        pulled_other_again = state.get_task_message(dst_task_ids=[other_dst_task_id])

        self.assertEqual(len(pulled), 1)
        self.assertEqual(pulled_again, [])
        self.assertEqual(len(pulled_other), 1)
        self.assertEqual(pulled_other_again, [])
        pulled_message = pulled[0]
        pulled_other_message = pulled_other[0]
        self.assertEqual(
            pulled_message.metadata.message_id, message.metadata.message_id
        )
        self.assertEqual(
            pulled_other_message.metadata.message_id,
            other_destination.metadata.message_id,
        )
        self.assertEqual(pulled_message.metadata.run_id, run_id)
        self.assertEqual(pulled_message.metadata.src_node_id, SUPERLINK_NODE_ID)
        self.assertEqual(pulled_message.metadata.dst_node_id, SUPERLINK_NODE_ID)
        self.assertEqual(pulled_message.metadata.src_task_id, src_task_id)
        self.assertEqual(pulled_message.metadata.dst_task_id, dst_task_id)
        self.assertTrue(pulled_message.has_content())

    def test_store_task_message_validates_task_relationship(self) -> None:
        """Task Messages should only be stored for valid same-run destinations."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        other_run_id = self.other_task_run_id(state)
        src_task_id = state.create_task(task_type=TaskType.AGENT_APP, run_id=run_id)
        dst_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        other_run_task_id = state.create_task(
            task_type=TaskType.MODEL, run_id=other_run_id
        )
        finished_src_task_id = state.create_task(
            task_type=TaskType.AGENT_APP, run_id=run_id
        )
        finished_dst_task_id = state.create_task(
            task_type=TaskType.MODEL, run_id=run_id
        )
        assert (
            src_task_id is not None
            and dst_task_id is not None
            and other_run_task_id is not None
            and finished_src_task_id is not None
            and finished_dst_task_id is not None
        )
        assert state.finish_task(finished_src_task_id, SubStatus.FAILED, "done")
        assert state.finish_task(finished_dst_task_id, SubStatus.FAILED, "done")

        missing_task_id = (
            max(
                src_task_id,
                dst_task_id,
                other_run_task_id,
                finished_src_task_id,
                finished_dst_task_id,
            )
            + 1
        )
        while state.get_tasks(task_ids=[missing_task_id]):
            missing_task_id += 1
        invalid_messages = [
            create_task_message(missing_task_id, dst_task_id, run_id),
            create_task_message(src_task_id, missing_task_id, run_id),
            create_task_message(src_task_id, other_run_task_id, run_id),
            create_task_message(src_task_id, finished_dst_task_id, run_id),
            create_task_message(src_task_id, dst_task_id, 0),
            create_task_message(src_task_id, dst_task_id, other_run_id),
        ]
        finished_source_message = create_task_message(
            finished_src_task_id, dst_task_id, run_id
        )

        for message in invalid_messages:
            self.assertFalse(state.store_task_message(message))
        self.assertTrue(state.store_task_message(finished_source_message))

        pulled = state.get_task_message(dst_task_ids=[dst_task_id])

        self.assertEqual(len(pulled), 1)
        self.assertEqual(
            pulled[0].metadata.message_id,
            finished_source_message.metadata.message_id,
        )
        self.assertEqual(state.get_task_message(dst_task_ids=[other_run_task_id]), [])
        self.assertEqual(
            state.get_task_message(dst_task_ids=[finished_dst_task_id]), []
        )
        self.assertEqual(state.get_task_message(), [])

    def test_get_task_message_does_not_return_expired_messages(self) -> None:
        """Getting task Messages should not return expired Messages."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        src_task_id = state.create_task(task_type=TaskType.AGENT_APP, run_id=run_id)
        dst_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        assert src_task_id is not None and dst_task_id is not None

        msg_ttl = 60.0
        current = now()
        expired = create_task_message(
            src_task_id,
            dst_task_id,
            run_id,
            created_at=current.timestamp(),
            ttl=msg_ttl,
        )
        self.assertTrue(state.store_task_message(expired))

        future = current + timedelta(seconds=msg_ttl + 1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future
            self.assertEqual(state.get_task_message(dst_task_ids=[dst_task_id]), [])

    def test_get_task_message_limit(self) -> None:
        """Getting task Messages should respect the provided limit."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        src_task_id = state.create_task(task_type=TaskType.AGENT_APP, run_id=run_id)
        dst_task_id = state.create_task(task_type=TaskType.MODEL, run_id=run_id)
        assert src_task_id is not None and dst_task_id is not None

        current = now().timestamp()
        msg_1 = create_task_message(
            src_task_id, dst_task_id, run_id, created_at=current - 2.0
        )
        msg_2 = create_task_message(
            src_task_id, dst_task_id, run_id, created_at=current - 1.0
        )
        self.assertTrue(state.store_task_message(msg_2))
        self.assertTrue(state.store_task_message(msg_1))

        pulled = state.get_task_message(
            dst_task_ids=[dst_task_id], order_by="created_at", limit=1
        )
        pulled_next = state.get_task_message(
            dst_task_ids=[dst_task_id], order_by="created_at", limit=1
        )

        self.assertEqual(len(pulled), 1)
        self.assertEqual(len(pulled_next), 1)
        self.assertEqual(pulled[0].metadata.message_id, msg_1.metadata.message_id)
        self.assertEqual(pulled_next[0].metadata.message_id, msg_2.metadata.message_id)

    def test_store_and_get_task_events(self) -> None:
        """Task events should round-trip in assigned ID order."""
        # Prepare: Create one run with a task and two valid task events.
        state = self.state_factory()
        run_id = self.task_run_id(state)
        task_id = state.create_task(task_type=TaskType.AGENT_APP, run_id=run_id)
        assert task_id is not None
        event_1 = TaskEvent(
            run_id=run_id,
            task_id=task_id,
            event="response.created",
            data='{"type":"response.created"}',
        )
        event_2 = TaskEvent(
            run_id=run_id,
            task_id=task_id,
            event="response.output_text.delta",
            data='{"type":"response.output_text.delta","delta":"Hel"}',
        )

        # Execute: Store the events and read them through full and cursored fetches.
        self.assertFalse(state.store_task_events([]))
        self.assertTrue(state.store_task_events([event_1, event_2]))
        events = state.get_task_events(run_id=run_id, after_task_event_id=None)
        latest_id = events[-1].id
        after_first = state.get_task_events(
            run_id=run_id, after_task_event_id=events[0].id
        )
        no_new = state.get_task_events(run_id=run_id, after_task_event_id=latest_id)

        # Assert: Events keep assigned ID order and cursor filtering works.
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], TaskEvent)
        self.assertGreater(events[0].id, 0)
        self.assertGreater(events[1].id, events[0].id)
        self.assertTrue(events[0].timestamp)
        self.assertEqual(events[0].run_id, run_id)
        self.assertEqual(events[1].run_id, run_id)
        self.assertEqual(
            (events[0].task_id, events[0].event, events[0].data),
            (task_id, event_1.event, event_1.data),
        )
        self.assertEqual(
            (events[1].task_id, events[1].event, events[1].data),
            (task_id, event_2.event, event_2.data),
        )
        self.assertEqual(latest_id, events[1].id)
        self.assertEqual(after_first, [events[1]])
        self.assertEqual(no_new, [])

    @parameterized.expand(  # type: ignore
        [
            ("malformed", "{"),
            ("array", "[]"),
            ("string", '"value"'),
            ("non_finite", '{"value": NaN}'),
        ]
    )
    def test_store_task_events_requires_json_object_payload(
        self, _name: str, data: str
    ) -> None:
        """Task event data should be a JSON object string."""
        # Prepare: Create one valid event followed by an invalid payload variant.
        state = self.state_factory()
        run_id = self.task_run_id(state)
        task_id = state.create_task(task_type=TaskType.AGENT_APP, run_id=run_id)
        assert task_id is not None

        # Execute: Attempt to store the mixed event batch.
        self.assertFalse(
            state.store_task_events(
                [
                    TaskEvent(
                        run_id=run_id,
                        task_id=task_id,
                        event="response.created",
                        data='{"type":"response.created"}',
                    ),
                    TaskEvent(
                        run_id=run_id,
                        task_id=task_id,
                        event="response.output_text.delta",
                        data=data,
                    ),
                ]
            )
        )

        # Assert: The invalid payload rejects the whole batch.
        events = state.get_task_events(run_id=run_id, after_task_event_id=None)
        self.assertEqual(events, [])

    def test_reserve_nonce_first_reservation_succeeds(self) -> None:
        """A new nonce reservation should succeed."""
        state = self.state_factory()
        reserved = state.reserve_nonce(
            namespace="superexec:test",
            nonce="nonce-1",
            expires_at=now().timestamp() + 60.0,
        )
        self.assertTrue(reserved)

    def test_reserve_nonce_duplicate_is_rejected(self) -> None:
        """Reserving the same active nonce twice should fail on the second call."""
        state = self.state_factory()
        expires_at = now().timestamp() + 60.0
        self.assertTrue(
            state.reserve_nonce(
                namespace="superexec:test",
                nonce="nonce-1",
                expires_at=expires_at,
            )
        )
        self.assertFalse(
            state.reserve_nonce(
                namespace="superexec:test",
                nonce="nonce-1",
                expires_at=expires_at + 30.0,
            )
        )

    def test_reserve_nonce_invalid_inputs_return_false(self) -> None:
        """Invalid empty namespace/nonce values should be rejected."""
        state = self.state_factory()
        expires_at = now().timestamp() + 60.0

        self.assertFalse(
            state.reserve_nonce(
                namespace="",
                nonce="nonce-1",
                expires_at=expires_at,
            )
        )
        self.assertFalse(
            state.reserve_nonce(
                namespace="superexec:test",
                nonce="",
                expires_at=expires_at,
            )
        )

    def test_reserve_nonce_allows_reuse_after_expiry(self) -> None:
        """Nonce can be reused after its prior reservation expires."""
        state = self.state_factory()
        created_at = now()
        self.assertTrue(
            state.reserve_nonce(
                namespace="superexec:test",
                nonce="nonce-1",
                expires_at=created_at.timestamp() + 1.0,
            )
        )

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = created_at + timedelta(seconds=5)
            self.assertTrue(
                state.reserve_nonce(
                    namespace="superexec:test",
                    nonce="nonce-1",
                    expires_at=created_at.timestamp() + 10.0,
                )
            )
