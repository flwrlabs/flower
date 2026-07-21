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
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import call, patch

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
from flwr.supercore.constant import (
    OBJECT_PUSH_SESSION_TTL_SECONDS,
    AutomationStatus,
    TaskType,
)
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

    def test_bind_and_get_run_connectors(self) -> None:
        """Run connector bindings should be deterministic and idempotent."""
        state = self.state_factory()

        state.bind_connectors_to_run(
            run_id=42,
            connector_refs=["notion", "calendar", "notion"],
        )
        state.bind_connectors_to_run(run_id=42, connector_refs=["notion"])

        self.assertEqual(
            list(state.get_run_connector_refs(run_id=42)),
            ["calendar", "notion"],
        )

        self.assertFalse(
            state.bind_connectors_to_run(run_id=43, connector_refs="notion")
        )
        self.assertEqual(list(state.get_run_connector_refs(run_id=43)), [])

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

    def test_delete_sessions_in_run(self) -> None:
        """Deleting sessions for a run preserves sessions belonging to other runs."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        other_run_id = self.other_task_run_id(state)
        session_ids = [state.start_session(run_id) for _ in range(2)]
        other_session_id = state.start_session(other_run_id)
        object_ids = ["a" * 64, "b" * 64, "c" * 64]
        for session_id, object_id in zip(
            [*session_ids, other_session_id], object_ids, strict=True
        ):
            state.preregister_object_tree(ObjectTree(object_id=object_id), session_id)

        state.delete_sessions_in_run(run_id)

        for session_id in session_ids:
            with self.assertRaisesRegex(ValueError, "Unknown object push session"):
                state.preregister_object_tree(
                    ObjectTree(object_id="d" * 64), session_id
                )
        self.assertEqual(
            state.preregister_object_tree(
                ObjectTree(object_id="e" * 64), other_session_id
            ),
            ["e" * 64],
        )

    def test_store_object_rejects_invalid_session_membership(self) -> None:
        """Objects must be pending in a session belonging to the run."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        object_id = "a" * 64
        session_id = state.start_session(run_id)
        state.preregister_object_tree(ObjectTree(object_id=object_id), session_id)

        with (
            patch.object(state.object_store, "put") as put_object,
            patch.object(state, "_cleanup_push_session") as cleanup_session,
        ):
            self.assertFalse(
                state.store_object(run_id + 1, session_id, object_id, b"content")
            )
            self.assertFalse(
                state.store_object(run_id, "unknown-session", object_id, b"content")
            )
            self.assertFalse(
                state.store_object(run_id, session_id, "unknown-object-id", b"content")
            )

        put_object.assert_not_called()
        cleanup_session.assert_not_called()

    def test_store_object_resolves_empty_session_id(self) -> None:
        """An empty session ID resolves through pending object membership."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        object_id = "a" * 64
        session_id = state.start_session(run_id)
        state.preregister_object_tree(ObjectTree(object_id=object_id), session_id)

        with patch.object(state.object_store, "put") as put_object:
            self.assertTrue(state.store_object(run_id, "", object_id, b"content"))

        put_object.assert_called_once_with(object_id, b"content")

    def test_store_object_cleans_up_expired_session(self) -> None:
        """An object cannot be stored after its push session expires."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        object_id = "a" * 64
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        with patch("flwr.supercore.date.datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = created_at
            session_id = state.start_session(run_id)
            state.preregister_object_tree(ObjectTree(object_id=object_id), session_id)

        expired_at = created_at + timedelta(seconds=OBJECT_PUSH_SESSION_TTL_SECONDS + 1)
        with (
            patch("flwr.supercore.date.datetime.datetime") as mock_datetime,
            patch.object(state.object_store, "put") as put_object,
            patch.object(state, "_cleanup_push_session") as cleanup_session,
        ):
            mock_datetime.now.return_value = expired_at
            stored = state.store_object(run_id, session_id, object_id, b"content")

        self.assertFalse(stored)
        put_object.assert_not_called()
        cleanup_session.assert_called_once_with(session_id, cleanup_messages=True)

    def test_store_object_refreshes_session_and_cleans_up_on_completion(self) -> None:
        """A successful store refreshes TTL and cleans up an empty session."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        parent_id = "a" * 64
        child_id = "b" * 64
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        object_tree = ObjectTree(
            object_id=parent_id,
            children=[ObjectTree(object_id=child_id)],
        )
        with patch("flwr.supercore.date.datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = created_at
            session_id = state.start_session(run_id)
            state.preregister_object_tree(object_tree, session_id)

        first_store_at = created_at + timedelta(
            seconds=OBJECT_PUSH_SESSION_TTL_SECONDS - 1
        )
        second_store_at = created_at + timedelta(
            seconds=OBJECT_PUSH_SESSION_TTL_SECONDS + 1
        )
        with (
            patch.object(state.object_store, "put") as put_object,
            patch.object(state, "_cleanup_push_session") as cleanup_session,
        ):
            with patch("flwr.supercore.date.datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value = first_store_at
                self.assertTrue(
                    state.store_object(run_id, session_id, child_id, b"child")
                )
            cleanup_session.assert_not_called()

            with patch("flwr.supercore.date.datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value = second_store_at
                self.assertTrue(
                    state.store_object(run_id, session_id, parent_id, b"parent")
                )

        self.assertEqual(put_object.call_count, 2)
        cleanup_session.assert_called_once_with(session_id, cleanup_messages=False)

    def test_store_object_preserves_pending_claim_when_object_store_fails(self) -> None:
        """An ObjectStore error returns False without consuming the pending claim."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        object_id = "a" * 64
        session_id = state.start_session(run_id)
        state.preregister_object_tree(ObjectTree(object_id=object_id), session_id)

        with patch.object(
            state.object_store,
            "put",
            side_effect=[RuntimeError("write failed"), None],
        ) as put_object:
            self.assertFalse(
                state.store_object(run_id, session_id, object_id, b"content")
            )
            self.assertTrue(
                state.store_object(run_id, session_id, object_id, b"content")
            )

        self.assertEqual(put_object.call_count, 2)

    def test_get_object_returns_object_store_result_without_cleanup(self) -> None:
        """Available, unknown, and unowned unavailable objects are returned directly."""
        state = self.state_factory()
        with (
            patch.object(
                state.object_store,
                "get",
                side_effect=[b"content", None, b""],
            ) as load_object,
            patch.object(state, "_cleanup_push_session") as cleanup_session,
        ):
            self.assertEqual(state.get_object(1, "available"), b"content")
            self.assertIsNone(state.get_object(1, "unknown"))
            self.assertEqual(state.get_object(1, "unavailable"), b"")

        self.assertEqual(load_object.call_count, 3)
        cleanup_session.assert_not_called()

    def test_get_object_cleans_up_expired_sessions_and_reloads(self) -> None:
        """An unavailable object triggers cleanup for all expired sessions."""
        state = self.state_factory()
        run_id = self.task_run_id(state)
        object_id = "a" * 64
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        with patch("flwr.supercore.date.datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = created_at
            session_ids = []
            for root_object_id in ("b" * 64, "c" * 64):
                session_id = state.start_session(run_id)
                session_ids.append(session_id)
                state.preregister_object_tree(
                    ObjectTree(
                        object_id=root_object_id,
                        children=[ObjectTree(object_id=object_id)],
                    ),
                    session_id,
                )

        expired_at = created_at + timedelta(seconds=OBJECT_PUSH_SESSION_TTL_SECONDS + 1)
        with (
            patch("flwr.supercore.date.datetime.datetime") as mock_datetime,
            patch.object(
                state,
                "_cleanup_push_session",
                wraps=state._cleanup_push_session,  # pylint: disable=W0212
            ) as cleanup_session,
        ):
            mock_datetime.now.return_value = expired_at
            self.assertIsNone(state.get_object(run_id, "unknown-object-id"))
            cleanup_session.assert_not_called()
            self.assertIsNone(state.get_object(run_id, object_id))

        cleanup_session.assert_has_calls(
            [call(session_id, cleanup_messages=True) for session_id in session_ids],
            any_order=True,
        )
        self.assertEqual(cleanup_session.call_count, 2)

    def test_store_run_in_series_creates_id(self) -> None:
        """Storing a run in a run series should create a nonzero ID."""
        state = self.state_factory()

        series_id = state.store_run_in_series(
            run_id=123,
            federation_id="@me/fed-a",
            series_id=None,
            description="Initial description",
        )

        self.assertIsNotNone(series_id)
        assert series_id is not None
        self.assertGreater(series_id, 0)
        self.assertEqual(
            state.store_run_in_series(
                run_id=456,
                federation_id="@me/fed-a",
                series_id=series_id,
                description="Replacement description",
            ),
            series_id,
        )
        run_series = state.get_run_series(series_ids=[series_id])
        self.assertEqual(run_series[0].description, "Initial description")

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

    def test_advance_and_finish_automation(self) -> None:
        """Automation advance should update records and finish terminally."""
        state = self.state_factory()
        current = now()

        # Create a recurring automation with two finite occurrences.
        previous_next_run_at = (current - timedelta(seconds=30)).isoformat()
        next_run_at = (current + timedelta(seconds=30)).isoformat()
        recurring = self.store_automation(
            state,
            series_id=1,
            next_run_at=previous_next_run_at,
            fixed_interval=60,
            max_runs=2,
        )

        # Advance the first occurrence and reject the stale due-time claim.
        self.assertTrue(
            state.advance_automation(
                recurring.automation_id,
                previous_next_run_at=previous_next_run_at,
                next_run_at=next_run_at,
            )
        )
        self.assertFalse(
            state.advance_automation(
                recurring.automation_id,
                previous_next_run_at=previous_next_run_at,
                next_run_at=next_run_at,
            )
        )
        updated = state.list_automations(
            federation="@me/fed-a",
            statuses=[AutomationStatus.ACTIVE],
            order_by="updated_at",
        )
        self.assertEqual(updated[0].remaining_runs, 1)
        self.assertEqual(updated[0].next_run_at, next_run_at)

        # Advance the final occurrence, then complete the automation.
        self.assertTrue(
            state.advance_automation(
                recurring.automation_id,
                previous_next_run_at=next_run_at,
                next_run_at=None,
            )
        )
        self.assertTrue(
            state.finish_automation(
                recurring.automation_id,
                status=AutomationStatus.COMPLETED,
            )
        )
        completed = state.list_automations(
            federation="@me/fed-a",
            statuses=[AutomationStatus.COMPLETED],
            order_by="updated_at",
        )
        self.assertEqual(
            [automation.automation_id for automation in completed],
            [recurring.automation_id],
        )

        # Mark an advanced automation as failed when execution cannot proceed.
        failed_previous_next_run_at = (current - timedelta(seconds=15)).isoformat()
        failing = self.store_automation(
            state,
            series_id=2,
            next_run_at=failed_previous_next_run_at,
        )
        self.assertTrue(
            state.advance_automation(
                failing.automation_id,
                previous_next_run_at=failed_previous_next_run_at,
                next_run_at=None,
            )
        )
        self.assertTrue(
            state.finish_automation(
                failing.automation_id,
                status=AutomationStatus.FAILED,
            )
        )
        failed = state.list_automations(
            federation="@me/fed-a",
            statuses=[AutomationStatus.FAILED],
            order_by="updated_at",
        )
        self.assertEqual(
            [automation.automation_id for automation in failed],
            [failing.automation_id],
        )
        self.assertEqual(failed[0].next_run_at, failed_previous_next_run_at)

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
                usage_type="model_inference",
                provider="openai/gpt-test",
            ),
        )
        state.add_task_usage(
            task_id,
            TaskUsage(
                input_tokens=999,
                usage_type="model_inference",
                provider="openai/gpt-test",
            ),
        )

        usages = state.get_task_usage(task_ids=[task_id])

        self.assertEqual(len(usages), 2)
        usage = usages[0]
        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.usage_type, "model_inference")
        self.assertEqual(usage.provider, "openai/gpt-test")
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
