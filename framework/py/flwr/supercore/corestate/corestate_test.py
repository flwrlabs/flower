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


import unittest
from datetime import timedelta
from typing import Any, cast
from unittest.mock import patch

from parameterized import parameterized

from flwr.common import now
from flwr.common.constant import (
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    Status,
    SubStatus,
)
from flwr.proto.task_pb2 import TaskStatus  # pylint: disable=E0611
from flwr.supercore.constant import TaskType

from . import CoreState


class StateTest(unittest.TestCase):  # pylint: disable=R0904
    """Test all CoreState implementations."""

    # This is to True in each child class
    __test__ = False

    def state_factory(self) -> CoreState:
        """Provide state implementation to test."""
        raise NotImplementedError()

    def test_create_and_get_task(self) -> None:
        """Test creating and retrieving a task."""
        state = self.state_factory()

        task_id = state.create_task(
            task_type=TaskType.MODEL,
            run_id=42,
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
        self.assertEqual(task.run_id, 42)
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

    def test_get_tasks_missing_returns_empty(self) -> None:
        """Missing tasks should return an empty sequence."""
        state = self.state_factory()
        self.assertEqual(state.get_tasks(task_ids=[123]), [])

    def test_get_tasks_single_status_matches(self) -> None:
        """A single-item status sequence should match pending tasks."""
        state = self.state_factory()
        _ = state.create_task(task_type=TaskType.MODEL, run_id=42)

        tasks = state.get_tasks(statuses=[Status.PENDING])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].status.status, Status.PENDING)

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

    def test_get_task_returns_copy(self) -> None:
        """Retrieved task should be a defensive copy."""
        state = self.state_factory()
        task_id = state.create_task(
            task_type=TaskType.SERVER_APP,
            run_id=42,
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

    def test_claim_task_transitions_pending_to_starting(self) -> None:
        """Claiming a task should create a token and move it to starting."""
        state = self.state_factory()
        task_id = state.create_task(task_type="flwr-model", run_id=42)
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

        # Missing tasks cannot be claimed.
        self.assertIsNone(state.claim_task(61016))

        claimed_task_id = state.create_task(task_type="flwr-model", run_id=42)
        finished_task_id = state.create_task(task_type="flwr-model", run_id=42)
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
        task_id = state.create_task(task_type="flwr-model", run_id=42)
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
        task_id = state.create_task(task_type="flwr-model", run_id=42)
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

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type="flwr-model", run_id=42)
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

    def test_expired_task_token_transitions_task_to_finished_failed(self) -> None:
        """Expired task claims should transition tasks to FINISHED:FAILED."""
        state = self.state_factory()
        fixed_now = now()

        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            task_id = state.create_task(task_type="flwr-model", run_id=42)
            assert task_id is not None

            token = state.claim_task(task_id)
            assert token is not None

            mock_dt.now.return_value = fixed_now + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
            )
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

    def test_get_task_by_token_returns_none_for_unknown_token(self) -> None:
        """Unknown task tokens should not resolve to a task."""
        state = self.state_factory()

        self.assertIsNone(state.get_task_by_token("missing-token"))

    def test_create_verify_and_delete_token(self) -> None:
        """Test creating, verifying, and deleting tokens."""
        # Prepare
        state = self.state_factory()
        run_id = 42

        # Execute: create a token
        token = state.create_token(run_id)
        assert token is not None

        # Assert: token should be valid
        self.assertTrue(state.verify_token(run_id, token))

        # Execute: delete the token
        state.delete_token(run_id)

        # Assert: token should no longer be valid
        self.assertFalse(state.verify_token(run_id, token))

    def test_create_token_already_exists(self) -> None:
        """Test creating a token that already exists."""
        # Prepare
        state = self.state_factory()
        run_id = 42
        state.create_token(run_id)

        # Execute
        ret = state.create_token(run_id)

        # Assert: The return is None
        self.assertIsNone(ret)

    def test_get_run_id_by_token(self) -> None:
        """Test retrieving run ID by token."""
        # Prepare
        state = self.state_factory()
        run_id = 42
        token = state.create_token(run_id)
        assert token is not None

        # Execute: get run ID by token
        retrieved_run_id1 = state.get_run_id_by_token(token)
        retrieved_run_id2 = state.get_run_id_by_token("nonexistent_token")

        # Assert: should return the correct run ID
        self.assertEqual(retrieved_run_id1, run_id)
        self.assertIsNone(retrieved_run_id2)

    def test_acknowledge_app_heartbeat_success(self) -> None:
        """Test successfully acknowledging an app heartbeat."""
        # Prepare
        state = self.state_factory()
        run_id = 42
        token = state.create_token(run_id)
        assert token is not None

        # Execute: acknowledge heartbeat
        result = state.acknowledge_app_heartbeat(token)

        # Assert: should return True
        self.assertTrue(result)

        # Assert: token should still be valid
        self.assertTrue(state.verify_token(run_id, token))

    def test_acknowledge_app_heartbeat_nonexistent_token(self) -> None:
        """Test acknowledging heartbeat with nonexistent token."""
        # Prepare
        state = self.state_factory()

        # Execute: acknowledge heartbeat with invalid token
        result = state.acknowledge_app_heartbeat("nonexistent_token")

        # Assert: should return False
        self.assertFalse(result)

    def test_acknowledge_app_heartbeat_extends_expiration_and_cleanup(self) -> None:
        """Test that acknowledging app heartbeat extends token expiration and cleanup is
        performed when expired."""
        # Prepare
        state = self.state_factory()
        created_at = now()
        run_id1 = 42
        run_id2 = 123
        token1 = state.create_token(run_id1)
        token2 = state.create_token(run_id2)
        assert token1 is not None and token2 is not None

        # Execute: send heartbeat for token2 to keep it alive
        state.acknowledge_app_heartbeat(token2)

        # Mock datetime to simulate time passage
        # token1 should expire in HEARTBEAT_DEFAULT_INTERVAL
        # token2 should expire in HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
        with patch("datetime.datetime") as mock_dt:
            # Advance time just before token1 expiration
            mock_dt.now.return_value = created_at + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL - 1
            )

            # Verify tokens are valid
            self.assertTrue(state.verify_token(run_id1, token1))
            self.assertTrue(state.verify_token(run_id2, token2))

            # Advance time past token1 expiration
            mock_dt.now.return_value = created_at + timedelta(
                seconds=HEARTBEAT_DEFAULT_INTERVAL + 1
            )

            # Assert: token1 should be cleaned up, token2 should still be valid
            self.assertFalse(state.verify_token(run_id1, token1))
            self.assertTrue(state.verify_token(run_id2, token2))

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
