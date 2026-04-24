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

from flwr.common import now
from flwr.common.constant import HEARTBEAT_DEFAULT_INTERVAL, Status
from flwr.proto.task_pb2 import TaskStatus  # pylint: disable=E0611

from . import CoreState


class StateTest(unittest.TestCase):
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
            task_type="flwr-model",
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
        self.assertEqual(task.type, "flwr-model")
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
        _ = state.create_task(task_type="flwr-model", run_id=42)

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
            task_type="flwr-serverapp",
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
