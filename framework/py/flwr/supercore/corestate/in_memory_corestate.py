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
"""In-memory CoreState implementation."""


import hashlib
import secrets
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from flwr.common import now
from flwr.common.constant import (
    FLWR_TASK_TOKEN_LENGTH,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    TASK_ID_NUM_BYTES,
    Status,
    SubStatus,
)
from flwr.common.typing import Fab
from flwr.proto.task_pb2 import Task, TaskStatus  # pylint: disable=E0611

from ..object_store import ObjectStore
from .corestate import CoreState
from .utils import generate_rand_int_from_bytes


@dataclass
class TokenRecord:
    """Record containing token and heartbeat information."""

    token: str
    active_until: int


class InMemoryCoreState(CoreState):  # pylint: disable=too-many-instance-attributes
    """In-memory CoreState implementation."""

    def __init__(self, object_store: ObjectStore) -> None:
        self._object_store = object_store
        self.fab_store: dict[str, Fab] = {}
        self.lock_fab_store = Lock()
        self.nonce_store: dict[tuple[str, str], float] = {}
        self.lock_nonce_store = Lock()
        self.task_store: dict[int, Task] = {}
        # Store task ID to token mapping
        self.task_token_store: dict[int, TokenRecord] = {}
        # Store token to task ID mapping
        self.task_token_to_task_id: dict[str, int] = {}
        self.task_logs: dict[int, list[tuple[float, str]]] = {}
        self.log_lock = Lock()
        self.lock_task_store = Lock()

    @property
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""
        return self._object_store

    def store_fab(self, fab: Fab) -> str:
        """Store a FAB."""
        fab_hash = hashlib.sha256(fab.content).hexdigest()
        if fab.hash_str and fab.hash_str != fab_hash:
            raise ValueError(
                f"FAB hash mismatch: provided {fab.hash_str}, computed {fab_hash}"
            )
        with self.lock_fab_store:
            # Keep launch behavior: last write wins for metadata under the same
            # content hash.
            self.fab_store[fab_hash] = Fab(
                hash_str=fab_hash,
                content=fab.content,
                verifications=dict(fab.verifications),
            )
        return fab_hash

    def get_fab(self, fab_hash: str) -> Fab | None:
        """Return a FAB by hash."""
        with self.lock_fab_store:
            if (fab := self.fab_store.get(fab_hash)) is None:
                return None
            # Launch tradeoff: do not recompute content hash on reads; rely on
            # write-time validation and hash-addressed lookup.
            return Fab(
                hash_str=fab.hash_str,
                content=fab.content,
                verifications=dict(fab.verifications),
            )

    def add_task_log(self, task_id: int, log_message: str) -> None:
        """Add a log entry to the task logs for the specified `task_id`."""
        with self.lock_task_store:
            if task_id not in self.task_store:
                raise ValueError(f"Task {task_id} not found")
        with self.log_lock:
            timestamp = now().timestamp()
            task_logs = self.task_logs.setdefault(task_id, [])
            task_logs.append((timestamp, log_message))

    def get_task_log(
        self, task_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get task logs for the specified `task_id`."""
        # We don't check if the task exists before querying logs
        # because the task_id is validated by the authz layer

        with self.log_lock:
            task_logs = self.task_logs.get(task_id, [])
            if after_timestamp is None:
                after_timestamp = 0.0
            timestamps = [timestamp for timestamp, _ in task_logs]
            # Polling is strict-after: entries at the checkpoint timestamp have
            # already been delivered, so resume after the rightmost equal value.
            index = bisect_right(timestamps, after_timestamp)
            latest_timestamp = task_logs[-1][0] if index < len(task_logs) else 0.0
            return "".join(log for _, log in task_logs[index:]), latest_timestamp

    def create_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        task_type: str,
        run_id: int,
        fab_hash: str | None = None,
        model_ref: str | None = None,
        connector_ref: str | None = None,
    ) -> int | None:
        """Create a task and return its ID."""
        with self.lock_task_store:
            task_id = generate_rand_int_from_bytes(TASK_ID_NUM_BYTES)

            task = Task(
                task_id=task_id,
                type=task_type,
                run_id=run_id,
                status=TaskStatus(status=Status.PENDING, sub_status="", details=""),
                pending_at=now().isoformat(),
                fab_hash=fab_hash,
                model_ref=model_ref,
                connector_ref=connector_ref,
            )

            self.task_store[task_id] = task
            return task_id

    def get_tasks(  # pylint: disable=too-many-arguments
        self,
        *,
        task_ids: Sequence[int] | None = None,
        run_ids: Sequence[int] | None = None,
        statuses: Sequence[str] | None = None,
        order_by: Literal["pending_at"] | None = None,
        ascending: bool = True,
        limit: int | None = None,
    ) -> Sequence[Task]:
        """Retrieve information about tasks based on the specified filters."""
        if order_by not in (None, "pending_at"):
            raise AssertionError("`order_by` must be 'pending_at' or None")

        if limit is not None and limit < 0:
            raise AssertionError("`limit` must be >= 0")

        if isinstance(statuses, str):
            raise ValueError("`statuses` must be a sequence of strings")

        with self.lock_task_store:
            matched_task_ids = set(self.task_store.keys())

            if task_ids is not None:
                if not task_ids:
                    return []
                matched_task_ids &= set(task_ids)

            if run_ids is not None:
                if not run_ids:
                    return []
                run_id_set = set(run_ids)
                matched_task_ids &= {
                    task_id
                    for task_id in matched_task_ids
                    if self.task_store[task_id].run_id in run_id_set
                }

            if statuses is not None:
                if not statuses:
                    return []
                status_set = set(statuses)
                matched_task_ids &= {
                    task_id
                    for task_id in matched_task_ids
                    if self.task_store[task_id].status.status in status_set
                }

            tasks = [self.task_store[task_id] for task_id in matched_task_ids]

            if order_by is not None:
                tasks = sorted(
                    tasks,
                    key=lambda task: task.pending_at,
                    reverse=not ascending,
                )

            if limit is not None:
                tasks = tasks[:limit]

            result: list[Task] = []
            for task in tasks:
                task_copy = Task()
                task_copy.CopyFrom(task)
                result.append(task_copy)
            return result

    def claim_task(self, task_id: int) -> str | None:
        """Atomically claim a pending task."""
        token = secrets.token_hex(FLWR_TASK_TOKEN_LENGTH)
        with self.lock_task_store:
            task = self.task_store.get(task_id)
            if task is None or task_id in self.task_token_store:
                return None
            if task.status.status != Status.PENDING:
                return None

            # Claiming moves the task into STARTING and records the heartbeat state.
            claimed_at = now()
            task.starting_at = claimed_at.isoformat()
            task.status.CopyFrom(
                TaskStatus(status=Status.STARTING, sub_status="", details="")
            )
            self.task_token_store[task_id] = TokenRecord(
                token=token,
                active_until=int(claimed_at.timestamp()) + HEARTBEAT_DEFAULT_INTERVAL,
            )
            self.task_token_to_task_id[token] = task_id
            return token

    def activate_task(self, task_id: int) -> bool:
        """Move a task from starting to running."""
        with self.lock_task_store:
            # Expire non-responsive tasks before transitioning task status.
            self._cleanup_expired_task_tokens_locked()

            # Transition task from STARTING -> RUNNING.
            task = self.task_store.get(task_id)
            if task is None or task.status.status != Status.STARTING:
                return False

            task.running_at = now().isoformat()
            task.status.CopyFrom(
                TaskStatus(status=Status.RUNNING, sub_status="", details="")
            )
            return True

    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished."""
        with self.lock_task_store:
            # Expire non-responsive tasks before transitioning task status.
            self._cleanup_expired_task_tokens_locked()

            # Transition task to FINISHED
            task = self.task_store.get(task_id)
            if task is None or task.status.status == Status.FINISHED:
                return False

            if sub_status == SubStatus.COMPLETED:
                # Only allow transition to COMPLETED if currently RUNNING
                if task.status.status != Status.RUNNING:
                    return False

            task.finished_at = now().isoformat()
            task.status.CopyFrom(
                TaskStatus(
                    status=Status.FINISHED, sub_status=sub_status, details=details
                )
            )

            # Revoke any existing task token now that the task is finished.
            if (record := self.task_token_store.pop(task_id, None)) is not None:
                self.task_token_to_task_id.pop(record.token, None)
            return True

    def acknowledge_task_heartbeat(self, task_id: int) -> bool:
        """Extend heartbeat state for the claimed task."""
        with self.lock_task_store:
            # Heartbeats are accepted only for starting and running tasks
            self._cleanup_expired_task_tokens_locked()
            task = self.task_store.get(task_id)
            record = self.task_token_store.get(task_id)
            if task is None or record is None or task.status.status == Status.FINISHED:
                return False

            now_int = int(now().timestamp())
            record.active_until = (
                now_int + HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
            )
            return True

    def get_task_by_token(self, token: str) -> Task | None:
        """Return the task associated with the task token, if valid."""
        with self.lock_task_store:
            # Resolve tokens after cleanup so callers never receive expired claims.
            self._cleanup_expired_task_tokens_locked()
            task_id = self.task_token_to_task_id.get(token)
            if task_id is None:
                return None
            task = Task()
            task.CopyFrom(self.task_store[task_id])
            return task

    def _cleanup_expired_task_tokens_locked(self) -> None:
        """Remove expired task tokens.

        Callers must acquire `lock_task_store` before calling this method.
        Expired tasks are marked as finished with a failed status, and their
        tokens are removed.
        """
        expired_at = now()
        current = int(expired_at.timestamp())
        expired_tasks: list[Task] = []
        for task_id, record in list(self.task_token_store.items()):
            if record.active_until < current:
                # The task is considered expired. Mark it as finished with a failed
                # status if it's not already finished, and remove the token.
                task = self.task_store.get(task_id)
                if task and task.status.status != Status.FINISHED:
                    task.finished_at = expired_at.isoformat()
                    task.status.CopyFrom(
                        TaskStatus(
                            status=Status.FINISHED,
                            sub_status=SubStatus.FAILED,
                            details="No heartbeat received from the task",
                        )
                    )
                    expired_task = Task()
                    expired_task.CopyFrom(task)
                    expired_tasks.append(expired_task)
                del self.task_token_store[task_id]
                self.task_token_to_task_id.pop(record.token, None)

        if expired_tasks:
            self._on_task_tokens_expired(expired_tasks)

    def _on_task_tokens_expired(self, tasks: list[Task]) -> None:
        """Handle cleanup of expired task tokens.

        Override in subclasses to add custom cleanup logic.

        Parameters
        ----------
        tasks : list[Task]
            Copies of tasks whose claims expired and were marked FINISHED:FAILED.
        """

    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Atomically reserve a nonce in a namespace."""
        if namespace == "" or nonce == "":
            return False
        with self.lock_nonce_store:
            self._cleanup_expired_nonces()
            key = (namespace, nonce)
            if key in self.nonce_store:
                return False
            self.nonce_store[key] = expires_at
            return True

    def _cleanup_expired_nonces(self) -> None:
        """Delete nonce reservations that are no longer active."""
        current = now().timestamp()
        for key, expires_at in list(self.nonce_store.items()):
            if expires_at < current:
                del self.nonce_store[key]
