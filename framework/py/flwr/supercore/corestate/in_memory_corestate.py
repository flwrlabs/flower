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
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from flwr.common import now
from flwr.common.constant import (
    FLWR_APP_TOKEN_LENGTH,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    TASK_ID_NUM_BYTES,
    Status,
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
    active_until: float


class InMemoryCoreState(CoreState):  # pylint: disable=too-many-instance-attributes
    """In-memory CoreState implementation."""

    def __init__(self, object_store: ObjectStore) -> None:
        self._object_store = object_store
        self.fab_store: dict[str, Fab] = {}
        self.lock_fab_store = Lock()
        # Store run ID to token mapping and token to run ID mapping
        self.token_store: dict[int, TokenRecord] = {}
        self.token_to_run_id: dict[str, int] = {}
        self.lock_token_store = Lock()
        self.nonce_store: dict[tuple[str, str], float] = {}
        self.lock_nonce_store = Lock()
        self.task_store: dict[int, Task] = {}
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

            if statuses is not None:
                if not statuses:
                    return []
                status_set = set(statuses)
                matched_task_ids &= {
                    task_id
                    for task_id in matched_task_ids
                    if determine_task_status(self.task_store[task_id]).status
                    in status_set
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
                task_copy.status.CopyFrom(determine_task_status(task))
                result.append(task_copy)
            return result

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        with self.lock_token_store:
            if run_id in self.token_store:
                return None  # Token already created for this run ID

            self.token_store[run_id] = TokenRecord(
                token=token, active_until=now().timestamp() + HEARTBEAT_DEFAULT_INTERVAL
            )
            self.token_to_run_id[token] = run_id
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        self._cleanup_expired_tokens()
        with self.lock_token_store:
            record = self.token_store.get(run_id)
            return record is not None and record.token == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        with self.lock_token_store:
            record = self.token_store.pop(run_id, None)
            if record is not None:
                self.token_to_run_id.pop(record.token, None)

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        self._cleanup_expired_tokens()
        with self.lock_token_store:
            return self.token_to_run_id.get(token)

    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token."""
        # Clean up expired tokens
        self._cleanup_expired_tokens()

        with self.lock_token_store:
            # Return False if token is not found
            if token not in self.token_to_run_id:
                return False

            # Get the run_id and update heartbeat info
            run_id = self.token_to_run_id[token]
            record = self.token_store[run_id]
            current = now().timestamp()
            record.active_until = (
                current + HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
            )
            return True

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens and perform additional cleanup.

        This method is called before token operations to ensure integrity.
        Subclasses can override `_on_tokens_expired` to add custom cleanup logic.
        """
        with self.lock_token_store:
            current = now().timestamp()
            expired_records: list[tuple[int, float]] = []
            for run_id, record in list(self.token_store.items()):
                if record.active_until < current:
                    expired_records.append((run_id, record.active_until))
                    # Remove from both stores
                    del self.token_store[run_id]
                    self.token_to_run_id.pop(record.token, None)

            # Hook for subclasses
            if expired_records:
                self._on_tokens_expired(expired_records)

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Handle cleanup of expired tokens.

        Override in subclasses to add custom cleanup logic.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
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


def determine_task_status(task: Task) -> TaskStatus:
    """Determine the status of a task based on timestamp fields."""
    if task.pending_at:
        if task.finished_at:
            return TaskStatus(
                status=Status.FINISHED,
                sub_status=task.status.sub_status,
                details=task.status.details,
            )
        if task.starting_at:
            if task.running_at:
                return TaskStatus(status=Status.RUNNING, sub_status="", details="")
            return TaskStatus(status=Status.STARTING, sub_status="", details="")
        return TaskStatus(status=Status.PENDING, sub_status="", details="")
    raise ValueError(f"The task {task.task_id} does not have a valid status.")
