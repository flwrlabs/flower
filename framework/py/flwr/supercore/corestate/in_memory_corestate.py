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
from datetime import datetime, timedelta
from logging import ERROR
from threading import Lock
from typing import Literal, cast

from flwr.app.message import Message
from flwr.common import now
from flwr.common.constant import (
    FLWR_TASK_TOKEN_LENGTH,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    SERIES_ID_NUM_BYTES,
    TASK_ID_NUM_BYTES,
    Status,
    SubStatus,
)
from flwr.common.logger import log
from flwr.common.typing import Fab
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent, TaskStatus  # pylint: disable=E0611

from ..object_store import ObjectStore
from .corestate import CoreState
from .utils import (
    generate_rand_int_from_bytes,
    validate_task_event_data,
    validate_task_message,
)


@dataclass
class TokenRecord:
    """Record containing token and heartbeat information."""

    token: str
    active_until: datetime


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
        self.task_message_store: dict[str, Message] = {}
        self.lock_task_message_store = Lock()
        self.run_series_store: dict[int, RunSeries] = {}
        self.task_event_store: dict[int, list[TaskEvent]] = {}
        self.lock_task_event_store = Lock()
        self.lock_run_series_store = Lock()
        self._next_task_event_id = 1

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

    def store_run_in_series(
        self,
        run_id: int,
        federation: str,
        series_id: int | None,
    ) -> int | None:
        """Store a run in a run series and return the series ID."""
        with self.lock_run_series_store:
            if series_id is not None:
                # Reuse only an existing run series owned by the requested federation.
                existing = self.run_series_store.get(series_id)
                if existing is None:
                    log(ERROR, "Run series %d not found", series_id)
                    return None
                if existing.federation != federation:
                    log(
                        ERROR,
                        "Run series %d belongs to federation %r, not %r",
                        series_id,
                        existing.federation,
                        federation,
                    )
                    return None
                run_series = existing
                resolved_series_id = series_id

            else:
                # No series was provided, so create a new one before linking the run.
                new_series_id = generate_rand_int_from_bytes(SERIES_ID_NUM_BYTES)
                if new_series_id in self.run_series_store:
                    return None

                timestamp = now().isoformat()
                run_series = RunSeries(
                    series_id=new_series_id,
                    federation=federation,
                    description="",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                self.run_series_store[new_series_id] = run_series
                resolved_series_id = new_series_id

            # Store the membership last so callers only receive linked series IDs.
            if run_id in run_series.run_ids:
                return None
            run_series.run_ids.append(run_id)
            return resolved_series_id

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
        requesting_task_id: int | None = None,
    ) -> int | None:
        """Create a task and return its ID."""
        with self.lock_task_store:
            if requesting_task_id is not None:
                requesting_task = self.task_store.get(requesting_task_id)
                if (
                    requesting_task is None
                    or requesting_task.status.status == Status.FINISHED
                ):
                    return None

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
            # Expire non-responsive tasks before getting tasks
            self._cleanup_expired_task_tokens_locked()

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
                active_until=claimed_at + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL),
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

            ttl = timedelta(seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL)
            record.active_until = now() + ttl
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

    def store_task_message(  # pylint: disable=too-many-return-statements
        self, message: Message
    ) -> bool:
        """Store one task-addressed Message."""
        message_id = message.metadata.message_id
        if validate_task_message(message):
            return False
        src_task_id = cast(int, message.metadata.src_task_id)
        dst_task_id = cast(int, message.metadata.dst_task_id)

        with self.lock_task_store, self.lock_task_message_store:
            self._cleanup_expired_task_tokens_locked()
            self._cleanup_invalid_task_messages_locked(now().timestamp())
            src_task = self.task_store.get(src_task_id)
            dst_task = self.task_store.get(dst_task_id)
            if src_task is None or dst_task is None:
                return False
            if src_task.run_id != dst_task.run_id:
                log(
                    ERROR,
                    "Cannot store message: source task %d and destination task %d "
                    "belong to different runs.",
                    src_task_id,
                    dst_task_id,
                )
                return False
            if message.metadata.run_id != src_task.run_id:
                log(
                    ERROR,
                    "Cannot store message for task %s: message run ID %d "
                    "does not match task run ID %d.",
                    message_id,
                    message.metadata.run_id,
                    src_task.run_id,
                )
                return False
            if dst_task.status.status == Status.FINISHED:
                return False

            if message_id in self.task_message_store:
                return False
            self.task_message_store[message_id] = message
            return True

    def get_task_message(
        self,
        *,
        dst_task_ids: Sequence[int] | None = None,
        limit: int | None = None,
        order_by: Literal["created_at"] | None = None,
    ) -> Sequence[Message]:
        """Retrieve undelivered task-addressed Messages."""
        if order_by not in (None, "created_at"):
            raise AssertionError("`order_by` must be 'created_at' or None")
        if limit is not None and limit < 0:
            raise AssertionError("`limit` must be >= 0")
        if limit == 0:
            return []
        if dst_task_ids is not None and not dst_task_ids:
            return []

        with self.lock_task_store, self.lock_task_message_store:
            self._cleanup_expired_task_tokens_locked()
            current = now().timestamp()
            self._cleanup_invalid_task_messages_locked(current)

            # Filter by dst_task_id
            dst_task_id_set = set(dst_task_ids) if dst_task_ids is not None else None
            selected_messages = [
                msg
                for msg in self.task_message_store.values()
                if dst_task_id_set is None
                or msg.metadata.dst_task_id in dst_task_id_set
            ]

            # Apply requested sort order
            if order_by == "created_at":
                selected_messages.sort(key=lambda msg: msg.metadata.created_at)

            # Apply limit
            if limit is not None:
                selected_messages = selected_messages[:limit]

            # Delete selected messages from storage
            for msg in selected_messages:
                del self.task_message_store[msg.metadata.message_id]

        return selected_messages

    def store_task_events(
        self,
        events: Sequence[TaskEvent],
    ) -> bool:
        """Store task-produced run events."""
        if not events:
            return False

        try:
            for event in events:
                validate_task_event_data(event.data)
        except ValueError:
            return False

        with self.lock_task_event_store:
            current = now().isoformat()
            for event in events:
                task_events = self.task_event_store.setdefault(event.run_id, [])
                event.id = self._next_task_event_id
                event.timestamp = current
                task_events.append(event)
                self._next_task_event_id += 1

        return True

    def get_task_events(
        self,
        *,
        run_id: int | None = None,
        after_task_event_id: int | None = None,
    ) -> Sequence[TaskEvent]:
        """Return task-produced run events after the cursor."""
        cursor = after_task_event_id if after_task_event_id is not None else 0
        with self.lock_task_event_store:
            if run_id is None:
                events = [
                    event
                    for task_events in self.task_event_store.values()
                    for event in task_events
                ]
            else:
                events = list(self.task_event_store.get(run_id, []))
            return [
                event
                for event in sorted(events, key=lambda event: event.id)
                if event.id > cursor
            ]

    def _cleanup_expired_task_tokens_locked(self) -> None:
        """Remove expired task tokens.

        Callers must acquire `lock_task_store` before calling this method.
        Expired tasks are marked as finished with a failed status, and their
        tokens are removed.
        """
        current = now()
        expired_tasks: list[Task] = []
        for task_id, record in list(self.task_token_store.items()):
            if record.active_until < current:
                # The task is considered expired. Mark it as finished with a failed
                # status if it's not already finished, and remove the token.
                task = self.task_store.get(task_id)
                if task and task.status.status != Status.FINISHED:
                    task.finished_at = record.active_until.isoformat()
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

    def _cleanup_invalid_task_messages_locked(self, current: float) -> None:
        """Remove expired Messages and Messages for invalid destination tasks."""
        for message_id, message in list(self.task_message_store.items()):
            dst_task_id = message.metadata.dst_task_id
            if dst_task_id is None:
                del self.task_message_store[message_id]
                continue

            dst_task = self.task_store.get(dst_task_id)
            if (
                dst_task is None
                or dst_task.status.status == Status.FINISHED
                or message.metadata.created_at + message.metadata.ttl <= current
            ):
                del self.task_message_store[message_id]

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
