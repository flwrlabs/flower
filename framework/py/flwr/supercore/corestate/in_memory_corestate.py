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

# pylint: disable=too-many-lines
import hashlib
import secrets
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from logging import ERROR
from threading import Lock, RLock
from typing import Literal, cast
from uuid import uuid4

from flwr.app import Context, Message
from flwr.app.user_config import UserConfig
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
from flwr.proto.control_pb2 import Automation  # pylint: disable=E0611
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import (  # pylint: disable=E0611
    Task,
    TaskEvent,
    TaskStatus,
    TaskUsage,
)
from flwr.supercore.constant import OBJECT_PUSH_SESSION_TTL_SECONDS, AutomationStatus
from flwr.supercore.date import now
from flwr.supercore.fab import Fab
from flwr.supercore.typing import ConnectorOAuthSessionRecord, ConnectorRecord

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


@dataclass
class TaskUsageRecord:
    """Record containing task usage and reporting metadata."""

    id: int
    task_id: int
    run_id: int
    usage: TaskUsage
    created_at: datetime
    reported_at: datetime | None


@dataclass
class AutomationRecord:
    """Record containing automation metadata and run template."""

    automation: Automation
    fab_id: str | None
    fab_version: str | None
    fab_hash: str | None
    override_config: UserConfig
    federation_config: SimulationConfig | None
    primary_task_type: str


@dataclass
class ObjectPushSession:
    """In-memory object push session."""

    run_id: int
    expires_at: datetime
    root_object_ids: set[str]
    pending_object_ids: set[str]


class InMemoryCoreState(
    CoreState
):  # pylint: disable=R0904,too-many-instance-attributes
    """In-memory CoreState implementation."""

    def __init__(self, object_store: ObjectStore) -> None:
        self._object_store = object_store
        self.fab_store: dict[str, Fab] = {}
        self.lock_fab_store = Lock()
        self.connector_store: dict[tuple[str, str], ConnectorRecord] = {}
        self.lock_connector_store = Lock()
        self.run_connector_store: dict[int, set[str]] = {}
        self.lock_run_connector_store = Lock()
        self.connector_oauth_session_store: dict[str, ConnectorOAuthSessionRecord] = {}
        self.lock_connector_oauth_session_store = Lock()
        self.nonce_store: dict[tuple[str, str], float] = {}
        self.lock_nonce_store = Lock()
        self.run_series_store: dict[int, RunSeries] = {}
        self.lock_run_series_store = Lock()
        self.run_series_context_store: dict[int, Context] = {}
        self.lock_run_series_context_store = Lock()
        self.automation_store: dict[int, AutomationRecord] = {}
        self.lock_automation_store = Lock()
        self._next_automation_id = 1
        self.task_store: dict[int, Task] = {}
        # Store task ID to token mapping
        self.task_token_store: dict[int, TokenRecord] = {}
        # Store token to task ID mapping
        self.task_token_to_task_id: dict[str, int] = {}
        self.task_logs: dict[int, list[tuple[float, str]]] = {}
        self.log_lock = Lock()
        self.task_usage_store: dict[int, TaskUsageRecord] = {}
        self.lock_task_usage_store = Lock()
        self._next_task_usage_id = 1
        self.lock_task_store = Lock()
        self.task_message_store: dict[str, Message] = {}
        self.lock_task_message_store = Lock()
        self.task_event_store: dict[int, list[TaskEvent]] = {}
        self.lock_task_event_store = Lock()
        self._next_task_event_id = 1
        self._object_push_sessions: dict[str, ObjectPushSession] = {}
        # Store root object ID to session ID mapping
        self._object_push_session_by_root: dict[str, str] = {}
        self._lock_object_push_sessions = RLock()

    @property
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""
        return self._object_store

    def start_session(self, run_id: int) -> str:
        """Start a run-scoped object push session."""
        session_id = str(uuid4())
        with self._lock_object_push_sessions:
            self._object_push_sessions[session_id] = ObjectPushSession(
                run_id=run_id,
                expires_at=now() + timedelta(seconds=OBJECT_PUSH_SESSION_TTL_SECONDS),
                root_object_ids=set(),
                pending_object_ids=set(),
            )
        return session_id

    def delete_sessions_in_run(self, run_id: int) -> None:
        """Delete all object push session bookkeeping for a run."""
        with self._lock_object_push_sessions:
            session_ids = [
                session_id
                for session_id, session in self._object_push_sessions.items()
                if session.run_id == run_id
            ]
            for session_id in session_ids:
                self._cleanup_push_session(session_id, cleanup_messages=False)

    def preregister_object_tree(
        self, object_tree: ObjectTree, session_id: str
    ) -> list[str]:
        """Preregister an object tree and record its missing objects."""
        with self._lock_object_push_sessions:
            session = self._object_push_sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown object push session: {session_id}")

            # Preregister the tree and collect its currently missing objects
            missing_objects = self.object_store.preregister(session.run_id, object_tree)

            # Remove bookkeeping for an older session owning the same root
            old_session_id = self._object_push_session_by_root.get(
                object_tree.object_id
            )
            if old_session_id is not None and old_session_id != session_id:
                self._cleanup_push_session(old_session_id, cleanup_messages=False)

            # Record root ownership and pending objects for the session
            session.root_object_ids.add(object_tree.object_id)
            session.pending_object_ids.update(missing_objects)
            self._object_push_session_by_root[object_tree.object_id] = session_id
            return missing_objects

    def store_object(
        self,
        run_id: int,
        session_id: str,
        object_id: str,
        object_content: bytes,
    ) -> bool:
        """Store an object if it is pending for an active push session."""
        with self._lock_object_push_sessions:
            # Support legacy SuperNodes that do not send a session ID
            if not session_id:
                sessions = self._object_push_sessions.items()
                session_id = next(
                    (
                        candidate_id
                        for candidate_id, candidate in sessions
                        if object_id in candidate.pending_object_ids
                    ),
                    "",
                )
                if not session_id:
                    return False

            # Validate session ownership and pending-object membership
            session = self._object_push_sessions.get(session_id)
            if (
                session is None
                or session.run_id != run_id
                or object_id not in session.pending_object_ids
            ):
                return False

            # Reject expired sessions and clean up their messages and objects
            if session.expires_at <= now():
                self._cleanup_push_session(session_id, cleanup_messages=True)
                return False

            # Store the object, mark it complete, and refresh the session TTL
            try:
                self.object_store.put(object_id, object_content)
            except Exception as err:  # pylint: disable=broad-exception-caught
                log(ERROR, "Failed to store object %s: %s", object_id, err)
                return False
            session.pending_object_ids.remove(object_id)
            session.expires_at = now() + timedelta(
                seconds=OBJECT_PUSH_SESSION_TTL_SECONDS
            )

            # Remove session bookkeeping once every pending object is stored
            if not session.pending_object_ids:
                self._cleanup_push_session(session_id, cleanup_messages=False)
            return True

    def get_object(self, run_id: int, object_id: str) -> bytes | None:
        """Get an object and clean up expired push sessions when needed."""
        with self._lock_object_push_sessions:
            # Return immediately unless the object is known but unavailable
            content = self.object_store.get(object_id)
            if content != b"":
                return content

            # Find expired sessions in this run that are waiting for the object
            current = now()
            expired_session_ids = [
                session_id
                for session_id, session in self._object_push_sessions.items()
                if session.run_id == run_id
                and object_id in session.pending_object_ids
                and session.expires_at <= current
            ]
            if not expired_session_ids:
                return content

            # Clean up every expired session, then return the resulting object state
            for session_id in expired_session_ids:
                self._cleanup_push_session(session_id, cleanup_messages=True)
            return self.object_store.get(object_id)

    def _cleanup_push_session(self, session_id: str, *, cleanup_messages: bool) -> None:
        """Remove an object push session and optionally its messages."""
        with self._lock_object_push_sessions:
            session = self._object_push_sessions.pop(session_id, None)
            if session is None:
                return

            # Remove root ownership entries still belonging to this session
            for object_id in session.root_object_ids:
                if self._object_push_session_by_root.get(object_id) == session_id:
                    del self._object_push_session_by_root[object_id]

        # Delete expired object trees and their message metadata
        if cleanup_messages and session.root_object_ids:
            for object_id in session.root_object_ids:
                self.object_store.delete(object_id)
            self._on_push_session_expired(session.root_object_ids)

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

    def upsert_connector(
        self,
        flwr_aid: str,
        connector_ref: str,
        credentials_json: str,
        config_json: str,
    ) -> bool:
        """Create or update a connector for an account."""
        if not flwr_aid or not connector_ref:
            return False
        connector = ConnectorRecord(
            flwr_aid=flwr_aid,
            connector_ref=connector_ref,
            credentials_json=credentials_json,
            config_json=config_json,
        )
        with self.lock_connector_store:
            self.connector_store[(flwr_aid, connector_ref)] = connector
        return True

    def get_connector(
        self, flwr_aid: str, connector_ref: str
    ) -> ConnectorRecord | None:
        """Return an account's connector, if present."""
        if not flwr_aid or not connector_ref:
            return None
        with self.lock_connector_store:
            return self.connector_store.get((flwr_aid, connector_ref))

    def delete_connector(self, flwr_aid: str, connector_ref: str) -> bool:
        """Delete an account's connector if it exists."""
        if not flwr_aid or not connector_ref:
            return False
        with self.lock_connector_store:
            return self.connector_store.pop((flwr_aid, connector_ref), None) is not None

    def bind_connectors_to_run(
        self, run_id: int, connector_refs: Sequence[str]
    ) -> bool:
        """Associate connector references with a run."""
        if isinstance(connector_refs, str):
            return False
        with self.lock_run_connector_store:
            self.run_connector_store.setdefault(run_id, set()).update(connector_refs)
        return True

    def get_run_connector_refs(self, run_id: int) -> Sequence[str]:
        """Return connector references associated with a run."""
        with self.lock_run_connector_store:
            return sorted(self.run_connector_store.get(run_id, set()))

    def create_connector_oauth_session(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        oauth_session_id: str,
        flwr_aid: str,
        connector_ref: str,
        state: str,
        redirect_uri: str,
        pkce_verifier: str | None,
        expires_at: datetime,
    ) -> ConnectorOAuthSessionRecord | None:
        """Create and return a connector OAuth session."""
        if (
            not oauth_session_id
            or not flwr_aid
            or not connector_ref
            or expires_at.utcoffset() is None
        ):
            return None
        expires_at = expires_at.astimezone(UTC)
        session = ConnectorOAuthSessionRecord(
            oauth_session_id=oauth_session_id,
            flwr_aid=flwr_aid,
            connector_ref=connector_ref,
            state=state,
            redirect_uri=redirect_uri,
            pkce_verifier=pkce_verifier,
            created_at=now().isoformat(),
            expires_at=expires_at.isoformat(),
            completed_at=None,
        )
        with self.lock_connector_oauth_session_store:
            if oauth_session_id in self.connector_oauth_session_store:
                return None
            self.connector_oauth_session_store[oauth_session_id] = session
        return session

    def get_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> ConnectorOAuthSessionRecord | None:
        """Return an account's connector OAuth session, if present."""
        if not oauth_session_id or not flwr_aid:
            return None
        with self.lock_connector_oauth_session_store:
            session = self.connector_oauth_session_store.get(oauth_session_id)
            if session is None or session.flwr_aid != flwr_aid:
                return None
            return session

    def complete_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> bool:
        """Mark a pending connector OAuth session as completed."""
        if not oauth_session_id or not flwr_aid:
            return False
        completed_at = now()
        with self.lock_connector_oauth_session_store:
            session = self.connector_oauth_session_store.get(oauth_session_id)
            if (
                session is None
                or session.flwr_aid != flwr_aid
                or session.completed_at is not None
                or datetime.fromisoformat(session.expires_at) <= completed_at
            ):
                return False
            self.connector_oauth_session_store[oauth_session_id] = replace(
                session, completed_at=completed_at.isoformat()
            )
        return True

    def get_run_series(
        self,
        *,
        series_ids: Sequence[int] | None = None,
        federation_ids: Sequence[str] | None = None,
        updated_before: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RunSeries]:
        """Return RunSeries metadata, optionally filtered by the given filters."""
        if limit is not None and limit < 0:
            raise AssertionError("`limit` must be >= 0")
        if (
            limit == 0
            or (series_ids is not None and not series_ids)
            or (federation_ids is not None and not federation_ids)
        ):
            return []

        series_id_set = set(series_ids) if series_ids is not None else None
        federation_id_set = set(federation_ids) if federation_ids is not None else None

        with self.lock_run_series_store:
            run_series = []
            for record in self.run_series_store.values():
                if series_id_set is not None and record.series_id not in series_id_set:
                    continue
                if (
                    federation_id_set is not None
                    and record.federation not in federation_id_set
                ):
                    continue
                if updated_before is not None and record.updated_at >= updated_before:
                    continue
                run_series.append(record)
            run_series.sort(key=lambda record: record.updated_at, reverse=True)
            if limit is not None:
                run_series = run_series[:limit]
            return list(run_series)

    def get_run_series_context(self, series_id: int) -> Context | None:
        """Return the shared Context for the specified RunSeries, if present."""
        with self.lock_run_series_context_store:
            return self.run_series_context_store.get(series_id)

    def set_run_series_context(self, series_id: int, context: Context) -> None:
        """Set the shared Context for the specified RunSeries."""
        with self.lock_run_series_context_store:
            self.run_series_context_store[series_id] = context

    def store_run_in_series(
        self,
        run_id: int,
        federation_id: str,
        series_id: int | None,
        description: str | None = None,
    ) -> int | None:
        """Store a run in a run series and return the series ID."""
        with self.lock_run_series_store:
            if series_id is not None:
                # Reuse only an existing run series owned by the requested federation.
                existing = self.run_series_store.get(series_id)
                if existing is None:
                    log(ERROR, "Run series %d not found", series_id)
                    return None
                if existing.federation != federation_id:
                    log(
                        ERROR,
                        "Run series %d belongs to federation %r, not %r",
                        series_id,
                        existing.federation,
                        federation_id,
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
                    federation=federation_id,
                    description=description if description is not None else "",
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                self.run_series_store[new_series_id] = run_series
                resolved_series_id = new_series_id

            # Store the membership last so callers only receive linked series IDs.
            if run_id in run_series.run_ids:
                return None
            run_series.run_ids.append(run_id)
            if series_id is not None:
                run_series.updated_at = now().isoformat()
            return resolved_series_id

    def store_automation(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        federation_id: str,
        flwr_aid: str,
        fab_id: str | None,
        fab_version: str | None,
        fab_hash: str | None,
        override_config: UserConfig,
        federation_config: SimulationConfig | None,
        primary_task_type: str,
        series_id: int,
        next_run_at: str,
        fixed_interval: int | None = None,
        max_runs: int | None = None,
    ) -> Automation:
        """Store an automation and return its metadata."""
        with self.lock_automation_store:
            current = now()
            automation_id = self._next_automation_id
            self._next_automation_id += 1
            automation = Automation(
                automation_id=automation_id,
                status=AutomationStatus.ACTIVE,
                federation=federation_id,
                series_id=series_id,
                flwr_aid=flwr_aid,
                created_at=current.isoformat(),
                updated_at=current.isoformat(),
                next_run_at=next_run_at,
                fixed_interval=fixed_interval,
                remaining_runs=max_runs,
            )

            self.automation_store[automation_id] = AutomationRecord(
                automation=automation,
                fab_id=fab_id,
                fab_version=fab_version,
                fab_hash=fab_hash,
                override_config=dict(override_config),
                federation_config=federation_config,
                primary_task_type=primary_task_type,
            )
            return automation

    def list_automations(  # pylint: disable=too-many-arguments
        self,
        *,
        federation: str | None = None,
        statuses: Sequence[str] | None = None,
        due_before: datetime | None = None,
        order_by: Literal["next_run_at", "updated_at"],
        limit: int | None = None,
    ) -> Sequence[Automation]:
        """Return automations matching the given filters."""
        if limit is not None and limit < 0:
            raise AssertionError("`limit` must be >= 0")
        if limit == 0 or (statuses is not None and not statuses):
            return []

        status_set = set(statuses) if statuses is not None else None
        cutoff = due_before.isoformat() if due_before is not None else None
        with self.lock_automation_store:
            automations: list[Automation] = []
            for record in self.automation_store.values():
                automation = record.automation

                # Apply federation filter.
                if federation is not None and automation.federation != federation:
                    continue

                # Apply status filter.
                if status_set is not None and automation.status not in status_set:
                    continue

                # Apply due time filter.
                if cutoff is not None and automation.next_run_at > cutoff:
                    continue
                # Finite automations with no remaining runs are already claimed.
                if (
                    cutoff is not None
                    and automation.HasField("remaining_runs")
                    and automation.remaining_runs == 0
                ):
                    continue

                automations.append(automation)

            if order_by == "updated_at":
                automations.sort(
                    key=lambda automation: automation.updated_at,
                    reverse=True,
                )
            else:
                automations.sort(key=lambda automation: automation.next_run_at)
            if limit is not None:
                automations = automations[:limit]
            return automations

    def stop_automation(self, automation_id: int) -> bool:
        """Stop an active automation."""
        with self.lock_automation_store:
            record = self.automation_store.get(automation_id)
            if record is None or record.automation.status != AutomationStatus.ACTIVE:
                return False

            stopped_at = now().isoformat()
            record.automation.status = AutomationStatus.STOPPED
            record.automation.updated_at = stopped_at
            record.automation.stopped_at = stopped_at
            return True

    def advance_automation(
        self,
        automation_id: int,
        *,
        previous_next_run_at: str,
        next_run_at: str | None,
    ) -> bool:
        """Advance an active automation occurrence."""
        with self.lock_automation_store:
            record = self.automation_store.get(automation_id)
            if (
                record is None
                or record.automation.status != AutomationStatus.ACTIVE
                or record.automation.next_run_at != previous_next_run_at
                or (
                    record.automation.HasField("remaining_runs")
                    and record.automation.remaining_runs == 0
                )
            ):
                return False

            if next_run_at is None and (
                not record.automation.HasField("remaining_runs")
                or record.automation.remaining_runs > 1
            ):
                return False

            record.automation.updated_at = now().isoformat()

            if record.automation.HasField("remaining_runs"):
                record.automation.remaining_runs = max(
                    record.automation.remaining_runs - 1, 0
                )

            if next_run_at is not None:
                record.automation.next_run_at = next_run_at
            return True

    def finish_automation(
        self,
        automation_id: int,
        *,
        status: Literal[AutomationStatus.COMPLETED, AutomationStatus.FAILED],
    ) -> bool:
        """Finish an active automation with a terminal status."""
        with self.lock_automation_store:
            record = self.automation_store.get(automation_id)
            if record is None or record.automation.status != AutomationStatus.ACTIVE:
                return False
            if status == AutomationStatus.COMPLETED and (
                not record.automation.HasField("remaining_runs")
                or record.automation.remaining_runs != 0
            ):
                return False

            record.automation.status = status
            record.automation.updated_at = now().isoformat()
            return True

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

    def add_task_usage(self, task_id: int, usage: TaskUsage) -> None:
        """Record usage for the specified task."""
        with self.lock_task_store:
            task = self.task_store.get(task_id)
            if task is None:
                return
            run_id = task.run_id

        with self.lock_task_usage_store:
            usage_id = self._next_task_usage_id
            self.task_usage_store[usage_id] = TaskUsageRecord(
                id=usage_id,
                task_id=task_id,
                run_id=run_id,
                usage=usage,
                created_at=now(),
                reported_at=None,
            )
            self._next_task_usage_id += 1

    def get_task_usage(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        task_ids: Sequence[int] | None = None,
    ) -> Sequence[TaskUsage]:
        """Retrieve task usage records based on the specified filters."""
        if (run_ids is not None and not run_ids) or (
            task_ids is not None and not task_ids
        ):
            return []

        with self.lock_task_usage_store:
            records = sorted(
                self.task_usage_store.values(), key=lambda record: record.id
            )
            if run_ids is not None:
                records = [record for record in records if record.run_id in run_ids]
            if task_ids is not None:
                records = [record for record in records if record.task_id in task_ids]
            return [record.usage for record in records]

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

            activated_at = now()
            task.running_at = activated_at.isoformat()
            task.status.CopyFrom(
                TaskStatus(status=Status.RUNNING, sub_status="", details="")
            )
            record = self.task_token_store.get(task_id)
            if record is not None:
                record.active_until = activated_at + timedelta(
                    seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
                )
            return True

    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished."""
        if sub_status not in (SubStatus.COMPLETED, SubStatus.STOPPED, SubStatus.FAILED):
            err = f"Invalid sub_status '{sub_status}' for finishing task {task_id}"
            log(ERROR, err)
            return False

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
        Expired starting tasks are moved back to pending. Expired running tasks
        are marked as finished with a failed status. Tokens are removed in both
        cases.
        """
        current = now()
        expired_tasks: list[Task] = []
        for task_id, record in list(self.task_token_store.items()):
            if record.active_until < current:
                task = self.task_store.get(task_id)
                if task and task.status.status == Status.STARTING:
                    task.starting_at = ""
                    task.status.CopyFrom(
                        TaskStatus(status=Status.PENDING, sub_status="", details="")
                    )
                elif task and task.status.status == Status.RUNNING:
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
