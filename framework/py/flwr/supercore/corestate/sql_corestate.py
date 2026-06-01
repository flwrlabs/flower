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
"""SQLAlchemy-based CoreState implementation."""


import hashlib
import json
import secrets
from collections.abc import Sequence
from datetime import datetime, timedelta
from logging import ERROR
from typing import Any, Literal, cast

from sqlalchemy import MetaData
from sqlalchemy.exc import IntegrityError

from flwr.app import Context, Message
from flwr.app.message import make_message
from flwr.app.metadata import Metadata
from flwr.common import now
from flwr.common.constant import (
    FLWR_TASK_TOKEN_LENGTH,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    SERIES_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    TASK_ID_NUM_BYTES,
    Status,
    SubStatus,
)
from flwr.common.logger import log
from flwr.common.serde import recorddict_from_proto, recorddict_to_proto
from flwr.common.serde_utils import error_from_proto, error_to_proto
from flwr.common.typing import Fab
from flwr.proto.error_pb2 import Error as ProtoError  # pylint: disable=E0611

# pylint: disable-next=E0611
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent, TaskStatus  # pylint: disable=E0611
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64

from ..object_store import ObjectStore
from .corestate import CoreState
from .utils import (
    context_from_bytes,
    context_to_bytes,
    generate_rand_int_from_bytes,
    timestamp_to_iso,
    validate_task_event_data,
    validate_task_message,
)

# Define SQL conditions for task statuses to ensure consistency across queries
STATUS_CONDITIONS = {
    Status.PENDING: "(starting_at IS NULL AND finished_at IS NULL)",
    Status.STARTING: "(starting_at IS NOT NULL AND running_at IS NULL "
    "AND finished_at IS NULL)",
    Status.RUNNING: "(running_at IS NOT NULL AND finished_at IS NULL)",
    Status.FINISHED: "(finished_at IS NOT NULL)",
}


class SqlCoreState(CoreState, SqlMixin):  # pylint: disable=R0904
    """SQLAlchemy-based CoreState implementation."""

    def __init__(self, database_path: str, object_store: ObjectStore) -> None:
        super().__init__(database_path)
        self._object_store = object_store

    @property
    def select_lock_sql(self) -> str:
        """Return the SQL clause for row-locking selected candidates."""
        return ""

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
        params = {
            "fab_hash": fab_hash,
            "content": fab.content,
            "verifications": json.dumps(fab.verifications),
        }
        # Keep launch behavior: last write wins for metadata under the same
        # content hash.
        query = """
            INSERT INTO fab (fab_hash, content, verifications)
            VALUES (:fab_hash, :content, :verifications)
            ON CONFLICT(fab_hash) DO UPDATE SET
                content = excluded.content,
                verifications = excluded.verifications
        """
        self.query(query, params)
        return fab_hash

    def get_fab(self, fab_hash: str) -> Fab | None:
        """Return a FAB by hash."""
        query = """
            SELECT fab_hash, content, verifications
            FROM fab
            WHERE fab_hash = :fab_hash
        """
        rows = self.query(query, {"fab_hash": fab_hash})
        if not rows:
            return None
        row = rows[0]
        # Launch tradeoff: do not recompute content hash on reads; rely on
        # write-time validation and hash-addressed lookup.
        return Fab(
            hash_str=row["fab_hash"],
            content=row["content"],
            verifications=json.loads(row["verifications"]),
        )

    def get_run_series(  # pylint: disable=R0914
        self,
        *,
        series_ids: Sequence[int] | None = None,
        federations: Sequence[str] | None = None,
        updated_before: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RunSeries]:
        """Return RunSeries metadata, optionally filtered by the given filters."""
        # Validate limit before building the SQL query.
        if limit is not None and limit < 0:
            raise ValueError("`limit` must be >= 0")
        if (
            limit == 0
            or (series_ids is not None and not series_ids)
            or (federations is not None and not federations)
        ):
            return []

        # Build optional filters for the run-series page.
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if series_ids is not None:
            sint64_series_ids = [uint64_to_int64(series_id) for series_id in series_ids]
            placeholders = ",".join(
                [f":sid_{i}" for i in range(len(sint64_series_ids))]
            )
            conditions.append(f"series_id IN ({placeholders})")
            params.update(
                {f"sid_{i}": series_id for i, series_id in enumerate(sint64_series_ids)}
            )
        if federations is not None:
            placeholders = ",".join([f":fed_{i}" for i in range(len(federations))])
            conditions.append(f"federation IN ({placeholders})")
            params.update({f"fed_{i}": fed for i, fed in enumerate(federations)})
        if updated_before is not None:
            conditions.append("updated_at < :updated_before")
            params["updated_before"] = datetime.fromisoformat(updated_before)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT :limit"
            params["limit"] = limit

        # Select the requested page before joining run IDs so limit applies to series.
        run_series_cte = f"""
            run_series_cte AS (
                SELECT series_id, federation, description, created_at, updated_at
                FROM run_series
                {where_clause}
                ORDER BY updated_at DESC
                {self.select_lock_sql}
                {limit_clause}
            )
        """
        query = f"""
            WITH {run_series_cte}
            SELECT
                run_series_cte.*,
                series_runs.run_id
            FROM run_series_cte
            LEFT JOIN series_runs
                ON series_runs.series_id = run_series_cte.series_id
        """
        rows = self.query(query, params)
        # Fold the joined rows back into one RunSeries per series.
        series_by_id: dict[int, RunSeries] = {}
        for row in rows:
            series_id = row["series_id"]
            if series_id not in series_by_id:
                series_by_id[series_id] = _run_series_from_row(row)
            if row["run_id"] is not None:
                series_by_id[series_id].run_ids.append(int64_to_uint64(row["run_id"]))
        return list(series_by_id.values())

    def get_run_series_context(self, series_id: int) -> Context | None:
        """Return the shared Context for the specified RunSeries, if present."""
        rows = self.query(
            """
            SELECT context
            FROM series_context
            WHERE series_id = :series_id
            """,
            {"series_id": uint64_to_int64(series_id)},
        )
        if not rows or rows[0]["context"] is None:
            return None
        return context_from_bytes(rows[0]["context"])

    def set_run_series_context(self, series_id: int, context: Context) -> None:
        """Set the shared Context for the specified RunSeries."""
        sint_series_id = uint64_to_int64(series_id)
        context_bytes = context_to_bytes(context)
        with self.session():
            self.query(
                """
                INSERT INTO series_context (series_id, context)
                VALUES (:series_id, :context)
                ON CONFLICT(series_id) DO UPDATE SET
                    context = excluded.context
                """,
                {"series_id": sint_series_id, "context": context_bytes},
            )

    def store_run_in_series(
        self,
        run_id: int,
        federation: str,
        series_id: int | None,
    ) -> int | None:
        """Store a run in a run series and return the series ID."""
        insert_query = """
            INSERT INTO run_series
            (series_id, federation, description, created_at, updated_at)
            VALUES
            (:series_id, :federation, :description, :created_at, :updated_at)
            ON CONFLICT(series_id) DO NOTHING
            RETURNING series_id
        """

        try:
            with self.session():
                if series_id is None:
                    # No series was provided, so create one before linking the run.
                    candidate = generate_rand_int_from_bytes(SERIES_ID_NUM_BYTES)
                    timestamp = now()
                    rows = self.query(
                        insert_query,
                        {
                            "series_id": uint64_to_int64(candidate),
                            "federation": federation,
                            "description": None,
                            "created_at": timestamp,
                            "updated_at": timestamp,
                        },
                    )
                    if rows:
                        resolved_series_id = candidate
                    else:
                        return None

                else:
                    rows = self.query(
                        """
                        UPDATE run_series
                        SET updated_at = :updated_at
                        WHERE series_id = :series_id AND federation = :federation
                        RETURNING series_id
                        """,
                        {
                            "series_id": uint64_to_int64(series_id),
                            "federation": federation,
                            "updated_at": now(),
                        },
                    )
                    if not rows:
                        log(
                            ERROR,
                            "Run series %d not found in federation %r",
                            series_id,
                            federation,
                        )
                        return None
                    resolved_series_id = series_id

                # Store the membership last so callers only receive linked series IDs.
                self.query(
                    """
                    INSERT INTO series_runs (series_id, run_id)
                    VALUES (:series_id, :run_id)
                    """,
                    {
                        "series_id": uint64_to_int64(resolved_series_id),
                        "run_id": uint64_to_int64(run_id),
                    },
                )
                return resolved_series_id
        except IntegrityError:
            return None

    def add_task_log(self, task_id: int, log_message: str) -> None:
        """Add a log entry to the task logs for the specified `task_id`."""
        sint64_task_id = uint64_to_int64(task_id)

        try:
            self.query(
                """
                INSERT INTO task_logs (timestamp, task_id, log)
                VALUES (:current_ts, :task_id, :log)
                """,
                {
                    "current_ts": now().timestamp(),
                    "task_id": sint64_task_id,
                    "log": log_message,
                },
            )
        except IntegrityError:
            raise ValueError(f"Task {task_id} not found") from None

    def get_task_log(
        self, task_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get task logs for the specified `task_id`."""
        sint64_task_id = uint64_to_int64(task_id)

        # We don't check if the task exists before querying logs
        # because the task_id is validated by the authz layer

        if after_timestamp is None:
            after_timestamp = 0.0

        # Polling is strict-after: entries at the checkpoint timestamp have
        # already been delivered.
        rows = self.query(
            """
            SELECT log, timestamp FROM task_logs
            WHERE task_id = :task_id AND timestamp > :after_timestamp
            ORDER BY timestamp
            """,
            {"task_id": sint64_task_id, "after_timestamp": after_timestamp},
        )
        latest_timestamp = rows[-1]["timestamp"] if rows else 0.0
        return "".join(row["log"] for row in rows), latest_timestamp

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
        task_id = generate_rand_int_from_bytes(TASK_ID_NUM_BYTES)
        sint64_task_id = uint64_to_int64(task_id)

        insert_query = """
            INSERT INTO task
            (task_id, type, run_id, fab_hash, model_ref, connector_ref, token,
             active_until, pending_at, starting_at, running_at, finished_at,
             sub_status, details)
            SELECT
             :task_id, :type, :run_id, :fab_hash, :model_ref, :connector_ref, :token,
             :active_until, :pending_at, :starting_at, :running_at, :finished_at,
             :sub_status, :details
            WHERE :requesting_task_id IS NULL
            OR EXISTS (
                SELECT 1
                FROM task
                WHERE task_id = :requesting_task_id
                AND finished_at IS NULL
            )
            RETURNING task_id;
        """

        params = {
            "task_id": sint64_task_id,
            "type": task_type,
            "run_id": uint64_to_int64(run_id),
            "fab_hash": fab_hash,
            "model_ref": model_ref,
            "connector_ref": connector_ref,
            "token": None,
            "active_until": None,
            "pending_at": now(),
            "starting_at": None,
            "running_at": None,
            "finished_at": None,
            "sub_status": "",
            "details": "",
            "requesting_task_id": (
                uint64_to_int64(requesting_task_id)
                if requesting_task_id is not None
                else None
            ),
        }

        with self.session():
            try:
                rows = self.query(insert_query, params)
                return task_id if rows else None
            except IntegrityError:
                return None

    def get_tasks(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
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

        conditions = []
        params: dict[str, Any] = {}

        if task_ids is not None:
            if not task_ids:
                return []
            sint64_task_ids = [uint64_to_int64(task_id) for task_id in task_ids]
            placeholders = ",".join([f":tid_{i}" for i in range(len(sint64_task_ids))])
            conditions.append(f"task_id IN ({placeholders})")
            params.update(
                {f"tid_{i}": task_id for i, task_id in enumerate(sint64_task_ids)}
            )

        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            placeholders = ",".join([f":rid_{i}" for i in range(len(sint64_run_ids))])
            conditions.append(f"run_id IN ({placeholders})")
            params.update(
                {f"rid_{i}": run_id for i, run_id in enumerate(sint64_run_ids)}
            )

        if statuses is not None:
            if not statuses:
                return []
            status_conditions = []
            for status, condition in STATUS_CONDITIONS.items():
                if status in statuses:
                    status_conditions.append(condition)
            if not status_conditions:
                return []
            conditions.append(f"({' OR '.join(status_conditions)})")

        query = """
            SELECT task_id, type, run_id, fab_hash, model_ref, connector_ref,
                   pending_at, starting_at, running_at, finished_at,
                   sub_status, details
            FROM task
        """
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        if order_by is not None:
            query += f" ORDER BY {order_by} {'ASC' if ascending else 'DESC'}"
        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        with self.session():
            # Clean up expired task tokens before querying tasks
            self._cleanup_expired_task_tokens()
            rows = self.query(query, params)

        result: list[Task] = []
        for row in rows:
            result.append(task_from_row(row))
        return result

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData needed for CoreState tables."""
        return create_corestate_metadata()

    def claim_task(self, task_id: int) -> str | None:
        """Atomically claim a pending task."""
        token = secrets.token_hex(FLWR_TASK_TOKEN_LENGTH)
        claimed_at = now()
        active_until = claimed_at + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL)
        sint64_task_id = uint64_to_int64(task_id)
        try:
            # The conditional UPDATE is the atomic claim: exactly one caller can
            # move a pending, unclaimed task to STARTING and attach a token.
            rows = self.query(
                f"""
                UPDATE task
                SET token = :token,
                    active_until = :active_until,
                    starting_at = :starting_at
                WHERE task_id = :task_id AND token IS NULL
                AND {STATUS_CONDITIONS[Status.PENDING]}
                RETURNING task_id
                """,
                {
                    "task_id": sint64_task_id,
                    "token": token,
                    "active_until": active_until,
                    "starting_at": claimed_at,
                },
            )
            if not rows:
                return None

            return token
        except IntegrityError:
            # Rare failure: generated token already exists (duplicate)
            return None

    def activate_task(self, task_id: int) -> bool:
        """Move a task from starting to running."""
        # Expire non-responsive tasks before transitioning task status.

        with self.session():
            self._cleanup_expired_task_tokens()

            # Activation is a strict STARTING -> RUNNING transition.
            rows = self.query(
                f"""
                UPDATE task
                SET running_at = :running_at
                WHERE task_id = :task_id AND {STATUS_CONDITIONS[Status.STARTING]}
                RETURNING task_id
                """,
                {"task_id": uint64_to_int64(task_id), "running_at": now()},
            )
        return len(rows) > 0

    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished."""
        if sub_status not in (SubStatus.COMPLETED, SubStatus.STOPPED, SubStatus.FAILED):
            err = f"Invalid sub_status '{sub_status}' for finishing task {task_id}"
            log(ERROR, err)
            return False

        sint64_task_id = uint64_to_int64(task_id)
        with self.session():
            self._cleanup_expired_task_tokens()
            # FINISHED:COMPLETED is only valid from RUNNING.
            completion_constraint = ""
            if sub_status == SubStatus.COMPLETED:
                completion_constraint = "AND running_at IS NOT NULL"

            rows = self.query(
                f"""
                UPDATE task
                SET finished_at = :finished_at,
                    sub_status = :sub_status,
                    details = :details,
                    active_until = NULL,
                    token = NULL
                WHERE task_id = :task_id
                AND finished_at IS NULL {completion_constraint}
                RETURNING task_id
                """,
                {
                    "task_id": sint64_task_id,
                    "finished_at": now(),
                    "sub_status": sub_status,
                    "details": details,
                },
            )
            if not rows:
                return False

            return True

    def acknowledge_task_heartbeat(self, task_id: int) -> bool:
        """Extend heartbeat state for the claimed task."""
        # Heartbeats are accepted only for active, unexpired task claims.
        with self.session():
            current = now()
            ttl = timedelta(seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL)
            self._cleanup_expired_task_tokens()
            rows = self.query(
                """
                UPDATE task
                SET active_until = :active_until
                WHERE task_id = :task_id
                AND active_until >= :current
                AND finished_at IS NULL
                RETURNING task_id
                """,
                {
                    "task_id": uint64_to_int64(task_id),
                    "current": current,
                    "active_until": current + ttl,
                },
            )
        return len(rows) > 0

    def get_task_by_token(self, token: str) -> Task | None:
        """Return the task associated with the task token, if valid."""
        rows = self.query(
            """
            SELECT * FROM task
            WHERE token = :token AND active_until >= :current AND finished_at IS NULL
            """,
            {"token": token, "current": now()},
        )
        if not rows:
            return None
        return task_from_row(rows[0])

    def store_task_message(self, message: Message) -> bool:
        """Store one task-addressed Message."""
        if validate_task_message(message):
            return False

        with self.session():
            self._cleanup_expired_task_tokens()
            message_dict = _task_message_to_row(message)
            try:
                inserted = self.query(
                    """
                    INSERT INTO task_message (
                        message_id, run_id, src_task_id, dst_task_id,
                        reply_to_message_id, created_at, ttl, message_type,
                        content, error
                    )
                    SELECT
                        :message_id, :run_id, :src_task_id, :dst_task_id,
                        :reply_to_message_id, :created_at, :ttl, :message_type,
                        :content, :error
                    FROM task AS src
                    JOIN task AS dst
                        ON dst.task_id = :dst_task_id
                    WHERE src.task_id = :src_task_id
                        AND src.run_id = :run_id
                        AND dst.run_id = :run_id
                        AND dst.finished_at IS NULL
                    RETURNING message_id
                    """,
                    message_dict,
                )
            except IntegrityError:
                return False
            return bool(inserted)

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

        with self.session():
            self._cleanup_expired_task_tokens()
            self._cleanup_invalid_task_messages()
            rows = self._claim_task_message_rows(dst_task_ids, order_by, limit)

        return [_task_message_from_row(row) for row in rows]

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

        current = now()
        params = [
            {
                "timestamp": current,
                "run_id": uint64_to_int64(event.run_id),
                "task_id": uint64_to_int64(event.task_id),
                "event": event.event,
                "data": event.data,
            }
            for event in events
        ]

        with self.session():
            self.query(
                """
                INSERT INTO task_event (timestamp, run_id, task_id, event, data)
                VALUES (:timestamp, :run_id, :task_id, :event, :data)
                """,
                params,
            )

        return True

    def get_task_events(
        self,
        *,
        run_id: int | None = None,
        after_task_event_id: int | None = None,
    ) -> Sequence[TaskEvent]:
        """Return task-produced run events after the cursor."""
        cursor = after_task_event_id if after_task_event_id is not None else 0
        conditions = ["id > :after_task_event_id"]
        params = {"after_task_event_id": cursor}
        if run_id is not None:
            conditions.append("run_id = :run_id")
            params["run_id"] = uint64_to_int64(run_id)

        rows = self.query(
            f"""
            SELECT id, timestamp, run_id, task_id, event, data
            FROM task_event
            WHERE {" AND ".join(conditions)}
            ORDER BY id ASC
            """,
            params,
        )

        return [
            TaskEvent(
                id=row["id"],
                timestamp=timestamp_to_iso(row["timestamp"]),
                run_id=int64_to_uint64(row["run_id"]),
                task_id=int64_to_uint64(row["task_id"]),
                event=row["event"],
                data=row["data"],
            )
            for row in rows
        ]

    def _claim_task_message_rows(
        self,
        dst_task_ids: Sequence[int] | None,
        order_by: Literal["created_at"] | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        """Atomically claim eligible task Messages."""
        conditions: list[str] = []
        params: dict[str, Any] = {}

        # Filter by destination task IDs
        if dst_task_ids is not None:
            sint64_dst_task_ids = [uint64_to_int64(t) for t in dst_task_ids]
            placeholders = ",".join(
                f":dtid_{i}" for i in range(len(sint64_dst_task_ids))
            )
            conditions.append(f"dst_task_id IN ({placeholders})")
            params.update(
                {f"dtid_{i}": tid for i, tid in enumerate(sint64_dst_task_ids)}
            )

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        order_clause = f"ORDER BY {order_by}" if order_by else ""
        limit_clause = "LIMIT :limit" if limit is not None else ""

        if limit is not None:
            params["limit"] = limit

        if order_by is not None or limit is not None:
            # Materialize candidates before deleting. Some backends can otherwise
            # re-evaluate same-table subqueries while DELETE scans rows.
            # `self.select_lock_sql` is an optional clause for backends that support
            # row-locking while selecting candidates. Keep it before LIMIT so locked
            # rows are skipped before limiting the result set.
            query = f"""
                WITH selected AS (
                    SELECT message_id
                    FROM task_message
                    {where_clause} {order_clause}
                    {self.select_lock_sql}
                    {limit_clause}
                )
                DELETE FROM task_message
                WHERE message_id IN (SELECT message_id FROM selected)
                RETURNING *
            """
        else:
            query = f"""
                DELETE FROM task_message
                {where_clause}
                RETURNING *
            """

        rows = self.query(query, params)

        # Sort claimed rows in-memory if requested
        # `ORDER BY` in the CTE determines which rows are claimed, but SQL does not
        # guarantee that `DELETE ... RETURNING` returns them in that order.
        if order_by is not None:
            rows.sort(key=lambda row: row[order_by])

        return rows

    def _cleanup_expired_task_tokens(self) -> None:
        """Remove expired task heartbeat records.

        Expired tasks are marked as finished with a failed status, and their tokens are
        removed.
        """
        expired_at = now()
        # Expired task claims are terminal failures and lose their token.
        rows = self.query(
            """
            UPDATE task
            SET token = NULL, finished_at = active_until, active_until = NULL,
                sub_status = :sub_status, details = :details
            WHERE token IS NOT NULL AND active_until < :current
            AND finished_at IS NULL
            RETURNING task_id, type, run_id, fab_hash, model_ref, connector_ref,
                      pending_at, starting_at, running_at, finished_at,
                      sub_status, details
            """,
            {
                "current": expired_at,
                "sub_status": SubStatus.FAILED,
                "details": "No heartbeat received from the task",
            },
        )
        if rows:
            self._on_task_tokens_expired([task_from_row(row) for row in rows])

    def _cleanup_invalid_task_messages(self) -> None:
        """Remove expired Messages and Messages for invalid destination tasks."""
        self.query(
            """
            DELETE FROM task_message
            WHERE (created_at + ttl) <= :current
            """,
            {"current": now().timestamp()},
        )

    def _on_task_tokens_expired(self, tasks: list[Task]) -> None:
        """Handle cleanup of expired task tokens.

        Override in subclasses to add custom cleanup logic.

        Parameters
        ----------
        tasks : list[Task]
            Tasks whose claims expired and were marked FINISHED:FAILED.
        """

    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Atomically reserve a nonce in a namespace."""
        if namespace == "" or nonce == "":
            return False
        cleanup_query = """
            DELETE FROM nonce_store
            WHERE expires_at < :current;
        """
        insert_query = """
            INSERT INTO nonce_store (namespace, nonce, expires_at)
            VALUES (:namespace, :nonce, :expires_at);
        """
        self.query(cleanup_query, {"current": now().timestamp()})
        try:
            self.query(
                insert_query,
                {"namespace": namespace, "nonce": nonce, "expires_at": expires_at},
            )
            return True
        # Duplicate nonce detected, treated as a replay attempt.
        # IntegrityError can only arise from (namespace, nonce) uniqueness.
        except IntegrityError:
            return False


def determine_task_status(row: dict[str, Any]) -> TaskStatus:
    """Determine the status of the task based on timestamp fields."""
    if row["pending_at"]:
        if row["finished_at"]:
            return TaskStatus(
                status=Status.FINISHED,
                sub_status=row["sub_status"],
                details=row["details"],
            )
        if row["starting_at"]:
            if row["running_at"]:
                return TaskStatus(status=Status.RUNNING, sub_status="", details="")
            return TaskStatus(status=Status.STARTING, sub_status="", details="")
        return TaskStatus(status=Status.PENDING, sub_status="", details="")
    task_id = int64_to_uint64(row["task_id"])
    raise ValueError(f"The task {task_id} does not have a valid status.")


def task_from_row(row: dict[str, Any]) -> Task:
    """Convert a database row to a Task object."""
    return Task(
        task_id=int64_to_uint64(row["task_id"]),
        type=row["type"],
        run_id=int64_to_uint64(row["run_id"]),
        pending_at=timestamp_to_iso(row["pending_at"]),
        starting_at=timestamp_to_iso(row["starting_at"]),
        running_at=timestamp_to_iso(row["running_at"]),
        finished_at=timestamp_to_iso(row["finished_at"]),
        status=determine_task_status(row),
        fab_hash=row["fab_hash"],
        model_ref=row["model_ref"],
        connector_ref=row["connector_ref"],
    )


def _run_series_from_row(row: dict[str, Any]) -> RunSeries:
    """Convert a database row to a RunSeries object."""
    return RunSeries(
        series_id=int64_to_uint64(row["series_id"]),
        federation=row["federation"],
        description=row["description"] or "",
        created_at=timestamp_to_iso(row["created_at"]),
        updated_at=timestamp_to_iso(row["updated_at"]),
    )


def _task_message_to_row(message: Message) -> dict[str, Any]:
    """Convert a task-addressed Message to database row values."""
    return {
        "message_id": message.metadata.message_id,
        "run_id": uint64_to_int64(message.metadata.run_id),
        "src_task_id": uint64_to_int64(cast(int, message.metadata.src_task_id)),
        "dst_task_id": uint64_to_int64(cast(int, message.metadata.dst_task_id)),
        "reply_to_message_id": message.metadata.reply_to_message_id,
        "created_at": message.metadata.created_at,
        "ttl": message.metadata.ttl,
        "message_type": message.metadata.message_type,
        "content": (
            recorddict_to_proto(message.content).SerializeToString()
            if message.has_content()
            else None
        ),
        "error": (
            error_to_proto(message.error).SerializeToString()
            if message.has_error()
            else None
        ),
    }


def _task_message_from_row(row: dict[str, Any]) -> Message:
    """Convert a task_message row to a Message."""
    content, error = None, None
    if row["content"] is not None:
        content = recorddict_from_proto(ProtoRecordDict.FromString(row["content"]))
    if row["error"] is not None:
        error = error_from_proto(ProtoError.FromString(row["error"]))

    metadata = Metadata(
        run_id=int64_to_uint64(row["run_id"]),
        message_id=row["message_id"],
        src_node_id=SUPERLINK_NODE_ID,
        dst_node_id=SUPERLINK_NODE_ID,
        reply_to_message_id=row["reply_to_message_id"] or "",
        group_id="",  # Task messages don't have this field for now
        created_at=row["created_at"],
        ttl=row["ttl"],
        message_type=row["message_type"],
        src_task_id=int64_to_uint64(row["src_task_id"]),
        dst_task_id=int64_to_uint64(row["dst_task_id"]),
    )
    return make_message(metadata=metadata, content=content, error=error)
