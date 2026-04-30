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
from typing import Any, Literal, cast

from sqlalchemy import MetaData, text
from sqlalchemy.exc import IntegrityError

from flwr.common import now
from flwr.common.constant import (
    FLWR_APP_TOKEN_LENGTH,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    TASK_ID_NUM_BYTES,
    Status,
    SubStatus,
)
from flwr.common.typing import Fab
from flwr.proto.task_pb2 import Task, TaskStatus  # pylint: disable=E0611
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64

from ..object_store import ObjectStore
from .corestate import CoreState
from .utils import generate_rand_int_from_bytes

# Define SQL conditions for task statuses to ensure consistency across queries
STATUS_CONDITIONS = {
    Status.PENDING: "(starting_at IS NULL AND finished_at IS NULL)",
    Status.STARTING: "(starting_at IS NOT NULL AND running_at IS NULL "
    "AND finished_at IS NULL)",
    Status.RUNNING: "(running_at IS NOT NULL AND finished_at IS NULL)",
    Status.FINISHED: "(finished_at IS NOT NULL)",
}


class SqlCoreState(CoreState, SqlMixin):
    """SQLAlchemy-based CoreState implementation."""

    def __init__(self, database_path: str, object_store: ObjectStore) -> None:
        super().__init__(database_path)
        self._object_store = object_store

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

    def create_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        task_type: str,
        run_id: int,
        fab_hash: str | None = None,
        model_ref: str | None = None,
        connector_ref: str | None = None,
    ) -> int | None:
        """Create a task and return its ID."""
        task_id = generate_rand_int_from_bytes(TASK_ID_NUM_BYTES)
        sint64_task_id = uint64_to_int64(task_id)

        insert_query = """
            INSERT INTO task
            (task_id, type, run_id, fab_hash, model_ref, connector_ref, token,
             active_until, pending_at, starting_at, running_at, finished_at,
             sub_status, details)
            VALUES
            (:task_id, :type, :run_id, :fab_hash, :model_ref, :connector_ref, :token,
             :active_until, :pending_at, :starting_at, :running_at, :finished_at,
             :sub_status, :details);
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
            "pending_at": now().isoformat(),
            "starting_at": None,
            "running_at": None,
            "finished_at": None,
            "sub_status": "",
            "details": "",
        }

        with self.session():
            try:
                self.query(insert_query, params)
                return task_id
            except IntegrityError:
                return None

    def get_tasks(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
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

        if statuses is not None:
            if not statuses:
                return []
            status_conditions = []
            if "pending" in statuses:
                status_conditions.append("starting_at IS NULL AND finished_at IS NULL")
            if "starting" in statuses:
                status_conditions.append(
                    "starting_at IS NOT NULL AND running_at IS NULL "
                    "AND finished_at IS NULL"
                )
            if "running" in statuses:
                status_conditions.append(
                    "running_at IS NOT NULL AND finished_at IS NULL"
                )
            if "finished" in statuses:
                status_conditions.append("finished_at IS NOT NULL")
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

        rows = self.query(query, params)

        result: list[Task] = []
        for row in rows:
            task = Task(
                task_id=int64_to_uint64(row["task_id"]),
                type=row["type"],
                run_id=int64_to_uint64(row["run_id"]),
                pending_at=row["pending_at"],
                starting_at=row["starting_at"],
                running_at=row["running_at"],
                finished_at=row["finished_at"],
                status=determine_task_status(row),
                fab_hash=row["fab_hash"],
                model_ref=row["model_ref"],
                connector_ref=row["connector_ref"],
            )
            result.append(task)
        return result

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData needed for CoreState tables."""
        return create_corestate_metadata()

    def claim_task(self, task_id: int) -> str | None:
        """Atomically claim a pending task."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)
        claimed_at = now()
        active_until = int(claimed_at.timestamp()) + HEARTBEAT_DEFAULT_INTERVAL
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
                    "starting_at": claimed_at.isoformat(),
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
                {"task_id": uint64_to_int64(task_id), "running_at": now().isoformat()},
            )
        return len(rows) > 0

    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished."""
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
                    "finished_at": now().isoformat(),
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
            current = int(now().timestamp())
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
                    "active_until": (
                        current + HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
                    ),
                },
            )
        return len(rows) > 0

    def get_task_id_by_token(self, token: str) -> int | None:
        """Return the task ID associated with the task token, if valid."""
        rows = self.query(
            """
            SELECT task_id FROM task
            WHERE token = :token AND active_until >= :current AND finished_at IS NULL
            """,
            {"token": token, "current": int(now().timestamp())},
        )
        if not rows:
            return None
        return int64_to_uint64(rows[0]["task_id"])

    def _cleanup_expired_task_tokens(self) -> None:
        """Remove expired task heartbeat records.

        Expired tasks are marked as finished with a failed status, and their tokens are
        removed.
        """
        expired_at = now()
        current = int(expired_at.timestamp())
        # Expired task claims are terminal failures and lose their token.
        self.query(
            """
            UPDATE task
            SET token = NULL,
                active_until = NULL,
                finished_at = :finished_at,
                sub_status = :sub_status,
                details = :details
            WHERE token IS NOT NULL AND active_until < :current
            """,
            {
                "current": current,
                "finished_at": expired_at.isoformat(),
                "sub_status": SubStatus.FAILED,
                "details": "No heartbeat received from the task",
            },
        )

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        current = now().timestamp()
        active_until = current + HEARTBEAT_DEFAULT_INTERVAL
        query = """
            INSERT INTO token_store (run_id, token, active_until)
            VALUES (:run_id, :token, :active_until)
            RETURNING token;
        """
        data = {
            "run_id": uint64_to_int64(run_id),
            "token": token,
            "active_until": active_until,
        }
        try:
            rows = self.query(query, data)
            return cast(str, rows[0]["token"])
        except IntegrityError:
            return None  # Token already created for this run ID

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        self._cleanup_expired_tokens()
        query = "SELECT token FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        rows = self.query(query, data)
        if not rows:
            return False
        return cast(str, rows[0]["token"]) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        query = "DELETE FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        self.query(query, data)

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        self._cleanup_expired_tokens()
        query = "SELECT run_id FROM token_store WHERE token = :token;"
        data = {"token": token}
        rows = self.query(query, data)
        if not rows:
            return None
        return int64_to_uint64(rows[0]["run_id"])

    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token."""
        # Clean up expired tokens
        self._cleanup_expired_tokens()

        # Update the active_until field
        current = now().timestamp()
        active_until = current + HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
        query = """
            UPDATE token_store
            SET active_until = :active_until
            WHERE token = :token
            RETURNING run_id;
        """
        data = {"active_until": active_until, "token": token}
        rows = self.query(query, data)
        return len(rows) > 0

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens and perform additional cleanup.

        This method is called before token operations to ensure integrity.
        Subclasses can override `_on_tokens_expired` to add custom cleanup logic.
        """
        current = now().timestamp()

        with self.session() as session:
            # Delete expired tokens and get their run_ids and active_until timestamps
            query = """
                DELETE FROM token_store
                WHERE active_until < :current
                RETURNING run_id, active_until;
            """
            rows = session.execute(text(query), {"current": current}).mappings().all()
            expired_records = [
                (int64_to_uint64(row["run_id"]), row["active_until"]) for row in rows
            ]

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
