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

# pylint: disable=too-many-lines
import hashlib
import json
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from logging import ERROR
from typing import Any, Literal, cast
from uuid import uuid4

from sqlalchemy import MetaData
from sqlalchemy.exc import IntegrityError

from flwr.app import Context, Message
from flwr.app.message import make_message
from flwr.app.metadata import Metadata
from flwr.app.user_config import UserConfig
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
from flwr.proto.control_pb2 import Automation  # pylint: disable=E0611
from flwr.proto.error_pb2 import Error as ProtoError  # pylint: disable=E0611
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611

# pylint: disable-next=E0611
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict
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
from flwr.supercore.sql_mixin import SqlMixin
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.typing import ConnectorOAuthSessionRecord, ConnectorRecord
from flwr.supercore.utils import (
    build_sql_in_params,
    int64_to_uint64,
    simulation_config_to_json,
    uint64_to_int64,
)

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

    def start_session(self, run_id: int) -> str:
        """Start a run-scoped object push session."""
        session_id = str(uuid4())
        self.query(
            """
            INSERT INTO object_push_sessions (
                session_id, run_id, expires_at, pending_count
            )
            VALUES (:session_id, :run_id, :expires_at, 0)
            """,
            {
                "session_id": session_id,
                "run_id": uint64_to_int64(run_id),
                "expires_at": now()
                + timedelta(seconds=OBJECT_PUSH_SESSION_TTL_SECONDS),
            },
        )
        return session_id

    def delete_sessions_in_run(self, run_id: int) -> None:
        """Delete all object push session bookkeeping for a run."""
        self.query(
            """
            DELETE FROM object_push_sessions
            WHERE run_id = :run_id
            """,
            {"run_id": uint64_to_int64(run_id)},
        )

    def preregister_object_tree(
        self, object_tree: ObjectTree, session_id: str
    ) -> list[str]:
        """Preregister an object tree and record its missing objects."""
        with self.session():
            # Load the run associated with the session
            rows = self.query(
                """
                SELECT run_id
                FROM object_push_sessions
                WHERE session_id = :session_id
                """,
                {"session_id": session_id},
            )
            if not rows:
                raise ValueError(f"Unknown object push session: {session_id}")
            run_id = int64_to_uint64(rows[0]["run_id"])

            # Preregister the tree and collect its currently missing objects
            missing_objects = self.object_store.preregister(run_id, object_tree)

            # Remove bookkeeping for an older session owning the same root
            rows = self.query(
                """
                SELECT session_id
                FROM object_push_session_roots
                WHERE root_object_id = :root_object_id AND session_id != :session_id
                """,
                {
                    "root_object_id": object_tree.object_id,
                    "session_id": session_id,
                },
            )
            if rows:
                self._cleanup_push_session(
                    rows[0]["session_id"], cleanup_messages=False
                )

            # Record ownership of the root
            self.query(
                """
                INSERT INTO object_push_session_roots (session_id, root_object_id)
                VALUES (:session_id, :root_object_id)
                """,
                {
                    "session_id": session_id,
                    "root_object_id": object_tree.object_id,
                },
            )

            # Record the objects that still need to be pushed
            if missing_objects:
                self.query(
                    """
                    INSERT INTO object_push_session_pending (session_id, object_id)
                    VALUES (:session_id, :object_id)
                    ON CONFLICT(session_id, object_id) DO NOTHING
                    """,
                    [
                        {"session_id": session_id, "object_id": object_id}
                        for object_id in missing_objects
                    ],
                )

            # Synchronize the materialized pending count.
            self.query(
                """
                UPDATE object_push_sessions
                SET pending_count = (
                    SELECT COUNT(*)
                    FROM object_push_session_pending
                    WHERE session_id = :session_id
                )
                WHERE session_id = :session_id
                """,
                {"session_id": session_id},
            )
            return missing_objects

    def _claim_pending_object(
        self,
        run_id: int,
        session_id: str,
        object_id: str,
    ) -> datetime | None:
        """Claim a pending object and return the push session expiry."""
        rows = self.query(
            """
            DELETE FROM object_push_session_pending AS pending
            WHERE pending.session_id = :session_id
              AND pending.object_id = :object_id
              AND EXISTS (
                  SELECT 1
                  FROM object_push_sessions AS session
                  WHERE session.session_id = :session_id
                    AND session.run_id = :run_id
              )
            RETURNING (
                SELECT expires_at
                FROM object_push_sessions
                WHERE session_id = :session_id
            ) AS expires_at
            """,
            {
                "session_id": session_id,
                "object_id": object_id,
                "run_id": uint64_to_int64(run_id),
            },
        )
        if not rows:
            return None

        expires_at = rows[0]["expires_at"]
        if isinstance(expires_at, str):  # SQLite returns string for TIMESTAMP column
            return datetime.fromisoformat(expires_at)
        return cast(datetime, expires_at)

    def store_object(
        self,
        run_id: int,
        session_id: str,
        object_id: str,
        object_content: bytes,
    ) -> bool:
        """Store an object if it is pending for an active push session."""
        try:
            with self.session():
                # Support legacy SuperNodes that do not send a session ID
                if not session_id:
                    rows = self.query(
                        """
                        SELECT session_id
                        FROM object_push_session_pending
                        WHERE object_id = :object_id
                        """,
                        {"object_id": object_id},
                    )
                    if not rows:
                        return False
                    session_id = rows[0]["session_id"]

                # Atomically validate the session and claim its pending object
                expires_at = self._claim_pending_object(run_id, session_id, object_id)
                if expires_at is None:
                    return False

                # Reject expired sessions and clean up their messages and objects
                if expires_at <= now():
                    self._cleanup_push_session(session_id, cleanup_messages=True)
                    return False

                # Store the object, decrement pending work, and refresh the session TTL
                self.object_store.put(object_id, object_content)
                rows = self.query(
                    """
                    UPDATE object_push_sessions
                    SET pending_count = pending_count - 1,
                        expires_at = :expires_at
                    WHERE session_id = :session_id
                    RETURNING pending_count
                    """,
                    {
                        "session_id": session_id,
                        "expires_at": now()
                        + timedelta(seconds=OBJECT_PUSH_SESSION_TTL_SECONDS),
                    },
                )
                pending_count = rows[0]["pending_count"]

                # Remove session bookkeeping once every pending object is stored
                if pending_count == 0:
                    self._cleanup_push_session(session_id, cleanup_messages=False)
                return True
        except Exception as err:  # pylint: disable=broad-exception-caught
            log(ERROR, "Failed to store object %s: %s", object_id, err)
            return False

    def get_object(self, run_id: int, object_id: str) -> bytes | None:
        """Get an object and clean up expired push sessions when needed."""
        with self.session():
            # Return immediately unless the object is known but unavailable
            content = self.object_store.get(object_id)
            if content != b"":
                return content

            # Find expired sessions in this run that are waiting for the object
            rows = self.query(
                """
                SELECT session.session_id
                FROM object_push_session_pending AS pending
                INNER JOIN object_push_sessions AS session
                    ON pending.session_id = session.session_id
                WHERE pending.object_id = :object_id
                  AND session.run_id = :run_id
                  AND session.expires_at <= :current
                """,
                {
                    "object_id": object_id,
                    "run_id": uint64_to_int64(run_id),
                    "current": now(),
                },
            )
            if not rows:
                return content

            # Clean up every expired session, then return the resulting object state
            for row in rows:
                self._cleanup_push_session(row["session_id"], cleanup_messages=True)
            return self.object_store.get(object_id)

    def _cleanup_push_session(self, session_id: str, *, cleanup_messages: bool) -> None:
        """Remove an object push session and optionally its messages."""
        with self.session():
            # Load message roots only when their data must also be cleaned up
            message_object_ids: set[str] = set()
            if cleanup_messages:
                rows = self.query(
                    """
                    SELECT root_object_id
                    FROM object_push_session_roots
                    WHERE session_id = :session_id
                    """,
                    {"session_id": session_id},
                )
                message_object_ids = {row["root_object_id"] for row in rows}

            # Delete the session and its cascaded root and pending rows
            self.query(
                """
                DELETE FROM object_push_sessions
                WHERE session_id = :session_id
                """,
                {"session_id": session_id},
            )

            # Delete expired object trees and their message metadata.
            if message_object_ids:
                for object_id in message_object_ids:
                    self.object_store.delete(object_id)
                self._on_push_session_expired(message_object_ids)

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
        params = {
            "flwr_aid": flwr_aid,
            "connector_ref": connector_ref,
            "credentials_json": credentials_json,
            "config_json": config_json,
        }
        self.query(
            """
            INSERT INTO connector (
                flwr_aid, connector_ref, credentials_json, config_json
            )
            VALUES (
                :flwr_aid, :connector_ref, :credentials_json, :config_json
            )
            ON CONFLICT(flwr_aid, connector_ref) DO UPDATE SET
                credentials_json = excluded.credentials_json,
                config_json = excluded.config_json
            """,
            params,
        )
        return True

    def get_connector(
        self, flwr_aid: str, connector_ref: str
    ) -> ConnectorRecord | None:
        """Return an account's connector, if present."""
        if not flwr_aid or not connector_ref:
            return None
        rows = self.query(
            """
            SELECT flwr_aid, connector_ref, credentials_json, config_json
            FROM connector
            WHERE flwr_aid = :flwr_aid
              AND connector_ref = :connector_ref
            """,
            {"flwr_aid": flwr_aid, "connector_ref": connector_ref},
        )
        if not rows:
            return None
        row = rows[0]
        return ConnectorRecord(
            flwr_aid=row["flwr_aid"],
            connector_ref=row["connector_ref"],
            credentials_json=row["credentials_json"],
            config_json=row["config_json"],
        )

    def delete_connector(self, flwr_aid: str, connector_ref: str) -> bool:
        """Delete an account's connector if it exists."""
        if not flwr_aid or not connector_ref:
            return False
        params = {"flwr_aid": flwr_aid, "connector_ref": connector_ref}
        with self.session():
            rows = self.query(
                """
                SELECT connector_ref
                FROM connector
                WHERE flwr_aid = :flwr_aid
                  AND connector_ref = :connector_ref
                """,
                params,
            )
            if not rows:
                return False
            self.query(
                """
                DELETE FROM connector
                WHERE flwr_aid = :flwr_aid
                  AND connector_ref = :connector_ref
                """,
                params,
            )
        return True

    def bind_connectors_to_run(
        self, run_id: int, connector_refs: Sequence[str]
    ) -> bool:
        """Associate connector references with a run."""
        if isinstance(connector_refs, str):
            return False
        stored_run_id = uint64_to_int64(run_id)
        bound_refs = set(self.get_run_connector_refs(run_id))
        data = [
            {
                "run_id": stored_run_id,
                "connector_ref": connector_ref,
            }
            for connector_ref in dict.fromkeys(connector_refs)
            if connector_ref not in bound_refs
        ]
        if data:
            self.query(
                """
                INSERT INTO run_connector (run_id, connector_ref)
                VALUES (:run_id, :connector_ref)
                """,
                data,
            )
        return True

    def get_run_connector_refs(self, run_id: int) -> Sequence[str]:
        """Return connector references associated with a run."""
        rows = self.query(
            """
            SELECT connector_ref
            FROM run_connector
            WHERE run_id = :run_id
            ORDER BY connector_ref
            """,
            {"run_id": uint64_to_int64(run_id)},
        )
        return [row["connector_ref"] for row in rows]

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
        created_at = now()
        session = ConnectorOAuthSessionRecord(
            oauth_session_id=oauth_session_id,
            flwr_aid=flwr_aid,
            connector_ref=connector_ref,
            state=state,
            redirect_uri=redirect_uri,
            pkce_verifier=pkce_verifier,
            created_at=created_at.isoformat(),
            expires_at=expires_at.isoformat(),
            completed_at=None,
        )
        try:
            self.query(
                """
                INSERT INTO connector_oauth_session (
                    oauth_session_id, flwr_aid, connector_ref, state,
                    redirect_uri, pkce_verifier, created_at, expires_at,
                    completed_at
                )
                VALUES (
                    :oauth_session_id, :flwr_aid, :connector_ref, :state,
                    :redirect_uri, :pkce_verifier, :created_at, :expires_at,
                    :completed_at
                )
                """,
                {
                    "oauth_session_id": session.oauth_session_id,
                    "flwr_aid": session.flwr_aid,
                    "connector_ref": session.connector_ref,
                    "state": session.state,
                    "redirect_uri": session.redirect_uri,
                    "pkce_verifier": session.pkce_verifier,
                    "created_at": created_at,
                    "expires_at": expires_at,
                    "completed_at": session.completed_at,
                },
            )
        except IntegrityError:
            return None
        return session

    def get_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> ConnectorOAuthSessionRecord | None:
        """Return an account's connector OAuth session, if present."""
        if not oauth_session_id or not flwr_aid:
            return None
        rows = self.query(
            """
            SELECT oauth_session_id, flwr_aid, connector_ref, state,
                   redirect_uri, pkce_verifier, created_at, expires_at,
                   completed_at
            FROM connector_oauth_session
            WHERE oauth_session_id = :oauth_session_id
              AND flwr_aid = :flwr_aid
            """,
            {"oauth_session_id": oauth_session_id, "flwr_aid": flwr_aid},
        )
        if not rows:
            return None
        return _connector_oauth_session_from_row(rows[0])

    def complete_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> bool:
        """Mark a pending connector OAuth session as completed."""
        if not oauth_session_id or not flwr_aid:
            return False
        completed_at = now()
        updated = self.query(
            """
            UPDATE connector_oauth_session
            SET completed_at = :completed_at
            WHERE oauth_session_id = :oauth_session_id
              AND flwr_aid = :flwr_aid
              AND completed_at IS NULL
              AND expires_at > :completed_at
            RETURNING oauth_session_id
            """,
            {
                "oauth_session_id": oauth_session_id,
                "flwr_aid": flwr_aid,
                "completed_at": completed_at,
            },
        )
        return bool(updated)

    def get_run_series(  # pylint: disable=R0914
        self,
        *,
        series_ids: Sequence[int] | None = None,
        federation_ids: Sequence[str] | None = None,
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
            or (federation_ids is not None and not federation_ids)
        ):
            return []

        # Build optional filters for the run-series page.
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if series_ids is not None:
            sint64_series_ids = [uint64_to_int64(series_id) for series_id in series_ids]
            placeholders, in_params = build_sql_in_params(sint64_series_ids, "sid")
            conditions.append(f"series_id IN ({placeholders})")
            params.update(in_params)
        if federation_ids is not None:
            placeholders, in_params = build_sql_in_params(federation_ids, "fed")
            conditions.append(f"federation_id IN ({placeholders})")
            params.update(in_params)
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
                SELECT series_id, federation_id, description, created_at, updated_at
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
        federation_id: str,
        series_id: int | None,
        description: str | None = None,
    ) -> int | None:
        """Store a run in a run series and return the series ID."""
        insert_query = """
            INSERT INTO run_series
            (series_id, federation_id, description, created_at, updated_at)
            VALUES
            (:series_id, :federation_id, :description, :created_at, :updated_at)
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
                            "federation_id": federation_id,
                            "description": description,
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
                        WHERE series_id = :series_id AND federation_id = :federation_id
                        RETURNING series_id
                        """,
                        {
                            "series_id": uint64_to_int64(series_id),
                            "federation_id": federation_id,
                            "updated_at": now(),
                        },
                    )
                    if not rows:
                        log(
                            ERROR,
                            "Run series %d not found in federation %r",
                            series_id,
                            federation_id,
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
        federation_config_json = None
        if federation_config is not None:
            federation_config_json = json.dumps(
                simulation_config_to_json(federation_config)
            )

        try:
            with self.session():
                current = now()
                rows = self.query(
                    """
                    INSERT INTO automation (
                        federation_id, status, series_id, flwr_aid,
                        fab_id, fab_version, fab_hash, override_config,
                        federation_config, primary_task_type,
                        created_at, updated_at, next_run_at, fixed_interval,
                        remaining_runs, stopped_at
                    )
                    VALUES (
                        :federation_id, :status, :series_id, :flwr_aid,
                        :fab_id, :fab_version, :fab_hash, :override_config,
                        :federation_config, :primary_task_type,
                        :created_at, :updated_at, :next_run_at, :fixed_interval,
                        :remaining_runs, :stopped_at
                    )
                    RETURNING *
                    """,
                    {
                        "federation_id": federation_id,
                        "status": AutomationStatus.ACTIVE,
                        "series_id": uint64_to_int64(series_id),
                        "flwr_aid": flwr_aid,
                        "fab_id": fab_id,
                        "fab_version": fab_version,
                        "fab_hash": fab_hash,
                        "override_config": json.dumps(override_config),
                        "federation_config": federation_config_json,
                        "primary_task_type": primary_task_type,
                        "created_at": current,
                        "updated_at": current,
                        "next_run_at": next_run_at,
                        "fixed_interval": fixed_interval,
                        "remaining_runs": max_runs,
                        "stopped_at": None,
                    },
                )
        except IntegrityError as exc:
            raise ValueError(f"Could not store automation: {exc}") from exc

        row = rows[0]
        return Automation(
            automation_id=row["automation_id"],
            status=row["status"],
            federation=row["federation_id"],
            series_id=int64_to_uint64(row["series_id"]),
            flwr_aid=row["flwr_aid"],
            created_at=timestamp_to_iso(row["created_at"]),
            updated_at=timestamp_to_iso(row["updated_at"]),
            next_run_at=timestamp_to_iso(row["next_run_at"]),
            fixed_interval=row["fixed_interval"],
            remaining_runs=row["remaining_runs"],
        )

    def list_automations(  # pylint: disable=too-many-arguments,too-many-locals
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

        conditions: list[str] = []
        params: dict[str, Any] = {}
        if federation is not None:
            conditions.append("federation_id = :federation_id")
            params["federation_id"] = federation
        if statuses is not None:
            placeholders = ",".join(f":status_{i}" for i in range(len(statuses)))
            conditions.append(f"status IN ({placeholders})")
            params.update({f"status_{i}": status for i, status in enumerate(statuses)})
        if due_before is not None:
            conditions.append("next_run_at <= :due_before")
            # Finite automations with no remaining runs are already claimed.
            conditions.append("(remaining_runs IS NULL OR remaining_runs > 0)")
            params["due_before"] = due_before.isoformat()

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_clause = "ORDER BY updated_at DESC, automation_id DESC"
        if order_by == "next_run_at":
            order_clause = "ORDER BY next_run_at ASC, automation_id ASC"

        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT :limit"
            params["limit"] = limit

        rows = self.query(
            f"""
            SELECT *
            FROM automation
            {where_clause}
            {order_clause}
            {limit_clause}
            """,
            params,
        )
        automations = []
        for row in rows:
            next_run_at = row["next_run_at"]
            stopped_at = row["stopped_at"]
            automations.append(
                Automation(
                    automation_id=row["automation_id"],
                    status=row["status"],
                    federation=row["federation_id"],
                    series_id=int64_to_uint64(row["series_id"]),
                    flwr_aid=row["flwr_aid"],
                    created_at=timestamp_to_iso(row["created_at"]),
                    updated_at=timestamp_to_iso(row["updated_at"]),
                    next_run_at=timestamp_to_iso(next_run_at),
                    fixed_interval=row["fixed_interval"],
                    remaining_runs=row["remaining_runs"],
                    stopped_at=timestamp_to_iso(stopped_at) if stopped_at else None,
                )
            )
        return automations

    def stop_automation(self, automation_id: int) -> bool:
        """Stop an active automation."""
        stopped_at = now()
        rows = self.query(
            """
            UPDATE automation
            SET status = :status,
                updated_at = :updated_at,
                stopped_at = :stopped_at
            WHERE automation_id = :automation_id
            AND status = :active_status
            RETURNING automation_id
            """,
            {
                "automation_id": automation_id,
                "status": AutomationStatus.STOPPED,
                "updated_at": stopped_at,
                "stopped_at": stopped_at,
                "active_status": AutomationStatus.ACTIVE,
            },
        )
        return bool(rows)

    def advance_automation(
        self,
        automation_id: int,
        *,
        previous_next_run_at: str,
        next_run_at: str | None,
    ) -> bool:
        """Advance an active automation occurrence."""
        timestamp = now()
        params: dict[str, Any] = {
            "automation_id": automation_id,
            "active_status": AutomationStatus.ACTIVE,
            "updated_at": timestamp,
            "previous_next_run_at": previous_next_run_at,
            "next_run_at": next_run_at,
        }

        rows = self.query(
            """
            UPDATE automation
            SET updated_at = :updated_at,
                next_run_at = CASE
                    WHEN remaining_runs IS NOT NULL AND remaining_runs <= 1
                        THEN next_run_at
                    ELSE :next_run_at
                END,
                remaining_runs = CASE
                    WHEN remaining_runs IS NULL
                        THEN NULL
                    WHEN remaining_runs > 0
                        THEN remaining_runs - 1
                    ELSE 0
                END
            WHERE automation_id = :automation_id
            AND status = :active_status
            AND next_run_at = :previous_next_run_at
            AND (remaining_runs IS NULL OR remaining_runs > 0)
            AND (:next_run_at IS NOT NULL OR remaining_runs <= 1)
            RETURNING automation_id
            """,
            params,
        )
        return bool(rows)

    def finish_automation(
        self,
        automation_id: int,
        *,
        status: Literal[AutomationStatus.COMPLETED, AutomationStatus.FAILED],
    ) -> bool:
        """Finish an active automation with a terminal status."""
        completed_condition = ""
        if status == AutomationStatus.COMPLETED:
            completed_condition = "AND remaining_runs = 0"

        rows = self.query(
            f"""
            UPDATE automation
            SET status = :status,
                updated_at = :updated_at
            WHERE automation_id = :automation_id
            AND status = :active_status
            {completed_condition}
            RETURNING automation_id
            """,
            {
                "automation_id": automation_id,
                "status": status,
                "updated_at": now(),
                "active_status": AutomationStatus.ACTIVE,
            },
        )
        return bool(rows)

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
            WHERE CAST(:requesting_task_id AS BIGINT) IS NULL
            OR EXISTS (
                SELECT 1
                FROM task
                WHERE task_id = CAST(:requesting_task_id AS BIGINT)
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
            placeholders, in_params = build_sql_in_params(sint64_task_ids, "tid")
            conditions.append(f"task_id IN ({placeholders})")
            params.update(in_params)

        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            placeholders, in_params = build_sql_in_params(sint64_run_ids, "rid")
            conditions.append(f"run_id IN ({placeholders})")
            params.update(in_params)

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

    def add_task_usage(self, task_id: int, usage: TaskUsage) -> None:
        """Record usage for the specified task."""
        with self.session():
            self.query(
                """
                INSERT INTO task_usage (
                    run_id, task_id, input_tokens, output_tokens, total_tokens,
                    usage_type, provider, created_at, reported_at
                )
                SELECT
                    run_id, task_id, :input_tokens, :output_tokens,
                    :total_tokens, :usage_type, :provider, :created_at, :reported_at
                FROM task
                WHERE task_id = :task_id
                """,
                _task_usage_to_row(task_id, usage),
            )

    def get_task_usage(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        task_ids: Sequence[int] | None = None,
    ) -> Sequence[TaskUsage]:
        """Retrieve task usage records based on the specified filters."""
        conditions = []
        params: dict[str, Any] = {}

        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            placeholders, in_params = build_sql_in_params(sint64_run_ids, "rid")
            conditions.append(f"run_id IN ({placeholders})")
            params.update(in_params)

        if task_ids is not None:
            if not task_ids:
                return []
            sint64_task_ids = [uint64_to_int64(task_id) for task_id in task_ids]
            placeholders, in_params = build_sql_in_params(sint64_task_ids, "tid")
            conditions.append(f"task_id IN ({placeholders})")
            params.update(in_params)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
            SELECT input_tokens, output_tokens, total_tokens, usage_type, provider
            FROM task_usage
            {where_clause}
            ORDER BY id ASC
        """

        rows = self.query(query, params)
        return [_task_usage_from_row(row) for row in rows]

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
            activated_at = now()
            active_until = activated_at + timedelta(
                seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
            )

            # Activation is a strict STARTING -> RUNNING transition.
            rows = self.query(
                f"""
                UPDATE task
                SET running_at = :running_at, active_until = :active_until
                WHERE task_id = :task_id AND {STATUS_CONDITIONS[Status.STARTING]}
                RETURNING task_id
                """,
                {
                    "task_id": uint64_to_int64(task_id),
                    "running_at": activated_at,
                    "active_until": active_until,
                },
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
            placeholders, in_params = build_sql_in_params(sint64_dst_task_ids, "dtid")
            conditions.append(f"dst_task_id IN ({placeholders})")
            params.update(in_params)

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

        Expired starting tasks are moved back to pending. Expired running tasks
        are marked as finished with a failed status. Tokens are removed in both
        cases.
        """
        expired_at = now()
        # Claims that never reached RUNNING are retryable launch failures.
        self.query(
            f"""
            UPDATE task
            SET token = NULL, active_until = NULL, starting_at = NULL,
                sub_status = '', details = ''
            WHERE token IS NOT NULL AND active_until < :current
            AND {STATUS_CONDITIONS[Status.STARTING]}
            """,
            {"current": expired_at},
        )

        # Expired running task claims are terminal failures and lose their token.
        rows = self.query(
            f"""
            UPDATE task
            SET token = NULL, finished_at = active_until, active_until = NULL,
                sub_status = :sub_status, details = :details
            WHERE token IS NOT NULL AND active_until < :current
            AND {STATUS_CONDITIONS[Status.RUNNING]}
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


def _connector_oauth_session_from_row(
    row: dict[str, Any],
) -> ConnectorOAuthSessionRecord:
    """Convert a connector OAuth session row to its persistence record."""
    return ConnectorOAuthSessionRecord(
        oauth_session_id=row["oauth_session_id"],
        flwr_aid=row["flwr_aid"],
        connector_ref=row["connector_ref"],
        state=row["state"],
        redirect_uri=row["redirect_uri"],
        pkce_verifier=row["pkce_verifier"],
        created_at=timestamp_to_iso(row["created_at"]),
        expires_at=timestamp_to_iso(row["expires_at"]),
        completed_at=timestamp_to_iso(row["completed_at"]) or None,
    )


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
        federation=row["federation_id"],
        description=row["description"] or "",
        created_at=timestamp_to_iso(row["created_at"]),
        updated_at=timestamp_to_iso(row["updated_at"]),
    )


def _task_usage_to_row(task_id: int, usage: TaskUsage) -> dict[str, Any]:
    """Convert a TaskUsage proto to database row values."""
    return {
        "task_id": uint64_to_int64(task_id),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "usage_type": usage.usage_type,
        "provider": usage.provider,
        "created_at": now(),
        "reported_at": None,
    }


def _task_usage_from_row(row: dict[str, Any]) -> TaskUsage:
    """Convert a task_usage row to a TaskUsage proto."""
    return TaskUsage(
        usage_type=row["usage_type"],
        provider=row["provider"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        total_tokens=row["total_tokens"],
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
