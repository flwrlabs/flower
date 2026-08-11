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

from sqlalchemy import (
    MetaData,
    case,
    delete,
    exists,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import Insert as PostgresInsert
from sqlalchemy.dialects.sqlite import Insert as SQLiteInsert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError

from flwr.app import Context, Message
from flwr.app.message import make_message
from flwr.app.metadata import Metadata
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
from flwr.proto.control_pb2 import Automation, StartRunRequest  # pylint: disable=E0611
from flwr.proto.error_pb2 import Error as ProtoError  # pylint: disable=E0611
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
from flwr.supercore.state.schema.corestate_models import Automation as AutomationModel
from flwr.supercore.state.schema.corestate_models import Connector as ConnectorModel
from flwr.supercore.state.schema.corestate_models import (
    ConnectorOAuthSession as ConnectorOAuthSessionModel,
)
from flwr.supercore.state.schema.corestate_models import Fab as FabModel
from flwr.supercore.state.schema.corestate_models import NonceStore as NonceStoreModel
from flwr.supercore.state.schema.corestate_models import (
    ObjectPushSession as ObjectPushSessionModel,
)
from flwr.supercore.state.schema.corestate_models import (
    ObjectPushSessionPending as ObjectPushSessionPendingModel,
)
from flwr.supercore.state.schema.corestate_models import (
    ObjectPushSessionRoot as ObjectPushSessionRootModel,
)
from flwr.supercore.state.schema.corestate_models import (
    RunConnector as RunConnectorModel,
)
from flwr.supercore.state.schema.corestate_models import RunSeries as RunSeriesModel
from flwr.supercore.state.schema.corestate_models import (
    SeriesContext as SeriesContextModel,
)
from flwr.supercore.state.schema.corestate_models import SeriesRuns as SeriesRunsModel
from flwr.supercore.state.schema.corestate_models import Task as TaskModel
from flwr.supercore.state.schema.corestate_models import TaskEvent as TaskEventModel
from flwr.supercore.state.schema.corestate_models import TaskLogsTable
from flwr.supercore.state.schema.corestate_models import TaskMessage as TaskMessageModel
from flwr.supercore.state.schema.corestate_models import TaskUsage as TaskUsageModel
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.typing import ConnectorOAuthSessionRecord, ConnectorRecord
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

    def dialect_insert(self, table: Any) -> SQLiteInsert | PostgresInsert:
        """Return a dialect-specific insert statement for CoreState upserts."""
        if self.database_backend == "sqlite":
            return sqlite_insert(table)

        raise NotImplementedError(
            f"No dialect-specific insert configured for {self.database_backend!r}."
        )

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
        expires_at = now() + timedelta(seconds=OBJECT_PUSH_SESSION_TTL_SECONDS)
        stmt = self.dialect_insert(ObjectPushSessionModel).values(
            session_id=session_id,
            run_id=uint64_to_int64(run_id),
            expires_at=expires_at,
            pending_count=0,
        )
        with self.session() as session:
            session.execute(stmt)
        return session_id

    def delete_sessions_in_run(self, run_id: int) -> None:
        """Delete all object push session bookkeeping for a run."""
        with self.session() as session:
            session.execute(
                delete(ObjectPushSessionModel).where(
                    ObjectPushSessionModel.run_id == uint64_to_int64(run_id)
                )
            )

    def preregister_object_tree(
        self, object_tree: ObjectTree, session_id: str
    ) -> list[str]:
        """Preregister an object tree and record its missing objects."""
        with self.session() as session:
            # Load the run associated with the session.
            push_session = session.get(
                ObjectPushSessionModel, session_id, populate_existing=True
            )
            if push_session is None:
                raise ValueError(f"Unknown object push session: {session_id}")
            run_id = int64_to_uint64(push_session.run_id)

            # Preregister the tree and collect its currently missing objects.
            missing_objects = self.object_store.preregister(run_id, object_tree)

            # Remove bookkeeping for an older session owning the same root.
            old_session_id = session.scalar(
                select(ObjectPushSessionRootModel.session_id).where(
                    ObjectPushSessionRootModel.root_object_id == object_tree.object_id,
                    ObjectPushSessionRootModel.session_id != session_id,
                )
            )
            if old_session_id is not None:
                self._cleanup_push_session(old_session_id, cleanup_messages=False)

            # Record ownership of the root.
            session.add(
                ObjectPushSessionRootModel(
                    session_id=session_id, root_object_id=object_tree.object_id
                )
            )

            # Record the objects that still need to be pushed.
            if missing_objects:
                stmt = self.dialect_insert(ObjectPushSessionPendingModel).values(
                    [
                        {"session_id": session_id, "object_id": object_id}
                        for object_id in missing_objects
                    ]
                )
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[
                        ObjectPushSessionPendingModel.session_id,
                        ObjectPushSessionPendingModel.object_id,
                    ]
                )
                session.execute(stmt)

            # Synchronize the materialized pending count.
            pending_count = session.scalar(
                select(func.count()).where(  # pylint: disable=not-callable
                    ObjectPushSessionPendingModel.session_id == session_id
                )
            )
            push_session.pending_count = int(pending_count or 0)
            return missing_objects

    def _claim_pending_object(
        self,
        run_id: int,
        session_id: str,
        object_id: str,
    ) -> datetime | None:
        """Claim a pending object and return the refreshed push session expiry."""
        with self.session() as session:
            claimed_session_id = session.scalar(
                delete(ObjectPushSessionPendingModel)
                .where(
                    ObjectPushSessionPendingModel.session_id == session_id,
                    ObjectPushSessionPendingModel.object_id == object_id,
                    select(ObjectPushSessionModel.session_id)
                    .where(
                        ObjectPushSessionModel.session_id == session_id,
                        ObjectPushSessionModel.run_id == uint64_to_int64(run_id),
                    )
                    .exists(),
                )
                .returning(ObjectPushSessionPendingModel.session_id)
            )
            if claimed_session_id is None:
                return None

            # Re-read after the successful claim. Another push can refresh the
            # session while this request waits to delete the pending row.
            expires_at = session.scalar(
                select(ObjectPushSessionModel.expires_at)
                .where(ObjectPushSessionModel.session_id == claimed_session_id)
                .execution_options(populate_existing=True)
            )
            return expires_at

    def store_object(
        self,
        run_id: int,
        session_id: str,
        object_id: str,
        object_content: bytes,
    ) -> bool:
        """Store an object if it is pending for an active push session."""
        try:
            with self.session() as session:
                # Support legacy SuperNodes that do not send a session ID.
                if not session_id:
                    resolved_session_id = session.scalar(
                        select(ObjectPushSessionPendingModel.session_id).where(
                            ObjectPushSessionPendingModel.object_id == object_id
                        )
                    )
                    if resolved_session_id is None:
                        return False
                    session_id = resolved_session_id

                # Atomically validate the session and claim its pending object.
                expires_at = self._claim_pending_object(run_id, session_id, object_id)
                if expires_at is None:
                    return False

                # Reject expired sessions and clean up their messages and objects.
                if expires_at <= now():
                    self._cleanup_push_session(session_id, cleanup_messages=True)
                    return False

                # Store the object, decrement pending work, and refresh the session TTL.
                self.object_store.put(object_id, object_content)
                refreshed_expires_at = now() + timedelta(
                    seconds=OBJECT_PUSH_SESSION_TTL_SECONDS
                )
                pending_count = session.scalar(
                    update(ObjectPushSessionModel)
                    .where(ObjectPushSessionModel.session_id == session_id)
                    .values(
                        pending_count=ObjectPushSessionModel.pending_count - 1,
                        expires_at=refreshed_expires_at,
                    )
                    .returning(ObjectPushSessionModel.pending_count)
                )
                if pending_count is None:
                    raise RuntimeError("Object push session disappeared after claim")

                # Remove session bookkeeping once every pending object is stored.
                if pending_count == 0:
                    self._cleanup_push_session(session_id, cleanup_messages=False)
                return True
        except Exception as err:  # pylint: disable=broad-exception-caught
            log(ERROR, "Failed to store object %s: %s", object_id, err)
            return False

    def get_object(self, run_id: int, object_id: str) -> bytes | None:
        """Get an object and clean up expired push sessions when needed."""
        with self.session() as session:
            # Return immediately unless the object is known but unavailable.
            content = self.object_store.get(object_id)
            if content != b"":
                return content

            # Find expired sessions in this run that are waiting for the object.
            expired_session_ids = list(
                session.scalars(
                    select(ObjectPushSessionModel.session_id)
                    .join(
                        ObjectPushSessionPendingModel,
                        ObjectPushSessionPendingModel.session_id
                        == ObjectPushSessionModel.session_id,
                    )
                    .where(
                        ObjectPushSessionPendingModel.object_id == object_id,
                        ObjectPushSessionModel.run_id == uint64_to_int64(run_id),
                        ObjectPushSessionModel.expires_at <= now(),
                    )
                )
            )
            if not expired_session_ids:
                return content

            # Clean up every expired session, then return the resulting object state.
            for expired_session_id in expired_session_ids:
                self._cleanup_push_session(expired_session_id, cleanup_messages=True)
            return self.object_store.get(object_id)

    def _cleanup_push_session(self, session_id: str, *, cleanup_messages: bool) -> None:
        """Remove an object push session and optionally its messages."""
        with self.session() as session:
            # Load message roots only when their data must also be cleaned up.
            message_object_ids: set[str] = set()
            if cleanup_messages:
                message_object_ids = set(
                    session.scalars(
                        select(ObjectPushSessionRootModel.root_object_id).where(
                            ObjectPushSessionRootModel.session_id == session_id
                        )
                    )
                )

            # Delete the session and its cascaded root and pending rows.
            session.execute(
                delete(ObjectPushSessionModel).where(
                    ObjectPushSessionModel.session_id == session_id
                )
            )

            # Delete expired object trees and their message metadata.
            if message_object_ids:
                for message_object_id in message_object_ids:
                    self.object_store.delete(message_object_id)
                self._on_push_session_expired(message_object_ids)

    def store_fab(self, fab: Fab) -> str:
        """Store a FAB."""
        fab_hash = hashlib.sha256(fab.content).hexdigest()
        if fab.hash_str and fab.hash_str != fab_hash:
            raise ValueError(
                f"FAB hash mismatch: provided {fab.hash_str}, computed {fab_hash}"
            )
        # Keep launch behavior: last write wins for metadata under the same
        # content hash.
        stmt = self.dialect_insert(FabModel).values(
            fab_hash=fab_hash,
            content=fab.content,
            verifications=json.dumps(fab.verifications),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[FabModel.fab_hash],
            set_={
                "content": stmt.excluded.content,
                "verifications": stmt.excluded.verifications,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return fab_hash

    def get_fab(self, fab_hash: str) -> Fab | None:
        """Return a FAB by hash."""
        with self.session() as session:
            row = session.get(FabModel, fab_hash, populate_existing=True)
            if row is None:
                return None
            # Launch tradeoff: do not recompute content hash on reads; rely on
            # write-time validation and hash-addressed lookup.
            return Fab(
                hash_str=row.fab_hash,
                content=row.content,
                verifications=json.loads(row.verifications),
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
        stmt = self.dialect_insert(ConnectorModel).values(
            flwr_aid=flwr_aid,
            connector_ref=connector_ref,
            credentials_json=credentials_json,
            config_json=config_json,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ConnectorModel.flwr_aid, ConnectorModel.connector_ref],
            set_={
                "credentials_json": stmt.excluded.credentials_json,
                "config_json": stmt.excluded.config_json,
            },
        )
        with self.session() as session:
            session.execute(stmt)
        return True

    def get_connector(
        self, flwr_aid: str, connector_ref: str
    ) -> ConnectorRecord | None:
        """Return an account's connector, if present."""
        if not flwr_aid or not connector_ref:
            return None
        with self.session() as session:
            row = session.get(
                ConnectorModel,
                (flwr_aid, connector_ref),
                populate_existing=True,
            )
            if row is None:
                return None
            return ConnectorRecord(
                flwr_aid=row.flwr_aid,
                connector_ref=row.connector_ref,
                credentials_json=row.credentials_json,
                config_json=row.config_json,
            )

    def delete_connector(self, flwr_aid: str, connector_ref: str) -> bool:
        """Delete an account's connector if it exists."""
        if not flwr_aid or not connector_ref:
            return False
        with self.session() as session:
            deleted_connector_ref = session.scalar(
                delete(ConnectorModel)
                .where(
                    ConnectorModel.flwr_aid == flwr_aid,
                    ConnectorModel.connector_ref == connector_ref,
                )
                .returning(ConnectorModel.connector_ref)
            )
            return deleted_connector_ref is not None

    def bind_connectors_to_run(
        self, run_id: int, connector_refs: Sequence[str]
    ) -> bool:
        """Associate connector references with a run."""
        if isinstance(connector_refs, str):
            return False
        stored_run_id = uint64_to_int64(run_id)
        bound_refs = set(self.get_run_connector_refs(run_id))
        run_connectors = [
            RunConnectorModel(run_id=stored_run_id, connector_ref=connector_ref)
            for connector_ref in dict.fromkeys(connector_refs)
            if connector_ref not in bound_refs
        ]
        if run_connectors:
            with self.session() as session:
                session.add_all(run_connectors)
        return True

    def get_run_connector_refs(self, run_id: int) -> Sequence[str]:
        """Return connector references associated with a run."""
        with self.session() as session:
            return list(
                session.scalars(
                    select(RunConnectorModel.connector_ref)
                    .where(RunConnectorModel.run_id == uint64_to_int64(run_id))
                    .order_by(RunConnectorModel.connector_ref)
                )
            )

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
        model = ConnectorOAuthSessionModel(
            oauth_session_id=oauth_session_id,
            flwr_aid=flwr_aid,
            connector_ref=connector_ref,
            state=state,
            redirect_uri=redirect_uri,
            pkce_verifier=pkce_verifier,
            created_at=created_at,
            expires_at=expires_at,
            completed_at=None,
        )
        try:
            with self.session() as db_session:
                db_session.add(model)
                db_session.flush()
                return _connector_oauth_session_from_model(model)
        except IntegrityError:
            return None

    def get_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> ConnectorOAuthSessionRecord | None:
        """Return an account's connector OAuth session, if present."""
        if not oauth_session_id or not flwr_aid:
            return None
        with self.session() as session:
            row = session.scalars(
                select(ConnectorOAuthSessionModel)
                .where(ConnectorOAuthSessionModel.oauth_session_id == oauth_session_id)
                .where(ConnectorOAuthSessionModel.flwr_aid == flwr_aid)
                .execution_options(populate_existing=True)
            ).one_or_none()
            if row is None:
                return None
            return _connector_oauth_session_from_model(row)

    def complete_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> bool:
        """Mark a pending connector OAuth session as completed."""
        if not oauth_session_id or not flwr_aid:
            return False
        completed_at = now()
        with self.session() as session:
            updated_oauth_session_id = session.scalar(
                update(ConnectorOAuthSessionModel)
                .where(
                    ConnectorOAuthSessionModel.oauth_session_id == oauth_session_id,
                    ConnectorOAuthSessionModel.flwr_aid == flwr_aid,
                    ConnectorOAuthSessionModel.completed_at.is_(None),
                    ConnectorOAuthSessionModel.expires_at > completed_at,
                )
                .values(completed_at=completed_at)
                .returning(ConnectorOAuthSessionModel.oauth_session_id)
            )
            return updated_oauth_session_id is not None

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

        page_query = select(RunSeriesModel.series_id)
        if series_ids is not None:
            sint64_series_ids = [uint64_to_int64(series_id) for series_id in series_ids]
            page_query = page_query.where(
                RunSeriesModel.series_id.in_(sint64_series_ids)
            )
        if federation_ids is not None:
            page_query = page_query.where(
                RunSeriesModel.federation_id.in_(federation_ids)
            )
        if updated_before is not None:
            page_query = page_query.where(
                RunSeriesModel.updated_at < datetime.fromisoformat(updated_before)
            )
        page_query = page_query.order_by(RunSeriesModel.updated_at.desc())
        if limit is not None:
            page_query = page_query.limit(limit)

        # Select the requested page before joining run IDs so limit applies to series.
        selected_series = page_query.subquery()
        query = (
            select(RunSeriesModel, SeriesRunsModel.run_id)
            .join(
                selected_series,
                RunSeriesModel.series_id == selected_series.c.series_id,
            )
            .outerjoin(
                SeriesRunsModel,
                SeriesRunsModel.series_id == RunSeriesModel.series_id,
            )
            .order_by(RunSeriesModel.updated_at.desc())
            .execution_options(populate_existing=True)
        )

        # Fold the joined rows back into one RunSeries per series.
        series_by_id: dict[int, RunSeries] = {}
        with self.session() as session:
            for series_model, stored_run_id in session.execute(query):
                series_id = series_model.series_id
                if series_id not in series_by_id:
                    series_by_id[series_id] = _run_series_from_model(series_model)
                if stored_run_id is not None:
                    series_by_id[series_id].run_ids.append(
                        int64_to_uint64(stored_run_id)
                    )
        return list(series_by_id.values())

    def get_run_series_context(self, series_id: int) -> Context | None:
        """Return the shared Context for the specified RunSeries, if present."""
        with self.session() as session:
            row = session.get(
                SeriesContextModel,
                uint64_to_int64(series_id),
                populate_existing=True,
            )
            if row is None or row.context is None:
                return None
            return context_from_bytes(row.context)

    def set_run_series_context(self, series_id: int, context: Context) -> None:
        """Set the shared Context for the specified RunSeries."""
        sint_series_id = uint64_to_int64(series_id)
        context_bytes = context_to_bytes(context)
        stmt = self.dialect_insert(SeriesContextModel).values(
            series_id=sint_series_id,
            context=context_bytes,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[SeriesContextModel.series_id],
            set_={"context": stmt.excluded.context},
        )
        with self.session() as session:
            session.execute(stmt)

    def store_run_in_series(
        self,
        run_id: int,
        federation_id: str,
        series_id: int | None,
        description: str | None = None,
    ) -> int | None:
        """Store a run in a run series and return the series ID."""
        try:
            with self.session() as session:
                if series_id is None:
                    # No series was provided, so create one before linking the run.
                    candidate = generate_rand_int_from_bytes(SERIES_ID_NUM_BYTES)
                    timestamp = now()
                    stmt = (
                        self.dialect_insert(RunSeriesModel)
                        .values(
                            series_id=uint64_to_int64(candidate),
                            federation_id=federation_id,
                            description=description,
                            created_at=timestamp,
                            updated_at=timestamp,
                        )
                        .on_conflict_do_nothing(
                            index_elements=[RunSeriesModel.series_id]
                        )
                        .returning(RunSeriesModel.series_id)
                    )
                    resolved_series_id = (
                        candidate if session.scalar(stmt) is not None else None
                    )
                    if resolved_series_id is None:
                        return None

                else:
                    updated_series_id = session.scalar(
                        update(RunSeriesModel)
                        .where(
                            RunSeriesModel.series_id == uint64_to_int64(series_id),
                            RunSeriesModel.federation_id == federation_id,
                        )
                        .values(updated_at=now())
                        .returning(RunSeriesModel.series_id)
                    )
                    if updated_series_id is None:
                        log(
                            ERROR,
                            "Run series %d not found in federation %r",
                            series_id,
                            federation_id,
                        )
                        return None
                    resolved_series_id = series_id

                # Store the membership last so callers only receive linked series IDs.
                session.add(
                    SeriesRunsModel(
                        series_id=uint64_to_int64(resolved_series_id),
                        run_id=uint64_to_int64(run_id),
                    )
                )
                session.flush()
                return resolved_series_id
        except IntegrityError:
            return None

    def store_automation(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        federation_id: str,
        flwr_aid: str,
        start_run_request: StartRunRequest,
        series_id: int,
        next_run_at: str,
        fixed_interval: int | None = None,
        max_runs: int | None = None,
    ) -> Automation:
        """Store an automation and return its metadata."""
        current = now()
        stmt = (
            insert(AutomationModel)
            .values(
                federation_id=federation_id,
                status=AutomationStatus.ACTIVE,
                series_id=uint64_to_int64(series_id),
                flwr_aid=flwr_aid,
                start_run_request=start_run_request.SerializeToString(),
                created_at=current,
                updated_at=current,
                next_run_at=datetime.fromisoformat(next_run_at),
                fixed_interval=fixed_interval,
                remaining_runs=max_runs,
                stopped_at=None,
            )
            .returning(AutomationModel)
        )
        try:
            with self.session() as session:
                automation = session.scalars(stmt).one()
                return _automation_from_model(automation)
        except IntegrityError as exc:
            raise ValueError(f"Could not store automation: {exc}") from exc

    def claim_automation(
        self,
        automation_id: int,
        *,
        previous_next_run_at: str,
        next_run_at: str | None,
    ) -> tuple[StartRunRequest, str] | None:
        """Claim an automation occurrence and return its unresolved run request."""
        stored_automation_id = uint64_to_int64(automation_id)
        query = select(
            AutomationModel.start_run_request,
            AutomationModel.flwr_aid,
        ).where(
            AutomationModel.automation_id == stored_automation_id,
            AutomationModel.status == AutomationStatus.ACTIVE,
            AutomationModel.start_run_request.is_not(None),
            AutomationModel.next_run_at == datetime.fromisoformat(previous_next_run_at),
            or_(
                AutomationModel.remaining_runs.is_(None),
                AutomationModel.remaining_runs > 0,
            ),
        )
        if next_run_at is None:
            query = query.where(AutomationModel.remaining_runs <= 1)

        with self.session() as session:
            row = session.execute(query).first()
            if row is None or not self.advance_automation(
                automation_id,
                previous_next_run_at=previous_next_run_at,
                next_run_at=next_run_at,
            ):
                return None

            start_run_request, flwr_aid = row
            request = StartRunRequest()
            request.ParseFromString(start_run_request)
            return request, flwr_aid

    def list_automations(  # pylint: disable=too-many-arguments,too-many-locals,too-many-boolean-expressions
        self,
        *,
        automation_ids: Sequence[int] | None = None,
        federations: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
        due_before: datetime | None = None,
        order_by: Literal["next_run_at", "updated_at"],
        limit: int | None = None,
    ) -> Sequence[Automation]:
        """Return automations matching the given filters."""
        if limit is not None and limit < 0:
            raise AssertionError("`limit` must be >= 0")
        if (
            limit == 0
            or (automation_ids is not None and not automation_ids)
            or (federations is not None and not federations)
            or (statuses is not None and not statuses)
        ):
            return []

        query = select(AutomationModel)
        if automation_ids is not None:
            query = query.where(
                AutomationModel.automation_id.in_(
                    [uint64_to_int64(automation_id) for automation_id in automation_ids]
                )
            )
        if federations is not None:
            query = query.where(AutomationModel.federation_id.in_(federations))
        if statuses is not None:
            query = query.where(AutomationModel.status.in_(statuses))
        if due_before is not None:
            query = query.where(
                AutomationModel.next_run_at <= due_before,
                or_(
                    AutomationModel.remaining_runs.is_(None),
                    AutomationModel.remaining_runs > 0,
                ),
            )

        if order_by == "next_run_at":
            query = query.order_by(
                AutomationModel.next_run_at.asc(), AutomationModel.automation_id.asc()
            )
        else:
            query = query.order_by(
                AutomationModel.updated_at.desc(), AutomationModel.automation_id.desc()
            )
        if limit is not None:
            query = query.limit(limit)

        with self.session() as session:
            automations = session.scalars(
                query.execution_options(populate_existing=True)
            ).all()
            return [_automation_from_model(model) for model in automations]

    def stop_automation(self, automation_id: int) -> bool:
        """Stop an active automation."""
        stopped_at = now()
        stmt = (
            update(AutomationModel)
            .where(
                AutomationModel.automation_id == uint64_to_int64(automation_id),
                AutomationModel.status == AutomationStatus.ACTIVE,
            )
            .values(
                status=AutomationStatus.STOPPED,
                updated_at=stopped_at,
                stopped_at=stopped_at,
            )
            .returning(AutomationModel.automation_id)
        )
        with self.session() as session:
            return session.scalar(stmt) is not None

    def advance_automation(
        self,
        automation_id: int,
        *,
        previous_next_run_at: str,
        next_run_at: str | None,
    ) -> bool:
        """Advance an active automation occurrence."""
        timestamp = now()
        next_run_at_dt = (
            datetime.fromisoformat(next_run_at) if next_run_at is not None else None
        )
        stmt = (
            update(AutomationModel)
            .where(
                AutomationModel.automation_id == uint64_to_int64(automation_id),
                AutomationModel.status == AutomationStatus.ACTIVE,
                AutomationModel.next_run_at
                == datetime.fromisoformat(previous_next_run_at),
                or_(
                    AutomationModel.remaining_runs.is_(None),
                    AutomationModel.remaining_runs > 0,
                ),
            )
            .values(
                updated_at=timestamp,
                next_run_at=case(
                    (
                        AutomationModel.remaining_runs.is_not(None)
                        & (AutomationModel.remaining_runs <= 1),
                        AutomationModel.next_run_at,
                    ),
                    else_=literal(
                        next_run_at_dt, type_=AutomationModel.next_run_at.type
                    ),
                ),
                remaining_runs=case(
                    (AutomationModel.remaining_runs.is_(None), None),
                    (
                        AutomationModel.remaining_runs > 0,
                        AutomationModel.remaining_runs - 1,
                    ),
                    else_=0,
                ),
            )
            .returning(AutomationModel.automation_id)
        )
        if next_run_at is None:
            stmt = stmt.where(AutomationModel.remaining_runs <= 1)

        with self.session() as session:
            return session.scalar(stmt) is not None

    def finish_automation(
        self,
        automation_id: int,
        *,
        status: Literal[AutomationStatus.COMPLETED, AutomationStatus.FAILED],
    ) -> bool:
        """Finish an active automation with a terminal status."""
        stmt = (
            update(AutomationModel)
            .where(
                AutomationModel.automation_id == uint64_to_int64(automation_id),
                AutomationModel.status == AutomationStatus.ACTIVE,
            )
            .values(status=status, updated_at=now())
            .returning(AutomationModel.automation_id)
        )
        if status == AutomationStatus.COMPLETED:
            stmt = stmt.where(AutomationModel.remaining_runs == 0)

        with self.session() as session:
            return session.scalar(stmt) is not None

    def add_task_log(self, task_id: int, log_message: str) -> None:
        """Add a log entry to the task logs for the specified `task_id`."""
        sint64_task_id = uint64_to_int64(task_id)

        try:
            with self.session() as session:
                session.execute(
                    insert(TaskLogsTable).values(
                        timestamp=now().timestamp(),
                        task_id=sint64_task_id,
                        log=log_message,
                    )
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
        with self.session() as session:
            rows = (
                session.execute(
                    select(TaskLogsTable.c.log, TaskLogsTable.c.timestamp)
                    .where(
                        TaskLogsTable.c.task_id == sint64_task_id,
                        TaskLogsTable.c.timestamp > after_timestamp,
                    )
                    .order_by(TaskLogsTable.c.timestamp)
                )
                .mappings()
                .all()
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

        task_values = select(
            literal(sint64_task_id, type_=TaskModel.task_id.type),
            literal(task_type, type_=TaskModel.type.type),
            literal(uint64_to_int64(run_id), type_=TaskModel.run_id.type),
            literal(fab_hash, type_=TaskModel.fab_hash.type),
            literal(model_ref, type_=TaskModel.model_ref.type),
            literal(connector_ref, type_=TaskModel.connector_ref.type),
            literal(now(), type_=TaskModel.pending_at.type),
        )
        if requesting_task_id is not None:
            sint64_requesting_task_id = uint64_to_int64(requesting_task_id)
            task_values = task_values.where(
                select(TaskModel.task_id)
                .where(
                    TaskModel.task_id == sint64_requesting_task_id,
                    TaskModel.finished_at.is_(None),
                )
                .exists()
            )

        insert_stmt = (
            insert(TaskModel)
            .from_select(
                [
                    TaskModel.task_id,
                    TaskModel.type,
                    TaskModel.run_id,
                    TaskModel.fab_hash,
                    TaskModel.model_ref,
                    TaskModel.connector_ref,
                    TaskModel.pending_at,
                ],
                task_values,
            )
            .returning(TaskModel.task_id)
        )

        with self.session() as session:
            try:
                return task_id if session.scalar(insert_stmt) is not None else None
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

        query = select(TaskModel)

        if task_ids is not None:
            if not task_ids:
                return []
            sint64_task_ids = [uint64_to_int64(task_id) for task_id in task_ids]
            query = query.where(TaskModel.task_id.in_(sint64_task_ids))

        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            query = query.where(TaskModel.run_id.in_(sint64_run_ids))

        if statuses is not None:
            if not statuses:
                return []
            status_conditions = []
            for status in STATUS_CONDITIONS:
                if status in statuses:
                    status_conditions.append(_task_status_filter(status))
            if not status_conditions:
                return []
            query = query.where(or_(*status_conditions))

        if order_by is not None:
            order_column = (
                TaskModel.pending_at.asc() if ascending else TaskModel.pending_at.desc()
            )
            query = query.order_by(order_column)
        if limit is not None:
            query = query.limit(limit)
        query = query.execution_options(populate_existing=True)

        with self.session() as session:
            # Clean up expired task tokens before querying tasks
            self._cleanup_expired_task_tokens()
            rows = session.scalars(query).all()
            return [task_from_model(row) for row in rows]

    def get_metadata(self) -> MetaData:
        """Return SQLAlchemy MetaData needed for CoreState tables."""
        return create_corestate_metadata()

    def add_task_usage(self, task_id: int, usage: TaskUsage) -> None:
        """Record usage for the specified task."""
        sint64_task_id = uint64_to_int64(task_id)
        usage_values = select(
            TaskModel.run_id,
            TaskModel.task_id,
            literal(usage.input_tokens, type_=TaskUsageModel.input_tokens.type),
            literal(usage.output_tokens, type_=TaskUsageModel.output_tokens.type),
            literal(usage.total_tokens, type_=TaskUsageModel.total_tokens.type),
            literal(usage.usage_type, type_=TaskUsageModel.usage_type.type),
            literal(usage.provider, type_=TaskUsageModel.provider.type),
            literal(now(), type_=TaskUsageModel.created_at.type),
        ).where(TaskModel.task_id == sint64_task_id)
        stmt = insert(TaskUsageModel).from_select(
            [
                TaskUsageModel.run_id,
                TaskUsageModel.task_id,
                TaskUsageModel.input_tokens,
                TaskUsageModel.output_tokens,
                TaskUsageModel.total_tokens,
                TaskUsageModel.usage_type,
                TaskUsageModel.provider,
                TaskUsageModel.created_at,
            ],
            usage_values,
        )

        with self.session() as session:
            session.execute(stmt)

    def get_task_usage(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        task_ids: Sequence[int] | None = None,
    ) -> Sequence[TaskUsage]:
        """Retrieve task usage records based on the specified filters."""
        query = select(TaskUsageModel).order_by(TaskUsageModel.id.asc())

        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            query = query.where(TaskUsageModel.run_id.in_(sint64_run_ids))

        if task_ids is not None:
            if not task_ids:
                return []
            sint64_task_ids = [uint64_to_int64(task_id) for task_id in task_ids]
            query = query.where(TaskUsageModel.task_id.in_(sint64_task_ids))

        with self.session() as session:
            rows = session.scalars(query).all()
            return [_task_usage_from_model(row) for row in rows]

    def claim_task(self, task_id: int) -> str | None:
        """Atomically claim a pending task."""
        token = secrets.token_hex(FLWR_TASK_TOKEN_LENGTH)
        claimed_at = now()
        active_until = claimed_at + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL)
        sint64_task_id = uint64_to_int64(task_id)
        try:
            # The conditional UPDATE is the atomic claim: exactly one caller can
            # move a pending, unclaimed task to STARTING and attach a token.
            with self.session() as session:
                claimed_task_id = session.scalar(
                    update(TaskModel)
                    .where(
                        TaskModel.task_id == sint64_task_id,
                        TaskModel.token.is_(None),
                        _task_status_filter(Status.PENDING),
                    )
                    .values(
                        token=token,
                        active_until=active_until,
                        starting_at=claimed_at,
                    )
                    .returning(TaskModel.task_id)
                )
            if claimed_task_id is None:
                return None

            return token
        except IntegrityError:
            # Rare failure: generated token already exists (duplicate)
            return None

    def activate_task(self, task_id: int) -> bool:
        """Move a task from starting to running."""
        # Expire non-responsive tasks before transitioning task status.

        with self.session() as session:
            self._cleanup_expired_task_tokens()
            activated_at = now()
            active_until = activated_at + timedelta(
                seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL
            )

            # Activation is a strict STARTING -> RUNNING transition.
            activated_task_id = session.scalar(
                update(TaskModel)
                .where(
                    TaskModel.task_id == uint64_to_int64(task_id),
                    _task_status_filter(Status.STARTING),
                )
                .values(running_at=activated_at, active_until=active_until)
                .returning(TaskModel.task_id)
            )
        return activated_task_id is not None

    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished."""
        if sub_status not in (SubStatus.COMPLETED, SubStatus.STOPPED, SubStatus.FAILED):
            err = f"Invalid sub_status '{sub_status}' for finishing task {task_id}"
            log(ERROR, err)
            return False

        sint64_task_id = uint64_to_int64(task_id)
        with self.session() as session:
            self._cleanup_expired_task_tokens()
            query = update(TaskModel).where(
                TaskModel.task_id == sint64_task_id,
                TaskModel.finished_at.is_(None),
            )
            # FINISHED:COMPLETED is only valid from RUNNING.
            if sub_status == SubStatus.COMPLETED:
                query = query.where(TaskModel.running_at.is_not(None))

            finished_task_id = session.scalar(
                query.values(
                    finished_at=now(),
                    sub_status=sub_status,
                    details=details,
                    active_until=None,
                    token=None,
                ).returning(TaskModel.task_id)
            )
            return finished_task_id is not None

    def acknowledge_task_heartbeat(self, task_id: int) -> bool:
        """Extend heartbeat state for the claimed task."""
        # Heartbeats are accepted only for active, unexpired task claims.
        with self.session() as session:
            current = now()
            ttl = timedelta(seconds=HEARTBEAT_PATIENCE * HEARTBEAT_DEFAULT_INTERVAL)
            self._cleanup_expired_task_tokens()
            acknowledged_task_id = session.scalar(
                update(TaskModel)
                .where(
                    TaskModel.task_id == uint64_to_int64(task_id),
                    TaskModel.active_until >= current,
                    TaskModel.finished_at.is_(None),
                )
                .values(active_until=current + ttl)
                .returning(TaskModel.task_id)
            )
        return acknowledged_task_id is not None

    def get_task_by_token(self, token: str) -> Task | None:
        """Return the task associated with the task token, if valid."""
        current = now()
        with self.session() as session:
            row = session.scalars(
                select(TaskModel)
                .where(
                    TaskModel.token == token,
                    TaskModel.finished_at.is_(None),
                )
                .execution_options(populate_existing=True)
            ).first()
            if row is None or row.active_until is None or row.active_until < current:
                return None
            return task_from_model(row)

    def store_task_message(self, message: Message) -> bool:
        """Store one task-addressed Message."""
        if validate_task_message(message):
            return False

        with self.session() as session:
            self._cleanup_expired_task_tokens()
            metadata = message.metadata
            stored_run_id = uint64_to_int64(metadata.run_id)
            stored_src_task_id = uint64_to_int64(cast(int, metadata.src_task_id))
            stored_dst_task_id = uint64_to_int64(cast(int, metadata.dst_task_id))
            serialized_content = (
                recorddict_to_proto(message.content).SerializeToString()
                if message.has_content()
                else None
            )
            serialized_error = (
                error_to_proto(message.error).SerializeToString()
                if message.has_error()
                else None
            )
            src_task_exists = exists().where(
                TaskModel.task_id == stored_src_task_id,
                TaskModel.run_id == stored_run_id,
            )
            dst_task_exists = exists().where(
                TaskModel.task_id == stored_dst_task_id,
                TaskModel.run_id == stored_run_id,
                TaskModel.finished_at.is_(None),
            )
            task_message_values = select(
                literal(metadata.message_id, type_=TaskMessageModel.message_id.type),
                literal(stored_run_id, type_=TaskMessageModel.run_id.type),
                literal(stored_src_task_id, type_=TaskMessageModel.src_task_id.type),
                literal(stored_dst_task_id, type_=TaskMessageModel.dst_task_id.type),
                literal(
                    metadata.reply_to_message_id,
                    type_=TaskMessageModel.reply_to_message_id.type,
                ),
                literal(metadata.created_at, type_=TaskMessageModel.created_at.type),
                literal(metadata.ttl, type_=TaskMessageModel.ttl.type),
                literal(
                    metadata.message_type, type_=TaskMessageModel.message_type.type
                ),
                literal(serialized_content, type_=TaskMessageModel.content.type),
                literal(serialized_error, type_=TaskMessageModel.error.type),
            ).where(src_task_exists, dst_task_exists)
            stmt = (
                insert(TaskMessageModel)
                .from_select(
                    [
                        TaskMessageModel.message_id,
                        TaskMessageModel.run_id,
                        TaskMessageModel.src_task_id,
                        TaskMessageModel.dst_task_id,
                        TaskMessageModel.reply_to_message_id,
                        TaskMessageModel.created_at,
                        TaskMessageModel.ttl,
                        TaskMessageModel.message_type,
                        TaskMessageModel.content,
                        TaskMessageModel.error,
                    ],
                    task_message_values,
                )
                .returning(TaskMessageModel.message_id)
            )
            try:
                inserted_message_id = session.scalar(stmt)
            except IntegrityError:
                return False
            return inserted_message_id is not None

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
            rows = self._claim_task_message_models(dst_task_ids, order_by, limit)
            snapshots = [_task_message_snapshot_from_model(row) for row in rows]
        return [_task_message_from_snapshot(row) for row in snapshots]

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
        event_rows = [
            {
                "timestamp": current,
                "run_id": uint64_to_int64(event.run_id),
                "task_id": uint64_to_int64(event.task_id),
                "event": event.event,
                "data": event.data,
            }
            for event in events
        ]

        with self.session() as session:
            session.execute(insert(TaskEventModel), event_rows)

        return True

    def get_task_events(
        self,
        *,
        run_id: int | None = None,
        after_task_event_id: int | None = None,
    ) -> Sequence[TaskEvent]:
        """Return task-produced run events after the cursor."""
        cursor = after_task_event_id if after_task_event_id is not None else 0
        query = (
            select(TaskEventModel)
            .where(TaskEventModel.id > cursor)
            .order_by(TaskEventModel.id.asc())
        )
        if run_id is not None:
            query = query.where(TaskEventModel.run_id == uint64_to_int64(run_id))

        with self.session() as session:
            rows = session.scalars(query).all()
            return [_task_event_from_model(row) for row in rows]

    def _claim_task_message_models(
        self,
        dst_task_ids: Sequence[int] | None,
        order_by: Literal["created_at"] | None,
        limit: int | None,
    ) -> list[TaskMessageModel]:
        """Atomically claim eligible task Messages."""
        query = select(TaskMessageModel.message_id)

        # Filter by destination task IDs
        if dst_task_ids is not None:
            sint64_dst_task_ids = [uint64_to_int64(t) for t in dst_task_ids]
            query = query.where(TaskMessageModel.dst_task_id.in_(sint64_dst_task_ids))
        if order_by is not None:
            query = query.order_by(TaskMessageModel.created_at.asc())
        if limit is not None:
            query = query.limit(limit)

        if order_by is not None or limit is not None:
            # Materialize candidates before deleting. Some backends can otherwise
            # re-evaluate same-table subqueries while DELETE scans rows.
            if self.select_lock_sql:
                if self.select_lock_sql.strip().upper() != "FOR UPDATE SKIP LOCKED":
                    raise NotImplementedError(
                        "Custom select_lock_sql values are not supported for ORM "
                        "task_message claims."
                    )
                query = query.with_for_update(skip_locked=True)
            selected = query.cte("selected")
            delete_query = delete(TaskMessageModel).where(
                TaskMessageModel.message_id.in_(select(selected.c.message_id))
            )
        else:
            delete_query = delete(TaskMessageModel)
            if dst_task_ids is not None:
                sint64_dst_task_ids = [uint64_to_int64(t) for t in dst_task_ids]
                delete_query = delete_query.where(
                    TaskMessageModel.dst_task_id.in_(sint64_dst_task_ids)
                )

        returning_query = delete_query.returning(TaskMessageModel)
        with self.session() as session:
            rows = list(session.scalars(returning_query))

        # Sort claimed rows in memory if requested. `ORDER BY` in the candidate
        # query determines which rows are claimed, but SQL does not guarantee that
        # `DELETE ... RETURNING` returns them in that order.
        if order_by is not None:
            rows.sort(key=lambda row: row.created_at)

        return rows

    def _cleanup_expired_task_tokens(self) -> None:
        """Remove expired task heartbeat records.

        Expired starting tasks are moved back to pending. Expired running tasks
        are marked as finished with a failed status. Tokens are removed in both
        cases.
        """
        expired_at = now()
        with self.session() as session:
            # Claims that never reached RUNNING are retryable launch failures.
            session.execute(
                update(TaskModel)
                .where(
                    TaskModel.token.is_not(None),
                    TaskModel.active_until < expired_at,
                    _task_status_filter(Status.STARTING),
                )
                .values(
                    token=None,
                    active_until=None,
                    starting_at=None,
                    sub_status="",
                    details="",
                )
            )

            # Expired running task claims are terminal failures and lose their token.
            expired_tasks = [
                task_from_model(row)
                for row in session.scalars(
                    update(TaskModel)
                    .where(
                        TaskModel.token.is_not(None),
                        TaskModel.active_until < expired_at,
                        _task_status_filter(Status.RUNNING),
                    )
                    .values(
                        token=None,
                        finished_at=TaskModel.active_until,
                        active_until=None,
                        sub_status=SubStatus.FAILED,
                        details="No heartbeat received from the task",
                    )
                    .returning(TaskModel)
                ).all()
            ]
        if expired_tasks:
            self._on_task_tokens_expired(expired_tasks)

    def _cleanup_invalid_task_messages(self) -> None:
        """Remove expired task Messages."""
        with self.session() as session:
            session.execute(
                delete(TaskMessageModel).where(
                    (TaskMessageModel.created_at + TaskMessageModel.ttl)
                    <= now().timestamp()
                )
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
        with self.session() as session:
            session.execute(
                delete(NonceStoreModel).where(
                    NonceStoreModel.expires_at < now().timestamp()
                )
            )
            stmt = (
                self.dialect_insert(NonceStoreModel)
                .values(namespace=namespace, nonce=nonce, expires_at=expires_at)
                .on_conflict_do_nothing(
                    index_elements=[NonceStoreModel.namespace, NonceStoreModel.nonce]
                )
                .returning(NonceStoreModel.nonce)
            )
            return session.scalar(stmt) is not None


def _connector_oauth_session_from_model(
    model: ConnectorOAuthSessionModel,
) -> ConnectorOAuthSessionRecord:
    """Convert a connector OAuth session model to its persistence record."""
    return ConnectorOAuthSessionRecord(
        oauth_session_id=model.oauth_session_id,
        flwr_aid=model.flwr_aid,
        connector_ref=model.connector_ref,
        state=model.state,
        redirect_uri=model.redirect_uri,
        pkce_verifier=model.pkce_verifier,
        created_at=_timestamp_to_iso_assuming_utc(model.created_at),
        expires_at=_timestamp_to_iso_assuming_utc(model.expires_at),
        completed_at=_timestamp_to_iso_assuming_utc(model.completed_at) or None,
    )


def _timestamp_to_iso_assuming_utc(value: datetime | str | None) -> str:
    """Return an ISO string, treating naïve SQLite timestamps as UTC."""
    if isinstance(value, datetime) and value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return timestamp_to_iso(value)


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


def _determine_task_model_status(model: TaskModel) -> TaskStatus:
    """Determine the status of the task based on timestamp fields."""
    if model.pending_at:
        if model.finished_at:
            return TaskStatus(
                status=Status.FINISHED,
                sub_status=model.sub_status,
                details=model.details,
            )
        if model.starting_at:
            if model.running_at:
                return TaskStatus(status=Status.RUNNING, sub_status="", details="")
            return TaskStatus(status=Status.STARTING, sub_status="", details="")
        return TaskStatus(status=Status.PENDING, sub_status="", details="")
    task_id = int64_to_uint64(model.task_id)
    raise ValueError(f"The task {task_id} does not have a valid status.")


def _task_status_filter(status: str) -> Any:
    """Return the ORM filter expression for a task status."""
    if status == Status.PENDING:
        return TaskModel.starting_at.is_(None) & TaskModel.finished_at.is_(None)
    if status == Status.STARTING:
        return (
            TaskModel.starting_at.is_not(None)
            & TaskModel.running_at.is_(None)
            & TaskModel.finished_at.is_(None)
        )
    if status == Status.RUNNING:
        return TaskModel.running_at.is_not(None) & TaskModel.finished_at.is_(None)
    if status == Status.FINISHED:
        return TaskModel.finished_at.is_not(None)
    raise ValueError(f"Unsupported task status {status!r}.")


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


def task_from_model(model: TaskModel) -> Task:
    """Convert a task ORM model to a Task object."""
    return Task(
        task_id=int64_to_uint64(model.task_id),
        type=model.type,
        run_id=int64_to_uint64(model.run_id),
        pending_at=timestamp_to_iso(model.pending_at),
        starting_at=timestamp_to_iso(model.starting_at),
        running_at=timestamp_to_iso(model.running_at),
        finished_at=timestamp_to_iso(model.finished_at),
        status=_determine_task_model_status(model),
        fab_hash=model.fab_hash,
        model_ref=model.model_ref,
        connector_ref=model.connector_ref,
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


def _run_series_from_model(model: RunSeriesModel) -> RunSeries:
    """Convert a run series ORM model to a RunSeries object."""
    return RunSeries(
        series_id=int64_to_uint64(model.series_id),
        federation=model.federation_id,
        description=model.description or "",
        created_at=timestamp_to_iso(model.created_at),
        updated_at=timestamp_to_iso(model.updated_at),
    )


def _task_usage_from_model(model: TaskUsageModel) -> TaskUsage:
    """Convert a task_usage ORM model to a TaskUsage proto."""
    return TaskUsage(
        usage_type=model.usage_type,
        provider=model.provider,
        input_tokens=model.input_tokens,
        output_tokens=model.output_tokens,
        total_tokens=model.total_tokens,
    )


def _task_event_from_model(model: TaskEventModel) -> TaskEvent:
    """Convert a task_event ORM model to a TaskEvent proto."""
    return TaskEvent(
        id=model.id,
        timestamp=timestamp_to_iso(model.timestamp),
        run_id=int64_to_uint64(model.run_id),
        task_id=int64_to_uint64(model.task_id),
        event=model.event,
        data=model.data,
    )


def _automation_from_model(model: AutomationModel) -> Automation:
    """Convert an automation model to its proto representation."""
    return Automation(
        automation_id=model.automation_id,
        status=model.status,
        federation=model.federation_id,
        series_id=int64_to_uint64(model.series_id),
        flwr_aid=model.flwr_aid,
        created_at=timestamp_to_iso(model.created_at),
        updated_at=timestamp_to_iso(model.updated_at),
        next_run_at=timestamp_to_iso(model.next_run_at),
        fixed_interval=model.fixed_interval,
        remaining_runs=model.remaining_runs,
        stopped_at=timestamp_to_iso(model.stopped_at) if model.stopped_at else None,
    )


def _task_message_snapshot_from_model(model: TaskMessageModel) -> dict[str, Any]:
    """Snapshot a claimed task_message model before the transaction commits."""
    return {
        "message_id": model.message_id,
        "run_id": model.run_id,
        "src_task_id": model.src_task_id,
        "dst_task_id": model.dst_task_id,
        "reply_to_message_id": model.reply_to_message_id,
        "created_at": model.created_at,
        "ttl": model.ttl,
        "message_type": model.message_type,
        "content": model.content,
        "error": model.error,
    }


def _task_message_from_snapshot(row: dict[str, Any]) -> Message:
    """Convert a claimed task_message snapshot to a Message."""
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
