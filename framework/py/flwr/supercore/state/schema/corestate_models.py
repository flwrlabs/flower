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
"""SQLAlchemy declarative model definitions for CoreState."""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Double,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    Text,
    UniqueConstraint,
    column,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from flwr.supercore.state.schema.types import UInt64, UTCDateTime


class FlwrBase(DeclarativeBase):
    """Base class for Flower OSS state models."""

    metadata = MetaData()


class NonceStore(FlwrBase):
    """Represent stored nonces for replay protection."""

    __tablename__ = "nonce_store"
    __table_args__ = (Index("idx_nonce_store_expires_at", "expires_at"),)

    namespace: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    nonce: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)


class Fab(FlwrBase):
    """Represent stored FAB contents."""

    __tablename__ = "fab"

    fab_hash: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    verifications: Mapped[str] = mapped_column(String, nullable=False)


class Message(FlwrBase):
    """Represent message delivery with retained metadata.

    Delivery state is inferred from the flag, deadline, and closing timestamps:

    - Not cleared, deadline in the future: pending and eligible for retrieval.
    - Not cleared, deadline reached: expired, awaiting passive cleanup.
    - Cleared, acknowledged_at set: acknowledged.
    - Cleared, cancelled_at set: cancelled.
    - Cleared, neither closing timestamp set: expired and cleaned up.

    The expires_at deadline is always set, regardless of the closing outcome.
    Clearing ends delivery and releases the ObjectStore reference while retaining
    metadata. Acknowledgment and cancellation timestamps are mutually exclusive.
    """

    __tablename__ = "message"
    __table_args__ = (
        CheckConstraint(
            "acknowledged_at IS NULL OR cancelled_at IS NULL",
            name="ck_message_terminal_timestamps_exclusive",
        ),
        CheckConstraint(
            "cleared OR (acknowledged_at IS NULL AND cancelled_at IS NULL)",
            name="ck_message_terminal_timestamps_require_cleared",
        ),
        Index(
            "idx_message_pending_dst_created_at",
            "dst_node_id",
            "created_at",
            "message_id",
            postgresql_where=column("cleared", Boolean).is_(False),
            sqlite_where=column("cleared", Boolean).is_(False),
        ),
        Index(
            "idx_message_pending_dst_run_created_at",
            "dst_node_id",
            "run_id",
            "created_at",
            "message_id",
            postgresql_where=column("cleared", Boolean).is_(False),
            sqlite_where=column("cleared", Boolean).is_(False),
        ),
        Index(
            "idx_message_pending_run_id",
            "run_id",
            postgresql_where=column("cleared", Boolean).is_(False),
            sqlite_where=column("cleared", Boolean).is_(False),
        ),
        Index("idx_message_reply_to_message_id", "reply_to_message_id"),
    )

    message_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    run_id: Mapped[int] = mapped_column(UInt64(), nullable=False)
    src_node_id: Mapped[int] = mapped_column(UInt64(), nullable=False)
    dst_node_id: Mapped[int] = mapped_column(UInt64(), nullable=False)
    src_task_id: Mapped[int | None] = mapped_column(UInt64(), nullable=True)
    dst_task_id: Mapped[int | None] = mapped_column(UInt64(), nullable=True)
    reply_to_message_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("message.message_id", ondelete="RESTRICT"), nullable=True
    )
    group_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[float] = mapped_column(Double, nullable=False)
    ttl: Mapped[float] = mapped_column(Double, nullable=False)
    message_type: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    cleared: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime(), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class Agent(FlwrBase):
    """Represent a persistent agent owned by a Flower account.

    ``app_spec`` identifies the AgentApp and an optional version.
    """

    __tablename__ = "agent"
    __table_args__ = (
        Index("idx_agent_flwr_aid", "flwr_aid"),
        {"sqlite_autoincrement": True},
    )

    agent_id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    app_spec: Mapped[str] = mapped_column(String, nullable=False)
    flwr_aid: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class RunSeries(FlwrBase):
    """Represent a run series."""

    __tablename__ = "run_series"

    series_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    federation_id: Mapped[str] = mapped_column(String, nullable=False)
    is_agent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class SeriesContext(FlwrBase):
    """Represent context data for a run series."""

    __tablename__ = "series_context"

    series_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    context: Mapped[bytes | None] = mapped_column(LargeBinary)


class CompactedContext(FlwrBase):
    """Represent a compacted event context for a run series.

    ``through_event_id`` is an inclusive event high-water mark for the series.
    """

    __tablename__ = "compacted_contexts"
    __table_args__ = (
        UniqueConstraint("series_id", "through_event_id"),
        Index("idx_compacted_contexts_series_id_id", "series_id", "id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    series_id: Mapped[int] = mapped_column(UInt64(), nullable=False)
    through_event_id: Mapped[int] = mapped_column(UInt64(), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(UInt64(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class SeriesRuns(FlwrBase):
    """Represent the runs belonging to a run series."""

    __tablename__ = "series_runs"
    __table_args__ = (Index("idx_series_runs_series_id", "series_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)


class Automation(FlwrBase):
    """Represent an automation schedule."""

    __tablename__ = "automation"
    __table_args__ = (
        Index("idx_automation_status_next_run_at", "status", "next_run_at"),
        Index(
            "idx_automation_federation_id_status_updated_at",
            "federation_id",
            "status",
            "updated_at",
        ),
    )

    automation_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, nullable=False
    )
    federation_id: Mapped[str] = mapped_column(String, nullable=False)
    flwr_aid: Mapped[str] = mapped_column(String, nullable=False)
    series_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    start_run_request: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    next_run_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    fixed_interval: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    remaining_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class FederationApp(FlwrBase):
    """Represent an app associated with a federation."""

    __tablename__ = "federation_app"
    __table_args__ = (
        Index(
            "idx_federation_app_federation_id_added_at",
            "federation_id",
            "added_at",
        ),
        Index(
            "idx_federation_app_app_id_federation_id",
            "app_id",
            "federation_id",
        ),
    )

    federation_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    app_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    fab_hash: Mapped[str] = mapped_column(String, nullable=False)
    app_type: Mapped[str] = mapped_column(String, nullable=False)
    is_hub_app: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    display_name: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    color: Mapped[str | None] = mapped_column(String, nullable=True)
    added_by: Mapped[str] = mapped_column(String, nullable=False)
    added_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class Connector(FlwrBase):
    """Represent connector configuration for a federation."""

    __tablename__ = "connector"
    __table_args__ = ({"sqlite_autoincrement": True},)

    connector_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True, nullable=False
    )
    federation_id: Mapped[str] = mapped_column(String, nullable=False)
    connector_ref: Mapped[str] = mapped_column(String, nullable=False)
    credentials_json: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False)
    deleted_by: Mapped[str | None] = mapped_column(String, nullable=True)


class ConnectorOAuthSession(FlwrBase):
    """Represent an OAuth authorization session for a connector."""

    __tablename__ = "connector_oauth_session"

    oauth_session_id: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    flwr_aid: Mapped[str] = mapped_column(String, nullable=False)
    federation_id: Mapped[str] = mapped_column(String, nullable=False)
    connector_ref: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    redirect_uri: Mapped[str] = mapped_column(String, nullable=False)
    pkce_verifier: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


class RunConnector(FlwrBase):
    """Represent the connectors attached to a run."""

    __tablename__ = "run_connector"

    run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    connector_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)


class Task(FlwrBase):
    """Represent a task."""

    __tablename__ = "task"
    __table_args__ = (
        Index("idx_task_run_id", "run_id"),
        Index("idx_task_token", "token"),
        Index("idx_task_active_until", "active_until"),
    )

    task_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    type: Mapped[str] = mapped_column(String, nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    fab_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    model_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    connector_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    connector_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token: Mapped[str | None] = mapped_column(String, nullable=True)
    active_until: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    pending_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    starting_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    running_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    sub_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("''")
    )
    details: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("''")
    )

    __mapper_args__ = {"primary_key": [task_id]}


class TaskEvent(FlwrBase):
    """Represent a task event."""

    __tablename__ = "task_event"
    __table_args__ = (
        Index("idx_task_event_run_id_id", "run_id", "id"),
        Index("idx_task_event_task_id", "task_id"),
    )

    id: Mapped[int] = mapped_column(
        # Both are signed 64-bit; SQLite requires INTEGER for autoincrement.
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    task_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("task.task_id"), nullable=False
    )
    event: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[str] = mapped_column(String, nullable=False)


class TaskMessage(FlwrBase):
    """Represent a pending task message.

    Messages are removed when retrieved or expired. An empty table is normal
    when no messages are waiting to be picked up.
    """

    __tablename__ = "task_message"
    __table_args__ = (
        Index("idx_task_message_dst_task_id_created_at", "dst_task_id", "created_at"),
        Index("idx_task_message_run_id", "run_id"),
    )

    message_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    src_task_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("task.task_id"), nullable=False
    )
    dst_task_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("task.task_id"), nullable=False
    )
    reply_to_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float] = mapped_column(Float, nullable=False)
    ttl: Mapped[float] = mapped_column(Float, nullable=False)
    message_type: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    error: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


TaskLogsTable = Table(
    "task_logs",
    FlwrBase.metadata,
    Column("timestamp", Float, nullable=False),
    Column("task_id", BigInteger, ForeignKey("task.task_id"), nullable=False),
    Column("log", String, nullable=False),
    Index("idx_task_logs_task_id_timestamp", "task_id", "timestamp"),
)


class ObjectPushSession(FlwrBase):
    """Represent an object push session."""

    __tablename__ = "object_push_sessions"
    __table_args__ = (
        Index("idx_object_push_sessions_run_id", "run_id"),
        Index("idx_object_push_sessions_expires_at", "expires_at"),
    )

    session_id: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    pending_count: Mapped[int] = mapped_column(Integer, nullable=False)


class ObjectPushSessionRoot(FlwrBase):
    """Represent a root object for an object push session.

    These records are removed when uploads complete or sessions are cleaned up,
    so the table may be empty when no uploads are in progress.
    """

    __tablename__ = "object_push_session_roots"
    __table_args__ = (Index("idx_object_push_session_roots_session_id", "session_id"),)

    session_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("object_push_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    root_object_id: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )


class ObjectPushSessionPending(FlwrBase):
    """Represent a pending object for an object push session.

    Entries are removed as objects are uploaded. The table may be empty once
    all pending uploads have finished.
    """

    __tablename__ = "object_push_session_pending"
    __table_args__ = (
        PrimaryKeyConstraint("session_id", "object_id"),
        Index(
            "idx_object_push_session_pending_object_id_session_id",
            "object_id",
            "session_id",
        ),
    )

    session_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("object_push_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    object_id: Mapped[str] = mapped_column(String, nullable=False)


class TaskUsage(FlwrBase):
    """Represent reported task usage."""

    __tablename__ = "task_usage"
    __table_args__ = (
        Index("idx_task_usage_run_id", "run_id"),
        Index("idx_task_usage_task_id", "task_id"),
        Index("idx_task_usage_reported_at", "reported_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    task_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("task.task_id"), nullable=False
    )
    input_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    usage_type: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(
        String, server_default="unknown", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    reported_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
