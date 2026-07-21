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
"""SQLAlchemy Core Table definitions for CoreState."""


from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
    text,
)


def create_corestate_metadata() -> MetaData:
    """Create and return MetaData with CoreState table definitions."""
    metadata = MetaData()

    # --------------------------------------------------------------------------
    #  Table: nonce_store
    # --------------------------------------------------------------------------
    nonce_store = Table(
        "nonce_store",
        metadata,
        Column("namespace", String, primary_key=True, nullable=False),
        Column("nonce", String, primary_key=True, nullable=False),
        Column("expires_at", Float, nullable=False),
    )
    Index("idx_nonce_store_expires_at", nonce_store.c.expires_at)

    # --------------------------------------------------------------------------
    #  Table: fab
    # --------------------------------------------------------------------------
    Table(
        "fab",
        metadata,
        Column("fab_hash", String, primary_key=True),
        Column("content", LargeBinary, nullable=False),
        Column("verifications", String, nullable=False),
    )

    # --------------------------------------------------------------------------
    #  Table: run_series
    # --------------------------------------------------------------------------
    Table(
        "run_series",
        metadata,
        Column("series_id", BigInteger, primary_key=True, nullable=False),
        Column("federation_id", String, nullable=False),
        Column("description", String, nullable=True),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
    )

    # --------------------------------------------------------------------------
    #  Table: series_context
    # --------------------------------------------------------------------------
    Table(
        "series_context",
        metadata,
        Column("series_id", BigInteger, primary_key=True, nullable=False),
        Column("context", LargeBinary),
    )

    # --------------------------------------------------------------------------
    #  Table: series_runs
    # --------------------------------------------------------------------------
    series_runs = Table(
        "series_runs",
        metadata,
        Column(
            "id",
            Integer,
            primary_key=True,
            autoincrement=True,
        ),
        Column("series_id", BigInteger, nullable=False),
        Column("run_id", BigInteger, unique=True, nullable=False),
    )
    Index("idx_series_runs_series_id", series_runs.c.series_id)

    # --------------------------------------------------------------------------
    #  Table: automation
    # --------------------------------------------------------------------------
    automation = Table(
        "automation",
        metadata,
        Column(
            "automation_id",
            Integer,
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        Column("federation_id", String, nullable=False),
        Column("flwr_aid", String, nullable=False),
        Column("series_id", BigInteger, nullable=False),
        Column("status", String, nullable=False),
        Column("fab_id", String, nullable=True),
        Column("fab_version", String, nullable=True),
        Column("fab_hash", String, nullable=True),
        Column("override_config", String, nullable=False),
        Column("federation_config", String, nullable=True),
        Column("primary_task_type", String, nullable=False),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
        Column("next_run_at", TIMESTAMP(timezone=True), nullable=False),
        Column("fixed_interval", BigInteger, nullable=True),
        Column("remaining_runs", Integer, nullable=True),
        Column("stopped_at", TIMESTAMP(timezone=True), nullable=True),
    )
    Index(
        "idx_automation_status_next_run_at",
        automation.c.status,
        automation.c.next_run_at,
    )
    Index(
        "idx_automation_federation_id_status_updated_at",
        automation.c.federation_id,
        automation.c.status,
        automation.c.updated_at,
    )

    # --------------------------------------------------------------------------
    #  Table: connector
    # --------------------------------------------------------------------------
    Table(
        "connector",
        metadata,
        Column("flwr_aid", String, primary_key=True, nullable=False),
        Column("connector_ref", String, primary_key=True, nullable=False),
        Column("credentials_json", String, nullable=False),
        Column("config_json", String, nullable=False),
    )

    # --------------------------------------------------------------------------
    #  Table: connector_oauth_session
    # --------------------------------------------------------------------------
    Table(
        "connector_oauth_session",
        metadata,
        Column("oauth_session_id", String, primary_key=True, nullable=False),
        Column("flwr_aid", String, nullable=False),
        Column("connector_ref", String, nullable=False),
        Column("state", String, nullable=False),
        Column("redirect_uri", String, nullable=False),
        Column("pkce_verifier", String, nullable=True),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        Column("expires_at", TIMESTAMP(timezone=True), nullable=False),
        Column("completed_at", TIMESTAMP(timezone=True), nullable=True),
    )

    # --------------------------------------------------------------------------
    #  Table: run_connector
    # --------------------------------------------------------------------------
    Table(
        "run_connector",
        metadata,
        Column("run_id", BigInteger, primary_key=True, nullable=False),
        Column("connector_ref", String, primary_key=True, nullable=False),
    )

    # --------------------------------------------------------------------------
    #  Table: task
    # --------------------------------------------------------------------------
    task = Table(
        "task",
        metadata,
        Column("task_id", BigInteger, nullable=False, unique=True),
        Column("type", String, nullable=False),
        Column("run_id", BigInteger, nullable=False),
        Column("fab_hash", String, nullable=True),
        Column("model_ref", String, nullable=True),
        Column("connector_ref", String, nullable=True),
        Column("token", String, nullable=True),
        Column("active_until", TIMESTAMP(timezone=True), nullable=True),
        Column("pending_at", TIMESTAMP(timezone=True), nullable=False),
        Column("starting_at", TIMESTAMP(timezone=True), nullable=True),
        Column("running_at", TIMESTAMP(timezone=True), nullable=True),
        Column("finished_at", TIMESTAMP(timezone=True), nullable=True),
        Column("sub_status", String, nullable=False, server_default=text("''")),
        Column("details", String, nullable=False, server_default=text("''")),
    )
    Index("idx_task_run_id", task.c.run_id)
    Index("idx_task_token", task.c.token)
    Index("idx_task_active_until", task.c.active_until)

    # --------------------------------------------------------------------------
    #  Table: task_event
    # --------------------------------------------------------------------------
    task_event = Table(
        "task_event",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("timestamp", TIMESTAMP(timezone=True), nullable=False),
        Column("run_id", BigInteger, nullable=False),
        Column("task_id", BigInteger, ForeignKey("task.task_id"), nullable=False),
        Column("event", String, nullable=False),
        Column("data", String, nullable=False),
    )
    Index("idx_task_event_run_id_id", task_event.c.run_id, task_event.c.id)
    Index("idx_task_event_task_id", task_event.c.task_id)

    # --------------------------------------------------------------------------
    #  Table: task_message
    # --------------------------------------------------------------------------
    task_message = Table(
        "task_message",
        metadata,
        Column("message_id", String, primary_key=True, nullable=False),
        Column("run_id", BigInteger, nullable=False),
        Column("src_task_id", BigInteger, ForeignKey("task.task_id"), nullable=False),
        Column("dst_task_id", BigInteger, ForeignKey("task.task_id"), nullable=False),
        Column("reply_to_message_id", String, nullable=True),
        Column("created_at", Float, nullable=False),
        Column("ttl", Float, nullable=False),
        Column("message_type", String, nullable=False),
        Column("content", LargeBinary, nullable=True),
        Column("error", LargeBinary, nullable=True),
    )
    Index(
        "idx_task_message_dst_task_id_created_at",
        task_message.c.dst_task_id,
        task_message.c.created_at,
    )
    Index("idx_task_message_run_id", task_message.c.run_id)

    # --------------------------------------------------------------------------
    #  Table: object_push_sessions
    # --------------------------------------------------------------------------
    object_push_sessions = Table(
        "object_push_sessions",
        metadata,
        Column("session_id", String, primary_key=True, nullable=False),
        Column("run_id", BigInteger, nullable=False),
        Column("expires_at", TIMESTAMP(timezone=True), nullable=False),
        Column("pending_count", Integer, nullable=False),
    )
    Index("idx_object_push_sessions_run_id", object_push_sessions.c.run_id)
    Index("idx_object_push_sessions_expires_at", object_push_sessions.c.expires_at)

    # --------------------------------------------------------------------------
    #  Table: object_push_session_roots
    # --------------------------------------------------------------------------
    object_push_session_roots = Table(
        "object_push_session_roots",
        metadata,
        Column(
            "session_id",
            String,
            ForeignKey("object_push_sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("root_object_id", String, primary_key=True, nullable=False),
    )
    Index(
        "idx_object_push_session_roots_session_id",
        object_push_session_roots.c.session_id,
    )

    # --------------------------------------------------------------------------
    #  Table: object_push_session_pending
    # --------------------------------------------------------------------------
    object_push_session_pending = Table(
        "object_push_session_pending",
        metadata,
        Column(
            "session_id",
            String,
            ForeignKey("object_push_sessions.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column("object_id", String, nullable=False),
        PrimaryKeyConstraint("session_id", "object_id"),
    )
    Index(
        "idx_object_push_session_pending_object_id_session_id",
        object_push_session_pending.c.object_id,
        object_push_session_pending.c.session_id,
    )

    # --------------------------------------------------------------------------
    #  Table: task_logs
    # --------------------------------------------------------------------------
    task_logs = Table(
        "task_logs",
        metadata,
        Column("timestamp", Float, nullable=False),
        Column("task_id", BigInteger, ForeignKey("task.task_id"), nullable=False),
        Column("log", String, nullable=False),
    )
    Index("idx_task_logs_task_id_timestamp", task_logs.c.task_id, task_logs.c.timestamp)

    # --------------------------------------------------------------------------
    #  Table: task_usage
    # --------------------------------------------------------------------------
    task_usage = Table(
        "task_usage",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", BigInteger, nullable=False),
        Column(
            "task_id",
            BigInteger,
            ForeignKey("task.task_id"),
            nullable=False,
        ),
        Column("input_tokens", BigInteger, nullable=True),
        Column("output_tokens", BigInteger, nullable=True),
        Column("total_tokens", BigInteger, nullable=True),
        Column("usage_type", String, nullable=False),
        Column("provider", String, server_default="unknown", nullable=False),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        Column("reported_at", TIMESTAMP(timezone=True), nullable=True),
    )
    Index("idx_task_usage_run_id", task_usage.c.run_id)
    Index("idx_task_usage_task_id", task_usage.c.task_id)
    Index("idx_task_usage_reported_at", task_usage.c.reported_at)

    return metadata
