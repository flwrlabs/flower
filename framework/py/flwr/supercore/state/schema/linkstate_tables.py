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
"""SQLAlchemy Core Table definitions for LinkState."""


from sqlalchemy import (
    TIMESTAMP,
    BigInteger,
    Column,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    text,
)


def create_linkstate_metadata() -> MetaData:
    """Create and return MetaData with LinkState table definitions."""
    metadata = MetaData()

    # --------------------------------------------------------------------------
    #  Table: node
    # --------------------------------------------------------------------------
    Table(
        "node",
        metadata,
        Column("node_id", BigInteger, unique=True),
        Column("owner_aid", String),
        Column("owner_name", String),
        Column("status", String),
        Column("registered_at", String),
        Column("last_activated_at", String, nullable=True),
        Column("last_deactivated_at", String, nullable=True),
        Column("unregistered_at", String, nullable=True),
        Column("online_until", Float, nullable=True),
        Column("heartbeat_interval", Float),
        Column("public_key", LargeBinary, unique=True),
        # Indexes
        # Used in delete_node and get_node_info (security/filtering)
        Index("idx_node_owner_aid", "owner_aid"),
        # Used in get_nodes and activation checks (frequent filtering)
        Index("idx_node_status", "status"),
        # Used in heartbeat checks to efficiently find expired nodes
        Index("idx_online_until", "online_until"),
    )

    # --------------------------------------------------------------------------
    #  Table: run
    # --------------------------------------------------------------------------
    Table(
        "run",
        metadata,
        Column("run_id", BigInteger, unique=True),
        Column("fab_id", String),
        Column("fab_version", String),
        Column("fab_hash", String),
        Column("override_config", String),
        Column("usage_reported_at", String, nullable=False, server_default=text("''")),
        Column("federation_id", String),
        Column("primary_task_id", BigInteger, nullable=False),
        Column("federation_config", String),
        Column("series_id", BigInteger, nullable=True),
        Column("flwr_aid", String),
        Column("bytes_sent", BigInteger, server_default="0"),
        Column("bytes_recv", BigInteger, server_default="0"),
        Column("clientapp_runtime", Float, server_default="0.0"),
        Index("idx_run_series_id", "series_id"),
    )

    # --------------------------------------------------------------------------
    #  Table: logs
    # --------------------------------------------------------------------------
    Table(
        "logs",
        metadata,
        Column("timestamp", Float),
        Column("run_id", BigInteger, ForeignKey("run.run_id")),
        Column("node_id", BigInteger),
        Column("log", String),
        # Composite PK
        UniqueConstraint("timestamp", "run_id", "node_id"),
    )

    # --------------------------------------------------------------------------
    #  Table: context
    # --------------------------------------------------------------------------
    Table(
        "context",
        metadata,
        Column("run_id", BigInteger, ForeignKey("run.run_id"), unique=True),
        Column("context", LargeBinary),
    )

    # --------------------------------------------------------------------------
    #  Table: connector
    # --------------------------------------------------------------------------
    Table(
        "connector",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("flwr_aid", String, nullable=False),
        Column("connector_ref", String, nullable=False),
        Column("credentials_json", String, nullable=False),
        Column("config_json", String, nullable=False),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        Column("updated_at", TIMESTAMP(timezone=True), nullable=False),
        Column("last_used_at", TIMESTAMP(timezone=True), nullable=True),
        UniqueConstraint(
            "flwr_aid",
            "connector_ref",
            name="uq_connector_flwr_aid_connector_ref",
        ),
    )

    # --------------------------------------------------------------------------
    #  Table: connector_oauth_session
    # --------------------------------------------------------------------------
    connector_oauth_session = Table(
        "connector_oauth_session",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("oauth_session_id", String, nullable=False, unique=True),
        Column("flwr_aid", String, nullable=False),
        Column("connector_ref", String, nullable=False),
        Column("state", String, nullable=False),
        Column("redirect_uri", String, nullable=False),
        Column("pkce_verifier", String, nullable=True),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        Column("expires_at", TIMESTAMP(timezone=True), nullable=False),
        Column("completed_at", TIMESTAMP(timezone=True), nullable=True),
        Column("status", String, nullable=False),
    )
    Index(
        "idx_connector_oauth_session_flwr_aid_status_expires_at",
        connector_oauth_session.c.flwr_aid,
        connector_oauth_session.c.status,
        connector_oauth_session.c.expires_at,
    )

    # --------------------------------------------------------------------------
    #  Table: run_connector
    # --------------------------------------------------------------------------
    Table(
        "run_connector",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("run_id", BigInteger, ForeignKey("run.run_id"), nullable=False),
        Column("flwr_aid", String, nullable=False),
        Column("connector_ref", String, nullable=False),
        Column("created_at", TIMESTAMP(timezone=True), nullable=False),
        ForeignKeyConstraint(
            ["flwr_aid", "connector_ref"],
            ["connector.flwr_aid", "connector.connector_ref"],
        ),
        UniqueConstraint(
            "run_id",
            "flwr_aid",
            "connector_ref",
            name="uq_run_connector_run_id_flwr_aid_connector_ref",
        ),
    )

    # --------------------------------------------------------------------------
    #  Table: message_ins
    # --------------------------------------------------------------------------
    Table(
        "message_ins",
        metadata,
        Column("message_id", String, unique=True),
        Column("group_id", String),
        Column("run_id", BigInteger, ForeignKey("run.run_id")),
        Column("src_node_id", BigInteger),
        Column("dst_node_id", BigInteger),
        Column("reply_to_message_id", String),
        Column("created_at", Float),
        Column("delivered_at", String),
        Column("ttl", Float),
        Column("message_type", String),
        Column("content", LargeBinary, nullable=True),
        Column("error", LargeBinary, nullable=True),
    )

    # --------------------------------------------------------------------------
    #  Table: message_res
    # --------------------------------------------------------------------------
    Table(
        "message_res",
        metadata,
        Column("message_id", String, unique=True),
        Column("group_id", String),
        Column("run_id", BigInteger, ForeignKey("run.run_id")),
        Column("src_node_id", BigInteger),
        Column("dst_node_id", BigInteger),
        Column("reply_to_message_id", String),
        Column("created_at", Float),
        Column("delivered_at", String),
        Column("ttl", Float),
        Column("message_type", String),
        Column("content", LargeBinary, nullable=True),
        Column("error", LargeBinary, nullable=True),
        Index(
            "idx_message_res_reply_to_message_id_unique",
            "reply_to_message_id",
            unique=True,
        ),
    )

    return metadata
