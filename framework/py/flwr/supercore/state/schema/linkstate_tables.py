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
    BigInteger,
    Column,
    Float,
    ForeignKey,
    Index,
    LargeBinary,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    text,
)

from flwr.supercore.constant import RunType


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
        Column("federation", String),
        Column("primary_task_id", BigInteger, nullable=False),
        Column("federation_config", String),
        Column("run_type", String, nullable=False, server_default=RunType.SERVER_APP),
        Column("flwr_aid", String),
        Column("bytes_sent", BigInteger, server_default="0"),
        Column("bytes_recv", BigInteger, server_default="0"),
        Column("clientapp_runtime", Float, server_default="0.0"),
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
    )

    # --------------------------------------------------------------------------
    #  Table: run_collection
    # --------------------------------------------------------------------------
    run_collection = Table(
        "run_collection",
        metadata,
        Column("id", BigInteger, primary_key=True, nullable=False),
        Column("collection_id", String, nullable=False),
        Column("flwr_aid", String, nullable=False),
        Column("title", String, nullable=True),
        Column("metadata_json", String, nullable=False),
        Column("created_at", Float, nullable=False),
        Column("updated_at", Float, nullable=False),
        Column("last_run_id", BigInteger, nullable=True),
        UniqueConstraint(
            "flwr_aid",
            "collection_id",
            name="uq_run_collection_flwr_aid_collection_id",
        ),
    )
    Index(
        "idx_run_collection_flwr_aid_created_at",
        run_collection.c.flwr_aid,
        run_collection.c.created_at,
    )

    # --------------------------------------------------------------------------
    #  Table: run_collection_item
    # --------------------------------------------------------------------------
    run_collection_item = Table(
        "run_collection_item",
        metadata,
        Column("id", BigInteger, primary_key=True, nullable=False),
        Column("collection_id", String, nullable=False),
        Column("flwr_aid", String, nullable=False),
        Column("item_index", BigInteger, nullable=False),
        Column("item_type", String, nullable=False),
        Column("item_json", String, nullable=False),
        Column("created_at", Float, nullable=False),
        Column("run_id", BigInteger, nullable=True),
        Column("task_id", BigInteger, nullable=True),
        Column("item_ref", String, nullable=True),
        Column("parent_item_ref", String, nullable=True),
        UniqueConstraint(
            "flwr_aid",
            "collection_id",
            "item_index",
            name="uq_run_collection_item_flwr_aid_collection_id_item_index",
        ),
    )
    Index(
        "idx_run_collection_item_flwr_aid_collection_id_item_index",
        run_collection_item.c.flwr_aid,
        run_collection_item.c.collection_id,
        run_collection_item.c.item_index,
    )
    Index(
        "idx_run_collection_item_flwr_aid_item_ref",
        run_collection_item.c.flwr_aid,
        run_collection_item.c.item_ref,
    )

    return metadata
