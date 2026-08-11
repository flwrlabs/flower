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
"""SQLAlchemy declarative model definitions for LinkState."""

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
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class LinkStateBase(DeclarativeBase):
    """Base class for LinkState models."""

    metadata = MetaData()


class Node(LinkStateBase):
    """Represent a SuperNode."""

    __tablename__ = "node"
    __table_args__ = (
        Index("idx_node_owner_aid", "owner_aid"),
        Index("idx_node_status", "status"),
        Index("idx_online_until", "online_until"),
    )

    node_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, nullable=True)
    owner_aid: Mapped[str | None] = mapped_column(String, nullable=True)
    owner_name: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str | None] = mapped_column(String, nullable=True)
    registered_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_activated_at: Mapped[str | None] = mapped_column(String, nullable=True)
    last_deactivated_at: Mapped[str | None] = mapped_column(String, nullable=True)
    unregistered_at: Mapped[str | None] = mapped_column(String, nullable=True)
    online_until: Mapped[float | None] = mapped_column(Float, nullable=True)
    heartbeat_interval: Mapped[float | None] = mapped_column(Float, nullable=True)
    public_key: Mapped[bytes | None] = mapped_column(
        LargeBinary, unique=True, nullable=True
    )

    __mapper_args__ = {"primary_key": [node_id]}


class Run(LinkStateBase):
    """Represent a run."""

    __tablename__ = "run"
    __table_args__ = (Index("idx_run_series_id", "series_id"),)

    run_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, nullable=True)
    fab_id: Mapped[str | None] = mapped_column(String, nullable=True)
    fab_version: Mapped[str | None] = mapped_column(String, nullable=True)
    fab_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    override_config: Mapped[str | None] = mapped_column(String, nullable=True)
    usage_reported_at: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("''")
    )
    federation_id: Mapped[str | None] = mapped_column(String, nullable=True)
    primary_task_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    federation_config: Mapped[str | None] = mapped_column(String, nullable=True)
    series_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    flwr_aid: Mapped[str | None] = mapped_column(String, nullable=True)
    bytes_sent: Mapped[int | None] = mapped_column(BigInteger, server_default="0")
    bytes_recv: Mapped[int | None] = mapped_column(BigInteger, server_default="0")
    clientapp_runtime: Mapped[float | None] = mapped_column(Float, server_default="0.0")

    __mapper_args__ = {"primary_key": [run_id]}


# Keep logs as a Core table: it has no non-null primary key. Its nullable
# (timestamp, run_id, node_id) unique constraint cannot provide a stable ORM
# identity because SQL NULL values may occur in multiple rows. Map this table
# only after the schema gains a reliable identity column or constraint.
LogsTable = Table(
    "logs",
    LinkStateBase.metadata,
    Column("timestamp", Float),
    Column("run_id", BigInteger, ForeignKey("run.run_id")),
    Column("node_id", BigInteger),
    Column("log", String),
    UniqueConstraint("timestamp", "run_id", "node_id"),
)


class Context(LinkStateBase):
    """Represent a run context."""

    __tablename__ = "context"

    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("run.run_id"), unique=True, nullable=True
    )
    context: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    __mapper_args__ = {"primary_key": [run_id]}


class MessageIns(LinkStateBase):
    """Represent an instruction message."""

    __tablename__ = "message_ins"

    message_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    group_id: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("run.run_id"), nullable=True
    )
    src_node_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    dst_node_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reply_to_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    delivered_at: Mapped[str | None] = mapped_column(String, nullable=True)
    ttl: Mapped[float | None] = mapped_column(Float, nullable=True)
    message_type: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    error: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    __mapper_args__ = {"primary_key": [message_id]}


class MessageRes(LinkStateBase):
    """Represent a reply message."""

    __tablename__ = "message_res"
    __table_args__ = (
        Index(
            "idx_message_res_reply_to_message_id_unique",
            "reply_to_message_id",
            unique=True,
        ),
    )

    message_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    group_id: Mapped[str | None] = mapped_column(String, nullable=True)
    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("run.run_id"), nullable=True
    )
    src_node_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    dst_node_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    reply_to_message_id: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    delivered_at: Mapped[str | None] = mapped_column(String, nullable=True)
    ttl: Mapped[float | None] = mapped_column(Float, nullable=True)
    message_type: Mapped[str | None] = mapped_column(String, nullable=True)
    content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    error: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    __mapper_args__ = {"primary_key": [message_id]}
