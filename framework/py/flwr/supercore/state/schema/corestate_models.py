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
    TIMESTAMP,
    BigInteger,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


class RunSeries(FlwrBase):
    """Represent a run series."""

    __tablename__ = "run_series"

    series_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    federation_id: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )


class SeriesContext(FlwrBase):
    """Represent context data for a run series."""

    __tablename__ = "series_context"

    series_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    context: Mapped[bytes | None] = mapped_column(LargeBinary)


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
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    next_run_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    fixed_interval: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    remaining_runs: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stopped_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )


class Connector(FlwrBase):
    """Represent connector configuration for an account."""

    __tablename__ = "connector"

    flwr_aid: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    connector_ref: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    credentials_json: Mapped[str] = mapped_column(String, nullable=False)
    config_json: Mapped[str] = mapped_column(String, nullable=False)


class ConnectorOAuthSession(FlwrBase):
    """Represent an OAuth authorization session for a connector."""

    __tablename__ = "connector_oauth_session"

    oauth_session_id: Mapped[str] = mapped_column(
        String, primary_key=True, nullable=False
    )
    flwr_aid: Mapped[str] = mapped_column(String, nullable=False)
    connector_ref: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    redirect_uri: Mapped[str] = mapped_column(String, nullable=False)
    pkce_verifier: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )


class RunConnector(FlwrBase):
    """Represent the connectors attached to a run."""

    __tablename__ = "run_connector"

    run_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)
    connector_ref: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)


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
    token: Mapped[str | None] = mapped_column(String, nullable=True)
    active_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    pending_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    starting_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    running_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    sub_status: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("''")
    )
    details: Mapped[str] = mapped_column(
        String, nullable=False, server_default=text("''")
    )

    __mapper_args__ = {"primary_key": [task_id]}
