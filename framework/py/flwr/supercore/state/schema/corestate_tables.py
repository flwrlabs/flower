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
    Column,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
)


def create_corestate_metadata() -> MetaData:
    """Create and return MetaData with CoreState table definitions."""
    metadata = MetaData()

    # --------------------------------------------------------------------------
    #  Table: token_store
    # --------------------------------------------------------------------------
    Table(
        "token_store",
        metadata,
        Column("run_id", Integer, primary_key=True, nullable=True),
        Column("token", String, unique=True, nullable=False),
        Column("active_until", Float),
    )

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
    #  Table: task
    # --------------------------------------------------------------------------
    Table(
        "task",
        metadata,
        Column("task_id", Integer, nullable=False, unique=True),
        Column("type", String, nullable=False),
        Column("run_id", Integer, nullable=False),
        Column("status", String, nullable=False),
        Column("fab_hash", String, nullable=True),
        Column("model_ref", String, nullable=True),
        Column("connector_ref", String, nullable=True),
        Column("token", String, nullable=False),
        Column("pending_at", String),
        Column("starting_at", String),
        Column("running_at", String),
        Column("finished_at", String),
    )

    return metadata
