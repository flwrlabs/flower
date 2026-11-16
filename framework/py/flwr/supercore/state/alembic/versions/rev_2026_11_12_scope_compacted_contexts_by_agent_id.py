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
"""Replace series-keyed compacted contexts with agent-keyed contexts.

Revision ID: 0149c42eef1d
Revises: 6cbf924cf5ec
Create Date: 2026-11-12 10:33:24.669713
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "0149c42eef1d"
down_revision: str | Sequence[str] | None = "6cbf924cf5ec"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Existing series-keyed rows cannot be assigned to an agent.
    op.drop_table("compacted_contexts")
    op.create_table(
        "compacted_contexts",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("agent_id", sa.BigInteger(), nullable=False),
        sa.Column("through_event_id", sa.BigInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["agent_id"], ["agent.agent_id"], name="fk_compacted_contexts_agent_id"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_id", "through_event_id", name="uq_compacted_contexts_agent_event"
        ),
    )
    op.create_index(
        "idx_compacted_contexts_agent_id_id", "compacted_contexts", ["agent_id", "id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("compacted_contexts")
    op.create_table(
        "compacted_contexts",
        sa.Column(
            "id",
            sa.BigInteger().with_variant(sa.Integer(), "sqlite"),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column("series_id", sa.BigInteger(), nullable=False),
        sa.Column("through_event_id", sa.BigInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_count", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("series_id", "through_event_id"),
    )
    op.create_index(
        "idx_compacted_contexts_series_id_id",
        "compacted_contexts",
        ["series_id", "id"],
    )
