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
"""Remove legacy SpaceMessage storage.

Revision ID: 28579cfada87
Revises: d17575bb3fde
Create Date: 2026-11-27 09:15:06.323010
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "28579cfada87"
down_revision: str | Sequence[str] | None = "d17575bb3fde"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove legacy message storage and its read cursor."""
    op.drop_table("space_message")
    with op.batch_alter_table("space_participant", schema=None) as batch_op:
        batch_op.drop_column("read_through_message_id")


def downgrade() -> None:
    """Recreate empty legacy storage without restoring deleted messages."""
    with op.batch_alter_table("space_participant", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "read_through_message_id",
                sa.BIGINT(),
                server_default=sa.text("'0'"),
                nullable=False,
            )
        )

    op.create_table(
        "space_message",
        sa.Column("id", sa.INTEGER(), nullable=False),
        sa.Column("space_id", sa.CHAR(length=32), nullable=False),
        sa.Column("flwr_aid", sa.VARCHAR(), nullable=True),
        sa.Column("agent_id", sa.BIGINT(), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(), nullable=False),
        sa.Column("content", sa.TEXT(), nullable=False),
        sa.Column(
            "hidden", sa.BOOLEAN(), server_default=sa.text("(false)"), nullable=False
        ),
        sa.CheckConstraint(
            "(flwr_aid IS NOT NULL AND agent_id IS NULL) OR "
            "(flwr_aid IS NULL AND agent_id IS NOT NULL)",
            name=op.f("ck_space_message_identity"),
        ),
        sa.ForeignKeyConstraint(["agent_id"], ["agent.agent_id"]),
        sa.ForeignKeyConstraint(["space_id"], ["space.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("space_message", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("idx_space_message_space_id_id"),
            ["space_id", "id"],
            unique=False,
        )
