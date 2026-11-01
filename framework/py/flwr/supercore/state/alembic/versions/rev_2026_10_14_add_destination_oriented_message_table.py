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
"""Add destination-oriented message table.

Revision ID: 6e5225b231fc
Revises: 70dd3c19bc7a
Create Date: 2026-10-14 15:53:15.931967
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from flwr.supercore.state.schema.types import UInt64, UTCDateTime

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "6e5225b231fc"
down_revision: str | Sequence[str] | None = "70dd3c19bc7a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "message",
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("run_id", UInt64(), nullable=False),
        sa.Column("src_node_id", UInt64(), nullable=False),
        sa.Column("dst_node_id", UInt64(), nullable=False),
        sa.Column("src_task_id", UInt64(), nullable=True),
        sa.Column("dst_task_id", UInt64(), nullable=True),
        sa.Column("reply_to_message_id", sa.String(), nullable=True),
        sa.Column("group_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.Double(), nullable=False),
        sa.Column("ttl", sa.Double(), nullable=False),
        sa.Column("message_type", sa.String(), nullable=False),
        sa.Column("expires_at", UTCDateTime(), nullable=False),
        sa.Column(
            "cleared", sa.Boolean(), server_default=sa.text("(false)"), nullable=False
        ),
        sa.Column("acknowledged_at", UTCDateTime(), nullable=True),
        sa.Column("cancelled_at", UTCDateTime(), nullable=True),
        sa.CheckConstraint(
            "acknowledged_at IS NULL OR cancelled_at IS NULL",
            name="ck_message_terminal_timestamps_exclusive",
        ),
        sa.CheckConstraint(
            "cleared OR (acknowledged_at IS NULL AND cancelled_at IS NULL)",
            name="ck_message_terminal_timestamps_require_cleared",
        ),
        sa.ForeignKeyConstraint(
            ["reply_to_message_id"], ["message.message_id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.create_index(
            "idx_message_pending_dst_created_at",
            ["dst_node_id", "created_at", "message_id"],
            unique=False,
            postgresql_where=sa.column("cleared", sa.Boolean()).is_(False),
            sqlite_where=sa.column("cleared", sa.Boolean()).is_(False),
        )
        batch_op.create_index(
            "idx_message_pending_dst_run_created_at",
            ["dst_node_id", "run_id", "created_at", "message_id"],
            unique=False,
            postgresql_where=sa.column("cleared", sa.Boolean()).is_(False),
            sqlite_where=sa.column("cleared", sa.Boolean()).is_(False),
        )
        batch_op.create_index(
            "idx_message_pending_run_id",
            ["run_id"],
            unique=False,
            postgresql_where=sa.column("cleared", sa.Boolean()).is_(False),
            sqlite_where=sa.column("cleared", sa.Boolean()).is_(False),
        )
        batch_op.create_index(
            "idx_message_reply_to_message_id", ["reply_to_message_id"], unique=False
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.drop_index("idx_message_reply_to_message_id")
        batch_op.drop_index(
            "idx_message_pending_run_id",
            postgresql_where=sa.column("cleared", sa.Boolean()).is_(False),
            sqlite_where=sa.column("cleared", sa.Boolean()).is_(False),
        )
        batch_op.drop_index(
            "idx_message_pending_dst_run_created_at",
            postgresql_where=sa.column("cleared", sa.Boolean()).is_(False),
            sqlite_where=sa.column("cleared", sa.Boolean()).is_(False),
        )
        batch_op.drop_index(
            "idx_message_pending_dst_created_at",
            postgresql_where=sa.column("cleared", sa.Boolean()).is_(False),
            sqlite_where=sa.column("cleared", sa.Boolean()).is_(False),
        )

    # SQLite checks self-referencing FKs during the implicit DELETE in DROP TABLE.
    op.execute(sa.text("UPDATE message SET reply_to_message_id = NULL"))
    op.drop_table("message")
