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
"""Remove legacy message tables.

Revision ID: 52b90538cdde
Revises: 4c9b3f185c5b
Create Date: 2026-10-28 12:15:15.430633
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "52b90538cdde"
down_revision: str | Sequence[str] | None = "4c9b3f185c5b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop old tables after their metadata has been copied to message."""
    op.drop_table("message_res")
    op.drop_table("message_ins")


def downgrade() -> None:
    """Recreate empty legacy tables without restoring deleted rows."""
    op.create_table(
        "message_ins",
        sa.Column("message_id", sa.VARCHAR(), nullable=True),
        sa.Column("group_id", sa.VARCHAR(), nullable=True),
        sa.Column("run_id", sa.BIGINT(), nullable=True),
        sa.Column("src_node_id", sa.BIGINT(), nullable=True),
        sa.Column("dst_node_id", sa.BIGINT(), nullable=True),
        sa.Column("reply_to_message_id", sa.VARCHAR(), nullable=True),
        sa.Column("created_at", sa.FLOAT(), nullable=True),
        sa.Column("delivered_at", sa.VARCHAR(), nullable=True),
        sa.Column("ttl", sa.FLOAT(), nullable=True),
        sa.Column("message_type", sa.VARCHAR(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("error", sa.LargeBinary(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["run.run_id"],
        ),
        sa.UniqueConstraint("message_id"),
    )
    op.create_table(
        "message_res",
        sa.Column("message_id", sa.VARCHAR(), nullable=True),
        sa.Column("group_id", sa.VARCHAR(), nullable=True),
        sa.Column("run_id", sa.BIGINT(), nullable=True),
        sa.Column("src_node_id", sa.BIGINT(), nullable=True),
        sa.Column("dst_node_id", sa.BIGINT(), nullable=True),
        sa.Column("reply_to_message_id", sa.VARCHAR(), nullable=True),
        sa.Column("created_at", sa.FLOAT(), nullable=True),
        sa.Column("delivered_at", sa.VARCHAR(), nullable=True),
        sa.Column("ttl", sa.FLOAT(), nullable=True),
        sa.Column("message_type", sa.VARCHAR(), nullable=True),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("error", sa.LargeBinary(), nullable=True),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["run.run_id"],
        ),
        sa.UniqueConstraint("message_id"),
    )
    with op.batch_alter_table("message_res", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("idx_message_res_reply_to_message_id_unique"),
            ["reply_to_message_id"],
            unique=True,
        )
