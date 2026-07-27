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
"""Make automation next_run_at non-nullable.

Revision ID: f7fffd269759
Revises: e2dd7937d7fc
Create Date: 2026-07-13 18:36:25.515301
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "f7fffd269759"
down_revision: str | Sequence[str] | None = "e2dd7937d7fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    columns = {
        column["name"]: column
        for column in sa.inspect(op.get_bind()).get_columns("automation")
    }
    if "next_run_at" not in columns:
        return

    # Backfill deployed rows before enforcing the non-null contract.
    op.execute(
        """
        UPDATE automation
        SET next_run_at = updated_at
        WHERE next_run_at IS NULL
        """
    )
    if columns["next_run_at"]["nullable"]:
        with op.batch_alter_table("automation", schema=None) as batch_op:
            batch_op.alter_column(
                "next_run_at",
                existing_type=sa.TIMESTAMP(timezone=True),
                existing_nullable=True,
                nullable=False,
            )


def downgrade() -> None:
    """Downgrade schema."""
    columns = {
        column["name"]: column
        for column in sa.inspect(op.get_bind()).get_columns("automation")
    }
    if "next_run_at" in columns and not columns["next_run_at"]["nullable"]:
        with op.batch_alter_table("automation", schema=None) as batch_op:
            batch_op.alter_column(
                "next_run_at",
                existing_type=sa.TIMESTAMP(timezone=True),
                existing_nullable=False,
                nullable=True,
            )
