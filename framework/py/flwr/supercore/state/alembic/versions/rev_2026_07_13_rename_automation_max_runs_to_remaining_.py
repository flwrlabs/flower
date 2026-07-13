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
"""Rename automation max_runs to remaining_runs.

Revision ID: f7fffd269759
Revises: cf405f09bb06
Create Date: 2026-07-13 18:36:25.515301
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "f7fffd269759"
down_revision: str | Sequence[str] | None = "cf405f09bb06"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _automation_column_names() -> set[str]:
    """Return the column names currently present on the automation table."""
    return {
        column["name"] for column in sa.inspect(op.get_bind()).get_columns("automation")
    }


def upgrade() -> None:
    """Upgrade schema."""
    columns = _automation_column_names()
    if "remaining_runs" in columns:
        if "max_runs" in columns:
            op.execute(
                """
                UPDATE automation
                SET remaining_runs = COALESCE(remaining_runs, max_runs)
                WHERE remaining_runs IS NULL
                """
            )
            with op.batch_alter_table("automation", schema=None) as batch_op:
                batch_op.drop_column("max_runs")
        return

    if "max_runs" in columns:
        with op.batch_alter_table("automation", schema=None) as batch_op:
            batch_op.alter_column(
                "max_runs",
                new_column_name="remaining_runs",
                existing_type=sa.Integer(),
                existing_nullable=True,
            )


def downgrade() -> None:
    """Downgrade schema."""
    # The current down revision already creates `remaining_runs`, so keep the
    # compatibility migration reversible as a no-op for this branch history.
