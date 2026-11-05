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
"""Rename conversation storage to space.

Revision ID: a8f6aa405dfa
Revises: 10abac2005b4
Create Date: 2026-11-04 21:24:52.061177
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "a8f6aa405dfa"
down_revision: str | Sequence[str] | None = "10abac2005b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table("conversation", "space")
    _rename_children("conversation", "space")


def downgrade() -> None:
    """Downgrade schema."""
    _rename_children("space", "conversation")
    op.rename_table("space", "conversation")


def _rename_children(source: str, target: str) -> None:
    """Rename child tables, columns, and indexes without rebuilding tables."""
    for suffix, old_columns, new_columns in (
        ("participant", ["flwr_aid"], ["flwr_aid"]),
        ("message", [f"{source}_id", "id"], [f"{target}_id", "id"]),
    ):
        old_table = f"{source}_{suffix}"
        new_table = f"{target}_{suffix}"
        op.rename_table(old_table, new_table)
        op.alter_column(
            new_table,
            f"{source}_id",
            new_column_name=f"{target}_id",
            existing_type=sa.Uuid(),
            existing_nullable=False,
        )
        op.drop_index(f"idx_{old_table}_{'_'.join(old_columns)}", table_name=new_table)
        op.create_index(
            f"idx_{new_table}_{'_'.join(new_columns)}",
            new_table,
            new_columns,
            unique=False,
        )
