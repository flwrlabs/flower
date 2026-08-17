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
"""Remove legacy logs and context tables.

Revision ID: a6f4d2c91b7e
Revises: 63be1836ddca
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a6f4d2c91b7e"
down_revision: str | Sequence[str] | None = "63be1836ddca"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Remove the superseded run log and run context tables."""
    op.drop_table("context")
    op.drop_table("logs")


def downgrade() -> None:
    """Recreate the legacy tables without attempting to restore deleted data."""
    op.create_table(
        "logs",
        sa.Column("timestamp", sa.Float(), nullable=True),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("node_id", sa.Integer(), nullable=True),
        sa.Column("log", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["run.run_id"]),
        sa.UniqueConstraint("timestamp", "run_id", "node_id"),
    )
    op.create_table(
        "context",
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("context", sa.LargeBinary(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["run.run_id"]),
        sa.UniqueConstraint("run_id"),
    )
