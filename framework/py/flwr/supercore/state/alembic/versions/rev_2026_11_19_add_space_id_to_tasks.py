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
"""Add Space ID to tasks.

Revision ID: d5b6e8a9a3bc
Revises: 78f7c7833f29
Create Date: 2026-11-19 20:15:13.160112
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "d5b6e8a9a3bc"
down_revision: str | Sequence[str] | None = "78f7c7833f29"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add task Space IDs and backfill them from existing runs."""
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.add_column(sa.Column("space_id", sa.Uuid(), nullable=True))

    op.execute(
        "UPDATE task SET space_id = "
        "(SELECT run.space_id FROM run WHERE run.run_id = task.run_id)"
    )


def downgrade() -> None:
    """Remove task Space IDs."""
    op.drop_column("task", "space_id")
