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
"""Rename federation columns to federation ID.

Revision ID: f3bd92e61ee6
Revises: 31da64063282
Create Date: 2026-07-02 20:57:40.001255
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "f3bd92e61ee6"
down_revision: str | Sequence[str] | None = "31da64063282"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.alter_column(
            "federation",
            new_column_name="federation_id",
            existing_type=sa.String(),
            existing_nullable=True,
        )

    with op.batch_alter_table("run_series", schema=None) as batch_op:
        batch_op.alter_column(
            "federation",
            new_column_name="federation_id",
            existing_type=sa.String(),
            existing_nullable=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("run_series", schema=None) as batch_op:
        batch_op.alter_column(
            "federation_id",
            new_column_name="federation",
            existing_type=sa.String(),
            existing_nullable=False,
        )

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.alter_column(
            "federation_id",
            new_column_name="federation",
            existing_type=sa.String(),
            existing_nullable=True,
        )
