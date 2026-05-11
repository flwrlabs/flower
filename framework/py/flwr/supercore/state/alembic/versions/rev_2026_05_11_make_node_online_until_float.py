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
"""Make node online_until a float.

Revision ID: 6f7d2e8a9b10
Revises: 11d1bdac4c8e
Create Date: 2026-05-11 00:00:00.000000
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "6f7d2e8a9b10"
down_revision: str | Sequence[str] | None = "11d1bdac4c8e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _is_postgresql() -> bool:
    """Return True if the migration is running against PostgreSQL."""
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    """Upgrade schema."""
    # `online_until` is stored as epoch seconds. SQLite already stores numeric values
    # in practice, while PostgreSQL needs an explicit conversion from TIMESTAMP.
    if _is_postgresql():
        op.execute(
            """
            ALTER TABLE node
            ALTER COLUMN online_until TYPE DOUBLE PRECISION
            USING EXTRACT(EPOCH FROM online_until AT TIME ZONE 'UTC')
            """
        )
        return

    with op.batch_alter_table("node", schema=None) as batch_op:
        batch_op.alter_column(
            "online_until",
            existing_type=sa.TIMESTAMP(),
            type_=sa.Float(),
            existing_nullable=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    if _is_postgresql():
        op.execute(
            """
            ALTER TABLE node
            ALTER COLUMN online_until TYPE TIMESTAMP
            USING (to_timestamp(online_until) AT TIME ZONE 'UTC')
            """
        )
        return

    with op.batch_alter_table("node", schema=None) as batch_op:
        batch_op.alter_column(
            "online_until",
            existing_type=sa.Float(),
            type_=sa.TIMESTAMP(),
            existing_nullable=True,
        )
