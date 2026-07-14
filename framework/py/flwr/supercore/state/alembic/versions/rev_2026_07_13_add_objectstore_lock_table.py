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
"""Add ObjectStore lock table.

Revision ID: c17f3d9b8a42
Revises: e2dd7937d7fc
Create Date: 2026-07-13 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "c17f3d9b8a42"
down_revision: str | Sequence[str] | None = "e2dd7937d7fc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "objectstore_locks",
        sa.Column("lock_id", sa.String(), nullable=False),
        sa.Column("lock_value", sa.Integer(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("lock_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("objectstore_locks")
