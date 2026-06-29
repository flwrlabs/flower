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
"""Add fab table.

Revision ID: 33e2f70642b1
Revises: 795243e997d8
Create Date: 2026-04-07 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "33e2f70642b1"
down_revision: str | Sequence[str] | None = "795243e997d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "fab",
        sa.Column("fab_hash", sa.String(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("verifications", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("fab_hash"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("fab")
