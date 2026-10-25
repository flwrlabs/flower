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
"""Normalize automation timestamps.

Revision ID: 7208a0cbdcd7
Revises: 28482626dbdc
Create Date: 2026-08-10 10:05:22.358475
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "7208a0cbdcd7"
down_revision: str | Sequence[str] | None = "28482626dbdc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return

    op.execute(
        sa.text(
            """
            UPDATE automation
            SET
                created_at = replace(created_at, 'T', ' '),
                updated_at = replace(updated_at, 'T', ' '),
                next_run_at = replace(next_run_at, 'T', ' '),
                stopped_at = replace(stopped_at, 'T', ' ')
            WHERE
                created_at LIKE '%T%'
                OR updated_at LIKE '%T%'
                OR next_run_at LIKE '%T%'
                OR stopped_at LIKE '%T%'
            """
        )
    )


def downgrade() -> None:
    """Downgrade schema."""
