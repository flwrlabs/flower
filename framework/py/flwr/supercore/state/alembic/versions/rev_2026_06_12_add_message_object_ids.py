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
"""Add ObjectStore root IDs for message rows.

Revision ID: a1d4e7b9c2f0
Revises: b2f7c9e5a4d1
Create Date: 2026-06-12 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1d4e7b9c2f0"
down_revision: str | Sequence[str] | None = "b2f7c9e5a4d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade database schema."""
    with op.batch_alter_table("message_ins", schema=None) as batch_op:
        batch_op.add_column(sa.Column("object_id", sa.String(), nullable=True))
    with op.batch_alter_table("message_res", schema=None) as batch_op:
        batch_op.add_column(sa.Column("object_id", sa.String(), nullable=True))

    op.execute("UPDATE message_ins SET object_id = message_id WHERE object_id IS NULL")
    op.execute("UPDATE message_res SET object_id = message_id WHERE object_id IS NULL")


def downgrade() -> None:
    """Downgrade database schema."""
    with op.batch_alter_table("message_res", schema=None) as batch_op:
        batch_op.drop_column("object_id")
    with op.batch_alter_table("message_ins", schema=None) as batch_op:
        batch_op.drop_column("object_id")
