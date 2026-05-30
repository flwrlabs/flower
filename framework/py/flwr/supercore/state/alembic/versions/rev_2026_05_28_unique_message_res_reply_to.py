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
"""Ensure each instruction has at most one reply.

Revision ID: b2f7c9e5a4d1
Revises: f934b49300f8
Create Date: 2026-05-28 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "b2f7c9e5a4d1"
down_revision: str | Sequence[str] | None = "f934b49300f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    duplicate_reply_rows = op.get_bind().execute(
        sa.text(
            """
            SELECT reply_to_message_id, COUNT(*) AS reply_count
            FROM message_res
            WHERE reply_to_message_id IS NOT NULL
            AND reply_to_message_id != ''
            GROUP BY reply_to_message_id
            HAVING COUNT(*) > 1
            LIMIT 5
            """
        )
    )
    duplicate_reply_ids = [row.reply_to_message_id for row in duplicate_reply_rows]
    if duplicate_reply_ids:
        raise RuntimeError(
            "Cannot add unique index on message_res.reply_to_message_id because "
            "duplicate replies already exist for reply_to_message_id values: "
            f"{', '.join(duplicate_reply_ids)}. Remove duplicate message_res rows "
            "before rerunning the migration."
        )

    op.create_index(
        "idx_message_res_reply_to_message_id_unique",
        "message_res",
        ["reply_to_message_id"],
        unique=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "idx_message_res_reply_to_message_id_unique",
        table_name="message_res",
    )
