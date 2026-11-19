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
"""Require agent default Space ID.

Revision ID: 3fb31831d5eb
Revises: 0969435b9ca6
Create Date: 2026-11-18 16:57:14.110463
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "3fb31831d5eb"
down_revision: str | Sequence[str] | None = "0969435b9ca6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Delete agents without a default Space and require one for future agents."""
    missing_space = "SELECT agent_id FROM agent WHERE default_space_id IS NULL"
    for table in (
        "automation",
        "compacted_contexts",
        "connector",
        "connector_oauth_session",
        "run",
        "space_message",
        "space_participant",
    ):
        op.execute(f"DELETE FROM {table} WHERE agent_id IN ({missing_space})")
    op.execute("DELETE FROM agent WHERE default_space_id IS NULL")
    _set_default_space_nullable(False)


def downgrade() -> None:
    """Allow agents without a default Space again."""
    _set_default_space_nullable(True)


def _set_default_space_nullable(nullable: bool) -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        op.alter_column(
            "agent", "default_space_id", existing_type=sa.Uuid(), nullable=nullable
        )
        return

    # SQLite needs an active transaction to defer checks across the table rebuild.
    with bind.begin_nested():
        enforce_fks = bind.scalar(sa.text("PRAGMA foreign_keys"))
        if enforce_fks:
            op.execute("PRAGMA defer_foreign_keys = ON")
        with op.batch_alter_table(
            "agent", table_kwargs={"sqlite_autoincrement": True}
        ) as batch_op:
            batch_op.alter_column(
                "default_space_id", existing_type=sa.Uuid(), nullable=nullable
            )
        if enforce_fks:
            # SQLite's deferred counter misses the rename. Check before clearing it.
            violation = bind.execute(sa.text("PRAGMA foreign_key_check")).first()
            if violation:
                raise RuntimeError(
                    f"Foreign key violation after rebuilding agent: {violation}"
                )
            op.execute("PRAGMA defer_foreign_keys = OFF")
