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
"""Use timestamp columns for task timestamps.

Revision ID: e937d3528d23
Revises: aac61834ee69
Create Date: 2026-05-14 15:15:35.278372
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "e937d3528d23"
down_revision: str | Sequence[str] | None = "aac61834ee69"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMP_COLUMNS = ("pending_at", "starting_at", "running_at", "finished_at")


def _dialect_name() -> str:
    """Return the active database dialect name."""
    return op.get_bind().dialect.name


def _is_sqlite() -> bool:
    """Return True if the migration is running against SQLite."""
    return _dialect_name() == "sqlite"


def _is_postgresql() -> bool:
    """Return True if the migration is running against PostgreSQL."""
    return _dialect_name() == "postgresql"


def _normalize_sqlite_task_timestamps() -> None:
    """Convert existing SQLite task timestamp values before altering columns."""
    op.add_column("task", sa.Column("_tmp_active_until", sa.String(), nullable=True))
    op.execute(
        """
        UPDATE task
        SET _tmp_active_until = datetime(active_until, 'unixepoch') || '+00:00'
        WHERE active_until IS NOT NULL
        """
    )
    for column in TIMESTAMP_COLUMNS:
        op.add_column("task", sa.Column(f"_tmp_{column}", sa.String(), nullable=True))
        # Existing values were written via datetime.isoformat(), which uses "T" as
        # the date/time separator. SQLAlchemy's SQLite TIMESTAMP storage writes
        # datetimes with a space separator, so normalize old rows before the type
        # change.
        op.execute(
            f"""
            UPDATE task
            SET _tmp_{column} = replace({column}, 'T', ' ')
            WHERE {column} IS NOT NULL
            """
        )


def _denormalize_sqlite_task_timestamps() -> None:
    """Convert existing SQLite task timestamp values before downgrading columns."""
    op.add_column(
        "task", sa.Column("_tmp_active_until", sa.BigInteger(), nullable=True)
    )
    op.execute(
        """
        UPDATE task
        SET _tmp_active_until = CAST(strftime('%s', active_until) AS BIGINT)
        WHERE active_until IS NOT NULL
        """
    )
    for column in TIMESTAMP_COLUMNS:
        op.add_column("task", sa.Column(f"_tmp_{column}", sa.String(), nullable=True))
        # The previous task-table schema stored these timestamps as strings written
        # with datetime.isoformat(), so restore the "T" separator when downgrading
        # from SQLite TIMESTAMP storage. The temporary column avoids Alembic's
        # batch-copy CAST from TIMESTAMP back to VARCHAR.
        op.execute(
            f"""
            UPDATE task
            SET _tmp_{column} = replace({column}, ' ', 'T')
            WHERE {column} IS NOT NULL
            """
        )


def _restore_sqlite_task_timestamps() -> None:
    """Restore preserved task timestamp values after altering columns."""
    op.execute(
        """
        UPDATE task
        SET active_until = _tmp_active_until
        WHERE _tmp_active_until IS NOT NULL
        """
    )
    for column in TIMESTAMP_COLUMNS:
        op.execute(
            f"""
            UPDATE task
            SET {column} = _tmp_{column}
            WHERE _tmp_{column} IS NOT NULL
            """
        )


def _drop_sqlite_timestamp_temp_columns() -> None:
    """Drop temporary columns used to preserve SQLite timestamp values."""
    with op.batch_alter_table("task", schema=None) as batch_op:
        for column in TIMESTAMP_COLUMNS:
            batch_op.drop_column(f"_tmp_{column}")
        batch_op.drop_column("_tmp_active_until")


def _upgrade_postgresql_task_timestamps() -> None:
    """Convert task timestamp columns on PostgreSQL."""
    op.execute(
        """
        ALTER TABLE task
        ALTER COLUMN active_until TYPE TIMESTAMP WITH TIME ZONE
            USING to_timestamp(active_until),
        ALTER COLUMN pending_at TYPE TIMESTAMP WITH TIME ZONE
            USING pending_at::timestamp with time zone,
        ALTER COLUMN starting_at TYPE TIMESTAMP WITH TIME ZONE
            USING starting_at::timestamp with time zone,
        ALTER COLUMN running_at TYPE TIMESTAMP WITH TIME ZONE
            USING running_at::timestamp with time zone,
        ALTER COLUMN finished_at TYPE TIMESTAMP WITH TIME ZONE
            USING finished_at::timestamp with time zone
        """
    )


def _downgrade_postgresql_task_timestamps() -> None:
    """Convert task timestamp columns back on PostgreSQL."""
    op.execute(
        """
        ALTER TABLE task
        ALTER COLUMN finished_at TYPE VARCHAR USING finished_at::text,
        ALTER COLUMN running_at TYPE VARCHAR USING running_at::text,
        ALTER COLUMN starting_at TYPE VARCHAR USING starting_at::text,
        ALTER COLUMN pending_at TYPE VARCHAR USING pending_at::text,
        ALTER COLUMN active_until TYPE BIGINT
            USING EXTRACT(EPOCH FROM active_until)::BIGINT
        """
    )


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    if _is_sqlite():
        _normalize_sqlite_task_timestamps()
    elif _is_postgresql():
        _upgrade_postgresql_task_timestamps()
        return

    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.alter_column(
            "active_until",
            existing_type=sa.BIGINT(),
            type_=sa.TIMESTAMP(timezone=True),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "pending_at",
            existing_type=sa.VARCHAR(),
            type_=sa.TIMESTAMP(timezone=True),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "starting_at",
            existing_type=sa.VARCHAR(),
            type_=sa.TIMESTAMP(timezone=True),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "running_at",
            existing_type=sa.VARCHAR(),
            type_=sa.TIMESTAMP(timezone=True),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "finished_at",
            existing_type=sa.VARCHAR(),
            type_=sa.TIMESTAMP(timezone=True),
            existing_nullable=True,
        )

    if _is_sqlite():
        _restore_sqlite_task_timestamps()
        _drop_sqlite_timestamp_temp_columns()

    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    if _is_sqlite():
        _denormalize_sqlite_task_timestamps()
    elif _is_postgresql():
        _downgrade_postgresql_task_timestamps()
        return

    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.alter_column(
            "finished_at",
            existing_type=sa.TIMESTAMP(timezone=True),
            type_=sa.VARCHAR(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "running_at",
            existing_type=sa.TIMESTAMP(timezone=True),
            type_=sa.VARCHAR(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "starting_at",
            existing_type=sa.TIMESTAMP(timezone=True),
            type_=sa.VARCHAR(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "pending_at",
            existing_type=sa.TIMESTAMP(timezone=True),
            type_=sa.VARCHAR(),
            existing_nullable=False,
        )
        batch_op.alter_column(
            "active_until",
            existing_type=sa.TIMESTAMP(timezone=True),
            type_=sa.BIGINT(),
            existing_nullable=True,
        )

    if _is_sqlite():
        _restore_sqlite_task_timestamps()
        _drop_sqlite_timestamp_temp_columns()

    # ### end Alembic commands ###
