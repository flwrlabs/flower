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
"""Squash post-nonce state migrations for upcoming release.

Revision ID: 26b5d6018750
Revises: b277e6f3656c
Create Date: 2026-05-15 14:16:55.405579
"""
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection, RowMapping

from flwr.common.constant import TASK_ID_NUM_BYTES, SubStatus
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate.utils import generate_rand_int_from_bytes
from flwr.supercore.date import now
from flwr.supercore.utils import uint64_to_int64

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "26b5d6018750"
down_revision: str | Sequence[str] | None = "b277e6f3656c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RUN_ID_FOREIGN_KEYS = (
    ("context", "context_run_id_fkey"),
    ("logs", "logs_run_id_fkey"),
    ("message_ins", "message_ins_run_id_fkey"),
    ("message_res", "message_res_run_id_fkey"),
)

# Historical run_type string values used by this migration.
# Keep these local constants because RunType has been removed from the framework.
SERVER_APP_RUN_TYPE = "serverapp"
SIMULATION_RUN_TYPE = "simulation"


def _is_postgresql() -> bool:
    """Return True if the migration is running against PostgreSQL."""
    return op.get_bind().dialect.name == "postgresql"


def _drop_run_id_foreign_keys() -> None:
    """Drop run_id foreign keys before changing column types on PostgreSQL."""
    if not _is_postgresql():
        return
    for table_name, constraint_name in _RUN_ID_FOREIGN_KEYS:
        op.drop_constraint(constraint_name, table_name, type_="foreignkey")


def _create_run_id_foreign_keys() -> None:
    """Recreate run_id foreign keys after changing column types on PostgreSQL."""
    if not _is_postgresql():
        return
    for table_name, constraint_name in _RUN_ID_FOREIGN_KEYS:
        op.create_foreign_key(
            constraint_name,
            table_name,
            "run",
            ["run_id"],
            ["run_id"],
        )


def _primary_task_type_from_run_type(run_type: str) -> str:
    """Return the primary task type for the given run type."""
    if run_type == SIMULATION_RUN_TYPE:
        return TaskType.SIMULATION
    if run_type == SERVER_APP_RUN_TYPE:
        return TaskType.SERVER_APP
    raise RuntimeError(
        f"Unsupported run_type while backfilling primary tasks: {run_type}"
    )


def _timestamp_from_run_value(value: Any) -> datetime | None:
    """Return a datetime for an old run timestamp value."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise RuntimeError(f"Unsupported timestamp value while backfilling task: {value}")


def _timestamp_to_run_value(value: Any) -> str | None:
    """Return an old run timestamp string for a task timestamp value."""
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value.replace(" ", "T", 1)
    raise RuntimeError(f"Unsupported timestamp value while restoring run: {value}")


def _generate_unique_task_id(bind: Connection, reserved_task_ids: set[int]) -> int:
    """Generate a task ID that does not collide with persisted or reserved tasks."""
    while True:
        task_id = generate_rand_int_from_bytes(
            TASK_ID_NUM_BYTES, exclude=reserved_task_ids
        )
        row = bind.execute(
            sa.text("SELECT 1 FROM task WHERE task_id = :task_id LIMIT 1"),
            {"task_id": uint64_to_int64(task_id)},
        ).first()
        if row is None:
            reserved_task_ids.add(task_id)
            return task_id


def _load_runs_for_primary_task_backfill(bind: Connection) -> list[RowMapping]:
    """Load historical runs that need a primary task backfill."""
    query = """
        SELECT run_id, fab_hash, pending_at, starting_at, running_at, finished_at,
            sub_status, details, run_type
        FROM run
    """
    return list(bind.execute(sa.text(query)).mappings().all())


def _validate_primary_task_backfill_runs(runs: Sequence[RowMapping]) -> None:
    """Validate that historical runs can be backfilled before mutating schema."""
    for run in runs:
        if not run["pending_at"]:
            raise RuntimeError(
                "Cannot backfill primary task for run "
                f"{run['run_id']} without pending_at."
            )
        _primary_task_type_from_run_type(run["run_type"])


def _is_in_flight_run(run: RowMapping) -> bool:
    """Return True if the historical run was STARTING or RUNNING."""
    return not run["finished_at"] and (run["starting_at"] or run["running_at"])


def _backfilled_primary_task_status(
    run: RowMapping, stopped_at: datetime
) -> tuple[datetime | None, str, str]:
    """Return the backfilled finished_at, sub_status, and details for a run."""
    if _is_in_flight_run(run):
        return stopped_at, SubStatus.STOPPED, "Run stopped during server upgrade."
    return (
        _timestamp_from_run_value(run["finished_at"]),
        run["sub_status"] or "",
        run["details"] or "",
    )


def _create_task_table() -> None:
    """Create the final task table."""
    op.create_table(
        "task",
        sa.Column("task_id", sa.BigInteger(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("fab_hash", sa.String(), nullable=True),
        sa.Column("model_ref", sa.String(), nullable=True),
        sa.Column("connector_ref", sa.String(), nullable=True),
        sa.Column("token", sa.String(), nullable=True),
        sa.Column("active_until", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("pending_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("starting_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("running_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("finished_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "sub_status", sa.String(), server_default=sa.text("''"), nullable=False
        ),
        sa.Column("details", sa.String(), server_default=sa.text("''"), nullable=False),
        sa.UniqueConstraint("task_id"),
    )
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.create_index("idx_task_run_id", ["run_id"], unique=False)
        batch_op.create_index("idx_task_token", ["token"], unique=False)
        batch_op.create_index("idx_task_active_until", ["active_until"], unique=False)


def _create_task_logs_table() -> None:
    """Create the final task_logs table."""
    op.create_table(
        "task_logs",
        sa.Column("timestamp", sa.Float(), nullable=False),
        sa.Column("task_id", sa.BigInteger(), nullable=False),
        sa.Column("log", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["task_id"], ["task.task_id"]),
    )
    with op.batch_alter_table("task_logs", schema=None) as batch_op:
        batch_op.create_index(
            "idx_task_logs_task_id_timestamp",
            ["task_id", "timestamp"],
            unique=False,
        )


def _create_task_message_table() -> None:
    """Create the final task_message table."""
    op.create_table(
        "task_message",
        sa.Column("message_id", sa.String(), nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("src_task_id", sa.BigInteger(), nullable=False),
        sa.Column("dst_task_id", sa.BigInteger(), nullable=False),
        sa.Column("reply_to_message_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("ttl", sa.Float(), nullable=False),
        sa.Column("message_type", sa.String(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=True),
        sa.Column("error", sa.LargeBinary(), nullable=True),
        sa.ForeignKeyConstraint(["dst_task_id"], ["task.task_id"]),
        sa.ForeignKeyConstraint(["src_task_id"], ["task.task_id"]),
        sa.PrimaryKeyConstraint("message_id"),
    )
    with op.batch_alter_table("task_message", schema=None) as batch_op:
        batch_op.create_index(
            "idx_task_message_dst_task_id_created_at",
            ["dst_task_id", "created_at"],
            unique=False,
        )
        batch_op.create_index("idx_task_message_run_id", ["run_id"], unique=False)


def _widen_integer_columns() -> None:
    """Widen existing integer identifier and counter columns."""
    with op.batch_alter_table("context", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )

    with op.batch_alter_table("logs", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "node_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )

    with op.batch_alter_table("message_ins", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "src_node_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "dst_node_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )

    with op.batch_alter_table("message_res", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "src_node_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "dst_node_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )

    with op.batch_alter_table("node", schema=None) as batch_op:
        batch_op.alter_column(
            "node_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("primary_task_id", sa.BigInteger(), nullable=True)
        )
        batch_op.alter_column(
            "run_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "bytes_sent",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
            existing_server_default=sa.text("'0'"),
        )
        batch_op.alter_column(
            "bytes_recv",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=True,
            existing_server_default=sa.text("'0'"),
        )

    with op.batch_alter_table("run_objects", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=False,
        )


def _narrow_integer_columns() -> None:
    """Restore identifier and counter columns to the parent revision types."""
    with op.batch_alter_table("run_objects", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=False,
        )

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.alter_column(
            "bytes_recv",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
            existing_server_default=sa.text("'0'"),
        )
        batch_op.alter_column(
            "bytes_sent",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
            existing_server_default=sa.text("'0'"),
        )
        batch_op.alter_column(
            "run_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )
        batch_op.drop_column("primary_task_id")

    with op.batch_alter_table("node", schema=None) as batch_op:
        batch_op.alter_column(
            "node_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )

    with op.batch_alter_table("message_res", schema=None) as batch_op:
        batch_op.alter_column(
            "dst_node_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "src_node_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "run_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )

    with op.batch_alter_table("message_ins", schema=None) as batch_op:
        batch_op.alter_column(
            "dst_node_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "src_node_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "run_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )

    with op.batch_alter_table("logs", schema=None) as batch_op:
        batch_op.alter_column(
            "node_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )
        batch_op.alter_column(
            "run_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )

    with op.batch_alter_table("context", schema=None) as batch_op:
        batch_op.alter_column(
            "run_id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=True,
        )


def _upgrade_node_online_until() -> None:
    """Convert node.online_until from timestamp to epoch seconds."""
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


def _downgrade_node_online_until() -> None:
    """Convert node.online_until from epoch seconds to timestamp."""
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


def _add_object_refcount_check() -> None:
    """Backfill object ref counts and add the nonnegative check constraint."""
    op.execute(
        """
        UPDATE objects
        SET ref_count = (
            SELECT COUNT(*)
            FROM object_children
            WHERE object_children.child_id = objects.object_id
        )
        """
    )
    with op.batch_alter_table("objects", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "ck_objects_ref_count_nonnegative",
            "ref_count >= 0",
        )


def _drop_object_refcount_check() -> None:
    """Drop the object ref_count check constraint."""
    with op.batch_alter_table("objects", schema=None) as batch_op:
        batch_op.drop_constraint(
            "ck_objects_ref_count_nonnegative",
            type_="check",
        )


def _backfill_primary_tasks(runs: Sequence[RowMapping]) -> None:
    """Create one primary task per historical run and link it from the run row."""
    bind = op.get_bind()
    stopped_at = now()

    insert_task_query = sa.text(
        """
        INSERT INTO task
        (task_id, type, run_id, fab_hash, model_ref, connector_ref, token,
         active_until, pending_at, starting_at, running_at, finished_at,
         sub_status, details)
        VALUES
        (:task_id, :type, :run_id, :fab_hash, :model_ref, :connector_ref, :token,
         :active_until, :pending_at, :starting_at, :running_at, :finished_at,
         :sub_status, :details)
        """
    )
    update_run_query = sa.text(
        "UPDATE run SET primary_task_id = :primary_task_id WHERE run_id = :run_id"
    )
    reserved_task_ids: set[int] = set()

    for run in runs:
        finished_at, sub_status, details = _backfilled_primary_task_status(
            run, stopped_at
        )
        task_id = _generate_unique_task_id(bind, reserved_task_ids)
        bind.execute(
            insert_task_query,
            {
                "task_id": uint64_to_int64(task_id),
                "type": _primary_task_type_from_run_type(run["run_type"]),
                "run_id": run["run_id"],
                "fab_hash": run["fab_hash"] or None,
                "model_ref": None,
                "connector_ref": None,
                "token": None,
                "active_until": None,
                "pending_at": _timestamp_from_run_value(run["pending_at"]),
                "starting_at": _timestamp_from_run_value(run["starting_at"]),
                "running_at": _timestamp_from_run_value(run["running_at"]),
                "finished_at": finished_at,
                "sub_status": sub_status,
                "details": details,
            },
        )
        bind.execute(
            update_run_query,
            {"primary_task_id": uint64_to_int64(task_id), "run_id": run["run_id"]},
        )


def _copy_logs_to_task_logs() -> None:
    """Copy existing run logs into the task log table."""
    op.execute(
        """
        INSERT INTO task_logs (timestamp, task_id, log)
        SELECT logs.timestamp, run.primary_task_id, logs.log
        FROM logs
        JOIN run ON run.run_id = logs.run_id
        JOIN task ON task.task_id = run.primary_task_id
        WHERE logs.timestamp IS NOT NULL
          AND logs.log IS NOT NULL
        """
    )


def _copy_task_logs_to_logs() -> None:
    """Copy task logs back into the parent revision log table."""
    op.execute(
        """
        INSERT INTO logs (timestamp, run_id, node_id, log)
        SELECT task_logs.timestamp, task.run_id, 0, task_logs.log
        FROM task_logs
        JOIN task ON task.task_id = task_logs.task_id
        WHERE NOT EXISTS (
            SELECT 1
            FROM logs
            WHERE logs.timestamp = task_logs.timestamp
              AND logs.run_id = task.run_id
              AND logs.node_id = 0
        )
        """
    )


def _drop_run_status_columns() -> None:
    """Drop run status columns moved to task."""
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.drop_column("pending_at")
        batch_op.drop_column("starting_at")
        batch_op.drop_column("running_at")
        batch_op.drop_column("finished_at")
        batch_op.drop_column("sub_status")
        batch_op.drop_column("details")


def _add_run_status_columns() -> None:
    """Restore run status columns from the parent revision."""
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.add_column(sa.Column("pending_at", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("starting_at", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("running_at", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("finished_at", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("sub_status", sa.VARCHAR(), nullable=True))
        batch_op.add_column(sa.Column("details", sa.VARCHAR(), nullable=True))


def _restore_run_status_columns_from_primary_tasks() -> None:
    """Copy primary task status data back to run rows before downgrade."""
    bind = op.get_bind()
    rows = bind.execute(
        sa.text(
            """
            SELECT r.run_id, t.pending_at, t.starting_at, t.running_at,
                   t.finished_at, t.sub_status, t.details
            FROM run AS r
            JOIN task AS t ON t.task_id = r.primary_task_id
            """
        )
    ).mappings()
    update_query = sa.text(
        """
        UPDATE run
        SET pending_at = :pending_at,
            starting_at = :starting_at,
            running_at = :running_at,
            finished_at = :finished_at,
            sub_status = :sub_status,
            details = :details
        WHERE run_id = :run_id
        """
    )
    for row in rows:
        bind.execute(
            update_query,
            {
                "run_id": row["run_id"],
                "pending_at": _timestamp_to_run_value(row["pending_at"]),
                "starting_at": _timestamp_to_run_value(row["starting_at"]),
                "running_at": _timestamp_to_run_value(row["running_at"]),
                "finished_at": _timestamp_to_run_value(row["finished_at"]),
                "sub_status": row["sub_status"],
                "details": row["details"],
            },
        )


def _drop_task_tables() -> None:
    """Drop task-related tables."""
    with op.batch_alter_table("task_message", schema=None) as batch_op:
        batch_op.drop_index("idx_task_message_run_id")
        batch_op.drop_index("idx_task_message_dst_task_id_created_at")
    op.drop_table("task_message")

    with op.batch_alter_table("task_logs", schema=None) as batch_op:
        batch_op.drop_index("idx_task_logs_task_id_timestamp")
    op.drop_table("task_logs")
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.drop_index("idx_task_run_id")
        batch_op.drop_index("idx_task_token")
        batch_op.drop_index("idx_task_active_until")
    op.drop_table("task")


def _create_token_store_table() -> None:
    """Recreate the token_store table from the parent revision."""
    op.create_table(
        "token_store",
        sa.Column("run_id", sa.INTEGER(), nullable=True),
        sa.Column("token", sa.VARCHAR(), nullable=False),
        sa.Column("active_until", sa.FLOAT(), nullable=True),
        sa.PrimaryKeyConstraint("run_id"),
        sa.UniqueConstraint("token"),
    )


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    runs = _load_runs_for_primary_task_backfill(bind)
    _validate_primary_task_backfill_runs(runs)

    _create_task_table()
    _create_task_logs_table()
    _create_task_message_table()

    _drop_run_id_foreign_keys()
    _widen_integer_columns()
    _upgrade_node_online_until()
    _create_run_id_foreign_keys()
    _add_object_refcount_check()

    _backfill_primary_tasks(runs)
    _copy_logs_to_task_logs()

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.alter_column(
            "primary_task_id",
            existing_type=sa.BigInteger(),
            nullable=False,
        )

    _drop_run_status_columns()
    op.drop_table("token_store")


def downgrade() -> None:
    """Downgrade schema."""
    _drop_object_refcount_check()
    _add_run_status_columns()
    _restore_run_status_columns_from_primary_tasks()
    _copy_task_logs_to_logs()
    _drop_task_tables()

    _drop_run_id_foreign_keys()
    _downgrade_node_online_until()
    _narrow_integer_columns()
    _create_run_id_foreign_keys()

    _create_token_store_table()
