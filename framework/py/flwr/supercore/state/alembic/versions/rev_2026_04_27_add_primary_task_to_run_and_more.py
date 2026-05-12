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
"""Add primary task metadata to run and task tables.

Revision ID: 8253e456d570
Revises: dee9b802b5c9
Create Date: 2026-04-27 13:00:44.155029
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine import Connection, RowMapping

from flwr.common.constant import TASK_ID_NUM_BYTES, SubStatus
from flwr.supercore.constant import RunType, TaskType
from flwr.supercore.corestate.utils import generate_rand_int_from_bytes
from flwr.supercore.date import now
from flwr.supercore.utils import uint64_to_int64

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "8253e456d570"
down_revision: str | Sequence[str] | None = "dee9b802b5c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _primary_task_type_from_run_type(run_type: str) -> str:
    """Return the primary task type for the given run type."""
    if run_type == RunType.SIMULATION:
        return TaskType.SIMULATION
    if run_type == RunType.SERVER_APP:
        return TaskType.SERVER_APP
    raise RuntimeError(
        f"Unsupported run_type while backfilling primary tasks: {run_type}"
    )


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
    run: RowMapping, stopped_at: str
) -> tuple[str | None, str, str]:
    """Return the backfilled finished_at, sub_status, and details for a run."""
    if _is_in_flight_run(run):
        return stopped_at, SubStatus.STOPPED, "Run stopped during server upgrade."
    return run["finished_at"] or None, run["sub_status"] or "", run["details"] or ""


def _backfill_primary_tasks(runs: Sequence[RowMapping]) -> None:
    """Create one primary task per historical run and link it from the run row."""
    bind = op.get_bind()
    stopped_at = now().isoformat()

    insert_task_query = sa.text(
        """
        INSERT INTO task
        (task_id, type, run_id, fab_hash, model_ref, connector_ref, token,
         pending_at, starting_at, running_at, finished_at, sub_status, details)
        VALUES
        (:task_id, :type, :run_id, :fab_hash, :model_ref, :connector_ref, :token,
         :pending_at, :starting_at, :running_at, :finished_at, :sub_status, :details)
        """
    )
    update_run_query = sa.text(
        "UPDATE run SET primary_task_id = :primary_task_id WHERE run_id = :run_id"
    )
    mark_run_stopped_query = sa.text(
        """
        UPDATE run
        SET finished_at = :finished_at, sub_status = :sub_status, details = :details
        WHERE run_id = :run_id
        """
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
                "pending_at": run["pending_at"],
                "starting_at": run["starting_at"] or None,
                "running_at": run["running_at"] or None,
                "finished_at": finished_at,
                "sub_status": sub_status,
                "details": details,
            },
        )
        bind.execute(
            update_run_query,
            {"primary_task_id": uint64_to_int64(task_id), "run_id": run["run_id"]},
        )
        if _is_in_flight_run(run):
            bind.execute(
                mark_run_stopped_query,
                {
                    "run_id": run["run_id"],
                    "finished_at": finished_at,
                    "sub_status": sub_status,
                    "details": details,
                },
            )


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    runs = _load_runs_for_primary_task_backfill(bind)
    _validate_primary_task_backfill_runs(runs)

    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("primary_task_id", sa.BigInteger(), nullable=True)
        )

    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "sub_status",
                sa.String(),
                server_default=sa.text("''"),
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column(
                "details",
                sa.String(),
                server_default=sa.text("''"),
                nullable=False,
            )
        )

    _backfill_primary_tasks(runs)

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.alter_column(
            "primary_task_id",
            existing_type=sa.BigInteger(),
            nullable=False,
        )

    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("task", schema=None) as batch_op:
        batch_op.drop_column("details")
        batch_op.drop_column("sub_status")

    with op.batch_alter_table("run", schema=None) as batch_op:
        batch_op.drop_column("primary_task_id")

    # ### end Alembic commands ###
