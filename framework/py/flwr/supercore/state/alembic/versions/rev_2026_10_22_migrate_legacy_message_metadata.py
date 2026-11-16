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
"""Migrate legacy message metadata.

Migration rules:
- Copy message_ins before message_res, preserving IDs and correlation references.
- Insert metadata in batches of 1000 rows using SQLAlchemy executemany.
- Preserve shared metadata and the stored signed BIGINT representation of IDs.
  Empty correlation references become NULL; task IDs are unknown and stay NULL.
- Set expires_at to created_at + ttl in UTC, without using TTL to infer cleared.
- Without delivered_at, cleared is False. Otherwise, cleared is True only if the
  message root is absent from ObjectStore. A preregistered root with empty content
  still counts as present.
- Leave acknowledged_at and cancelled_at NULL; their history is unknown.
- Skip message IDs already present in message without changing their delivery state.
  Run the migration without concurrent application writes to message.
- Missing required fields or a missing correlation target abort migration.
- Copy metadata only; do not modify legacy tables, ObjectStore, or push sessions.
- Downgrade does not remove copied rows or revert their delivery state.

Revision ID: 4c9b3f185c5b
Revises: d9bd9d502cf9
Create Date: 2026-10-22 15:38:27.691743
"""
from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op

from flwr.supercore.state.schema.types import UTCDateTime

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "4c9b3f185c5b"
down_revision: str | Sequence[str] | None = "d9bd9d502cf9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Copy legacy metadata without invoking current message policies."""
    connection = op.get_bind()
    metadata_columns = (
        "message_id",
        "run_id",
        "src_node_id",
        "dst_node_id",
        "reply_to_message_id",
        "group_id",
        "created_at",
        "ttl",
        "message_type",
    )
    # Freeze the columns used by this revision rather than importing evolving models.
    message = sa.table(
        "message",
        *(sa.column(name) for name in metadata_columns),
        sa.column("src_task_id"),
        sa.column("dst_task_id"),
        sa.column("expires_at", UTCDateTime()),
        sa.column("cleared", sa.Boolean()),
        sa.column("acknowledged_at", UTCDateTime()),
        sa.column("cancelled_at", UTCDateTime()),
    )
    objects = sa.table("objects", sa.column("object_id"))

    for table_name in ("message_ins", "message_res"):
        legacy = sa.table(
            table_name,
            *(sa.column(name) for name in metadata_columns),
            sa.column("delivered_at"),
        )
        not_migrated = ~sa.exists().where(message.c.message_id == legacy.c.message_id)
        orphan = connection.execute(
            sa.select(legacy.c.message_id, legacy.c.reply_to_message_id)
            .where(
                not_migrated,
                sa.func.nullif(legacy.c.reply_to_message_id, "").is_not(None),
                ~sa.exists().where(
                    message.c.message_id == legacy.c.reply_to_message_id
                ),
            )
            .limit(1)
        ).first()
        if orphan is not None:
            raise ValueError(
                f"Missing original message {orphan.reply_to_message_id} "
                f"for {orphan.message_id}"
            )

        cleared = sa.case(
            (sa.func.coalesce(legacy.c.delivered_at, "") == "", sa.false()),
            else_=~sa.exists().where(objects.c.object_id == legacy.c.message_id),
        )
        with connection.execute(
            sa.select(legacy, cleared.label("cleared")).where(not_migrated),
            execution_options={"yield_per": 1000},
        ) as result:
            for batch in result.mappings().partitions(1000):
                rows = []
                for row in batch:
                    values = {name: row[name] for name in metadata_columns}
                    values["reply_to_message_id"] = (
                        values["reply_to_message_id"] or None
                    )
                    if not values["message_id"]:
                        raise ValueError(f"Missing message ID in {table_name}")
                    values.update(
                        src_task_id=None,
                        dst_task_id=None,
                        expires_at=datetime.fromtimestamp(
                            values["created_at"] + values["ttl"], tz=UTC
                        ),
                        cleared=row["cleared"],
                        acknowledged_at=None,
                        cancelled_at=None,
                    )
                    rows.append(values)
                connection.execute(message.insert(), rows)


def downgrade() -> None:
    """Retain copied metadata and its potentially updated delivery state."""
