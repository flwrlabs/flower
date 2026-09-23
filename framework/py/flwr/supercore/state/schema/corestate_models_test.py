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
"""Tests for CoreState declarative models."""

from sqlalchemy import UniqueConstraint

from flwr.supercore.state.schema.corestate_models import FlwrBase, Task, TaskLogsTable
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata

CORESTATE_TABLE_NAMES = {
    "nonce_store",
    "fab",
    "run_series",
    "series_context",
    "series_runs",
    "automation",
    "federation_app",
    "connector",
    "connector_oauth_session",
    "run_connector",
    "task",
    "task_event",
    "task_message",
    "task_logs",
    "object_push_sessions",
    "object_push_session_roots",
    "object_push_session_pending",
    "task_usage",
}


def test_declarative_metadata_covers_all_mappable_core_tables() -> None:
    """Ensure CoreState metadata is fully defined by model metadata."""
    core_table_names = set(create_corestate_metadata().tables)
    model_table_names = set(FlwrBase.metadata.tables)

    assert model_table_names == CORESTATE_TABLE_NAMES
    assert core_table_names == model_table_names


def test_task_mapper_uses_task_id_as_identity_key() -> None:
    """Ensure the mapper-only primary key uses the existing unique task_id column."""
    assert [column.name for column in Task.__mapper__.primary_key] == ["task_id"]


def test_connector_uses_incrementing_id_and_unique_federation_reference() -> None:
    """Ensure connectors have a surrogate ID and retain their natural uniqueness."""
    table = FlwrBase.metadata.tables["connector"]

    assert [column.name for column in table.primary_key.columns] == ["connector_id"]
    assert table.c.connector_id.autoincrement is True
    assert any(
        {column.name for column in constraint.columns}
        == {"federation_id", "connector_ref"}
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    )
    for column_name in ("federation_id", "created_at", "created_by"):
        assert table.c[column_name].nullable is False
    for column_name in ("deleted_at", "deleted_by"):
        assert table.c[column_name].nullable is True


def test_task_logs_table_remains_unmapped_without_unique_identity_key() -> None:
    """Ensure keyless task_logs stays a table, not an ORM mapper."""
    assert "task_logs" in create_corestate_metadata().tables
    assert TaskLogsTable.primary_key.columns.values() == []
