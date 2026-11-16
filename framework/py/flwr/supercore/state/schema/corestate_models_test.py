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

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from flwr.supercore.state.schema.corestate_models import (
    Event,
    FlwrBase,
    Task,
    TaskLogsTable,
)
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.typing import JSONObject

CORESTATE_TABLE_NAMES = {
    "nonce_store",
    "fab",
    "file",
    "file_revision",
    "message",
    "agent",
    "space",
    "space_participant",
    "space_message",
    "events",
    "user_events",
    "user_event_feeds",
    "run_series",
    "compacted_contexts",
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


def test_event_payload_round_trips_nested_json() -> None:
    """Map and persist recursive JSON values through the Event ORM model."""
    payload: JSONObject = {
        "nested": {"values": [None, True, 42, 1.5, "text", {"items": []}]},
    }
    engine = create_engine("sqlite:///:memory:")
    FlwrBase.metadata.tables["events"].create(engine)
    event_id = UUID(int=1)

    with Session(engine) as session:
        session.add(
            Event(
                event_id=event_id,
                type="test",
                payload=payload,
                created_at=datetime.now(UTC),
            )
        )
        session.commit()

    with Session(engine) as session:
        event = session.get(Event, event_id)
        assert event is not None
        assert event.payload == payload


def test_connector_uses_incrementing_id_and_allows_multiple_provider_accounts() -> None:
    """Ensure connectors use an ID without provider-level uniqueness."""
    table = FlwrBase.metadata.tables["connector"]

    assert [column.name for column in table.primary_key.columns] == ["connector_id"]
    assert table.c.connector_id.autoincrement is True
    for column_name in ("federation_id", "created_at", "created_by"):
        assert table.c[column_name].nullable is False
    for column_name in ("deleted_at", "deleted_by"):
        assert table.c[column_name].nullable is True


def test_task_logs_table_remains_unmapped_without_unique_identity_key() -> None:
    """Ensure keyless task_logs stays a table, not an ORM mapper."""
    assert "task_logs" in create_corestate_metadata().tables
    assert TaskLogsTable.primary_key.columns.values() == []


@pytest.mark.parametrize("table_name", ["space_participant", "space_message"])
@pytest.mark.parametrize(
    ("flwr_aid", "agent_id", "valid"),
    [("user", None, True), (None, 1, True), (None, None, False), ("user", 1, False)],
)
def test_space_requires_exactly_one_identity(
    table_name: str,
    flwr_aid: str | None,
    agent_id: int | None,
    valid: bool,
) -> None:
    """Accept either identity and reject both or neither in both tables."""
    table = FlwrBase.metadata.tables[table_name]
    engine = create_engine("sqlite:///:memory:")
    table.create(engine)
    values: dict[str, object] = {
        "space_id": UUID(int=1),
        "flwr_aid": flwr_aid,
        "agent_id": agent_id,
    }
    if table_name == "space_message":
        values.update(content="Hello", created_at=datetime.now(UTC))

    with engine.begin() as conn:
        if valid:
            conn.execute(table.insert().values(**values))
        else:
            with pytest.raises(IntegrityError, match="CHECK constraint failed"):
                conn.execute(table.insert().values(**values))
