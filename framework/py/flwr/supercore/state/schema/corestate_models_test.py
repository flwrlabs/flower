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

from collections.abc import Iterator
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import Connection, Table, create_engine, select
from sqlalchemy.exc import IntegrityError

from flwr.supercore.state.schema.corestate_models import (
    ConversationParticipantTable,
    FlwrBase,
    Task,
    TaskLogsTable,
)
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata

CORESTATE_TABLE_NAMES = {
    "nonce_store",
    "fab",
    "file",
    "file_revision",
    "message",
    "agent",
    "conversation",
    "conversation_participant",
    "conversation_message",
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


@pytest.fixture(name="conversation_connection")
def fixture_conversation_connection() -> Iterator[Connection]:
    """Create conversation tables with one conversation and one agent."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys = ON")
        FlwrBase.metadata.create_all(
            conn,
            tables=[
                FlwrBase.metadata.tables["agent"],
                FlwrBase.metadata.tables["conversation"],
                ConversationParticipantTable,
                FlwrBase.metadata.tables["conversation_message"],
            ],
        )
        now = datetime.now(UTC)
        conn.execute(
            FlwrBase.metadata.tables["agent"]
            .insert()
            .values(
                agent_id=1,
                name="agent",
                app_spec="owner/agent",
                flwr_aid="user",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            FlwrBase.metadata.tables["conversation"]
            .insert()
            .values(id=UUID(int=1), created_at=now, title="Conversation")
        )
        yield conn


def test_conversation_round_trips_generated_uuid_and_timestamp(
    conversation_connection: Connection,
) -> None:
    """Generate UUID v4 IDs and preserve UTC timestamps."""
    table = FlwrBase.metadata.tables["conversation"]
    now = datetime.now(UTC)
    conversation_id = conversation_connection.execute(
        table.insert()
        .values(created_at=now, title="New conversation")
        .returning(table.c.id)
    ).scalar_one()
    row = conversation_connection.execute(
        select(table).where(table.c.id == conversation_id)
    ).one()

    assert isinstance(row.id, UUID)
    assert row.id.version == 4
    assert row.created_at == now
    assert row.title == "New conversation"
    assert set(table.c.keys()) == {"id", "created_at", "title"}


@pytest.mark.parametrize(
    "table",
    [ConversationParticipantTable, FlwrBase.metadata.tables["conversation_message"]],
)
@pytest.mark.parametrize(("flwr_aid", "agent_id"), [("user", None), (None, 1)])
def test_conversation_allows_user_or_agent(
    conversation_connection: Connection,
    table: Table,
    flwr_aid: str | None,
    agent_id: int | None,
) -> None:
    """Allow either identity and round-trip message text and UTC timestamps."""
    values: dict[str, object] = {
        "conversation_id": UUID(int=1),
        "flwr_aid": flwr_aid,
        "agent_id": agent_id,
    }
    now = datetime.now(UTC)
    if table is FlwrBase.metadata.tables["conversation_message"]:
        values.update(content="Hello 🌷\nSecond line", created_at=now)
    conversation_connection.execute(table.insert().values(**values))
    row = conversation_connection.execute(select(table)).one()

    assert row.conversation_id == UUID(int=1)
    assert row.flwr_aid == flwr_aid
    assert row.agent_id == agent_id
    if table is FlwrBase.metadata.tables["conversation_message"]:
        assert row.content == values["content"]
        assert row.created_at == now


@pytest.mark.parametrize(
    "table",
    [ConversationParticipantTable, FlwrBase.metadata.tables["conversation_message"]],
)
@pytest.mark.parametrize(("flwr_aid", "agent_id"), [(None, None), ("user", 1)])
def test_conversation_rejects_missing_or_ambiguous_identity(
    conversation_connection: Connection,
    table: Table,
    flwr_aid: str | None,
    agent_id: int | None,
) -> None:
    """Require exactly one user or agent identity in both tables."""
    values: dict[str, object] = {
        "conversation_id": UUID(int=1),
        "flwr_aid": flwr_aid,
        "agent_id": agent_id,
    }
    if table is FlwrBase.metadata.tables["conversation_message"]:
        values.update(content="Hello", created_at=datetime.now(UTC))

    with pytest.raises(IntegrityError, match="CHECK constraint failed"):
        conversation_connection.execute(table.insert().values(**values))


@pytest.mark.parametrize(("flwr_aid", "agent_id"), [("user", None), (None, 1)])
def test_conversation_participants_are_unique_per_conversation(
    conversation_connection: Connection,
    flwr_aid: str | None,
    agent_id: int | None,
) -> None:
    """Reject duplicate participants while allowing them in other conversations."""
    table = ConversationParticipantTable
    assert set(table.c.keys()) == {"conversation_id", "flwr_aid", "agent_id"}
    conversation_connection.execute(
        FlwrBase.metadata.tables["conversation"]
        .insert()
        .values(
            id=UUID(int=2), created_at=datetime.now(UTC), title="Other conversation"
        )
    )
    for conversation_id in (UUID(int=1), UUID(int=2)):
        conversation_connection.execute(
            table.insert().values(
                conversation_id=conversation_id, flwr_aid=flwr_aid, agent_id=agent_id
            )
        )

    with pytest.raises(IntegrityError, match="UNIQUE constraint failed"):
        conversation_connection.execute(
            table.insert().values(
                conversation_id=UUID(int=1), flwr_aid=flwr_aid, agent_id=agent_id
            )
        )


@pytest.mark.parametrize(
    "table",
    [ConversationParticipantTable, FlwrBase.metadata.tables["conversation_message"]],
)
@pytest.mark.parametrize(
    ("conversation_id", "agent_id"), [(UUID(int=99), 1), (UUID(int=1), 99)]
)
def test_conversation_rejects_unknown_references(
    conversation_connection: Connection,
    table: Table,
    conversation_id: UUID,
    agent_id: int,
) -> None:
    """Require referenced conversations and agents to exist."""
    values: dict[str, object] = {
        "conversation_id": conversation_id,
        "agent_id": agent_id,
    }
    if table is FlwrBase.metadata.tables["conversation_message"]:
        values.update(content="Hello", created_at=datetime.now(UTC))

    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        conversation_connection.execute(table.insert().values(**values))


def test_conversation_message_ids_increase_without_reuse(
    conversation_connection: Connection,
) -> None:
    """Allow repeated authors and keep increasing IDs after a message is deleted."""
    table = FlwrBase.metadata.tables["conversation_message"]
    statement = (
        table.insert()
        .values(
            conversation_id=UUID(int=1),
            flwr_aid="user",
            created_at=datetime.now(UTC),
            content="Hello",
        )
        .returning(table.c.id)
    )
    first_id = conversation_connection.execute(statement).scalar_one()
    second_id = conversation_connection.execute(statement).scalar_one()
    conversation_connection.execute(table.delete().where(table.c.id == second_id))
    third_id = conversation_connection.execute(statement).scalar_one()

    assert 0 < first_id < second_id < third_id
