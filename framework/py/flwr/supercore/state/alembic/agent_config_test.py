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
"""Tests for agent configuration migration and version ownership."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from sqlalchemy import MetaData, create_engine, select
from sqlalchemy.exc import IntegrityError

from flwr.supercore.state.alembic.utils import build_alembic_config
from flwr.supercore.state.schema.corestate_models import Agent, AgentConfig, Space


def test_migration_discards_agents_without_backfill(tmp_path: Path) -> None:
    """Delete agents and their dependents while retaining unrelated state and IDs."""
    engine = create_engine(f"sqlite:///{tmp_path / 'state.db'}")
    config = build_alembic_config(engine)
    command.upgrade(config, "0d0ae5ae64ef")
    metadata = MetaData()
    metadata.reflect(engine)
    agent_table = Agent.metadata.tables["agent"]
    config_table = AgentConfig.metadata.tables["agent_config"]
    timestamp = datetime.now(UTC)
    space_id = UUID(int=1).hex
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        conn.execute(
            metadata.tables["space"].insert(),
            {"id": space_id, "title": "Keep", "created_at": timestamp},
        )
        conn.execute(
            metadata.tables["agent"].insert(),
            {
                "agent_id": 42,
                "name": "Discard",
                "app_spec": "@flower/app",
                "flwr_aid": "alice",
                "default_space_id": space_id,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        for owner in ({"flwr_aid": "alice"}, {"agent_id": 42}):
            conn.execute(
                metadata.tables["space_participant"].insert(),
                {"space_id": space_id, **owner},
            )
            conn.execute(
                metadata.tables["space_message"].insert(),
                {
                    "space_id": space_id,
                    "content": "Message",
                    "created_at": timestamp,
                    **owner,
                },
            )
        conn.execute(
            metadata.tables["run"].insert(),
            {"run_id": 1, "primary_task_id": 1, "agent_id": 42},
        )
        conn.execute(
            metadata.tables["compacted_contexts"].insert(),
            {
                "agent_id": 42,
                "through_event_id": 1,
                "text": "Discard",
                "token_count": 1,
                "created_at": timestamp,
            },
        )
        conn.execute(
            metadata.tables["automation"].insert(),
            {
                "flwr_aid": "alice",
                "agent_id": 42,
                "status": "active",
                "created_at": timestamp,
                "updated_at": timestamp,
                "next_run_at": timestamp,
            },
        )
        for connector_id, owner in (
            (1, {"agent_id": 42}),
            (2, {"federation_id": "@alice/fed"}),
        ):
            conn.execute(
                metadata.tables["connector"].insert(),
                {
                    "connector_id": connector_id,
                    "connector_ref": "slack",
                    "config_json": "{}",
                    "credentials_json": "{}",
                    "created_at": timestamp,
                    "created_by": "alice",
                    **owner,
                },
            )
            conn.execute(
                metadata.tables["run_connector"].insert(),
                {"run_id": 1, "connector_id": connector_id},
            )
        conn.execute(
            metadata.tables["connector_oauth_session"].insert(),
            {
                "oauth_session_id": "discard",
                "flwr_aid": "alice",
                "agent_id": 42,
                "connector_ref": "slack",
                "state": "state",
                "redirect_uri": "http://localhost/callback",
                "created_at": timestamp,
                "expires_at": timestamp,
            },
        )

    command.upgrade(config, "8989c696c5bb")
    metadata = MetaData()
    metadata.reflect(engine)
    with engine.begin() as conn:
        for name in (
            "agent",
            "agent_config",
            "compacted_contexts",
            "automation",
            "connector_oauth_session",
        ):
            assert conn.execute(metadata.tables[name].select()).all() == []
        assert conn.execute(metadata.tables["space"].select()).one().title == "Keep"
        assert (
            conn.execute(metadata.tables["space_message"].select()).one().flwr_aid
            == "alice"
        )
        assert (
            conn.execute(metadata.tables["space_participant"].select()).one().flwr_aid
            == "alice"
        )
        assert (
            conn.execute(metadata.tables["connector"].select()).one().connector_id == 2
        )
        assert (
            conn.execute(metadata.tables["run_connector"].select()).one().connector_id
            == 2
        )
        run = conn.execute(metadata.tables["run"].select()).one()
        assert run.agent_id is None and run.agent_config_version is None
        assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []

        agent_id = conn.scalar(
            agent_table.insert()
            .values(
                flwr_aid="alice",
                config_version=1,
                revision=1,
                created_at=timestamp,
                updated_at=timestamp,
            )
            .returning(agent_table.c.agent_id)
        )
        assert isinstance(agent_id, int)
        assert agent_id > 42
        conn.execute(
            config_table.insert().values(
                agent_id=agent_id,
                version=1,
                name="New",
                app_spec="@flower/new",
                created_at=timestamp,
            )
        )
        conn.execute(
            metadata.tables["space_participant"].insert(),
            {"space_id": space_id, "agent_id": agent_id},
        )
        conn.execute(
            metadata.tables["run"].insert(),
            {
                "run_id": 2,
                "primary_task_id": 2,
                "agent_id": agent_id,
                "agent_config_version": 1,
            },
        )

    command.downgrade(config, "0d0ae5ae64ef")
    metadata = MetaData()
    metadata.reflect(engine)
    assert "agent_config" not in metadata.tables
    with engine.connect() as conn:
        agent = conn.execute(metadata.tables["agent"].select()).one()
        assert agent.agent_id == agent_id
        assert agent.name == "New" and agent.app_spec == "@flower/new"
        assert conn.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    engine.dispose()


@pytest.mark.parametrize("version", [2, 3])
def test_current_configuration_must_exist_for_same_agent(version: int) -> None:
    """Reject a missing version and a version that belongs only to another agent."""
    engine = create_engine("sqlite:///:memory:")
    agent_table = Agent.metadata.tables["agent"]
    config_table = AgentConfig.metadata.tables["agent_config"]
    timestamp = datetime.now(UTC)
    with engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
        Space.metadata.tables["space"].create(conn)
        agent_table.create(conn)
        config_table.create(conn)
        for agent_id, config_version in ((1, 1), (2, 2)):
            conn.execute(
                agent_table.insert().values(
                    agent_id=agent_id,
                    flwr_aid="alice",
                    config_version=config_version,
                    revision=1,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
            conn.execute(
                config_table.insert().values(
                    agent_id=agent_id,
                    version=config_version,
                    name="Agent",
                    app_spec="@flower/app",
                    created_at=timestamp,
                )
            )

    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        with engine.begin() as conn:
            conn.execute(
                agent_table.update()
                .where(Agent.agent_id == 1)
                .values(config_version=version)
            )
    with engine.connect() as conn:
        assert conn.scalar(select(Agent.config_version).where(Agent.agent_id == 1)) == 1
    engine.dispose()
