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
"""Tests for account-owned agents across CoreState implementations."""

from datetime import UTC
from pathlib import Path

import pytest

from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.date import now
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from . import CoreState


def _sql_state(database_path: str) -> SqlLinkState:
    """Create and initialize SQL state for agent tests."""
    state = SqlLinkState(
        database_path=database_path,
        federation_manager=NoOpFederationManager(),
        object_store=ObjectStoreFactory().store(),
    )
    state.initialize()
    return state


@pytest.fixture(name="state", params=["in-memory", "sqlite-memory", "sqlite-file"])
def corestate_fixture(request: pytest.FixtureRequest, tmp_path: Path) -> CoreState:
    """Provide each supported state backend."""
    if request.param == "in-memory":
        return InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())
    database_path = (
        ":memory:" if request.param == "sqlite-memory" else str(tmp_path / "state.db")
    )
    return _sql_state(database_path)


def test_create_agent_assigns_id_and_timestamps(state: CoreState) -> None:
    """Create an agent without requiring optional fields or a stored app."""
    before = now()
    agent = state.create_agent(
        flwr_aid="alice", name="Research Agent", app_spec="@flower/research"
    )
    after = now()

    assert agent is not None
    assert agent.agent_id > 0
    assert agent.name == "Research Agent"
    assert agent.flwr_aid == "alice"
    assert agent.app_spec == "@flower/research"
    assert agent.title is None
    assert agent.description is None
    assert before <= agent.created_at <= after
    assert agent.created_at.tzinfo == UTC
    assert agent.updated_at == agent.created_at
    assert state.list_agents("alice") == [agent]


@pytest.mark.parametrize("title", [None, "Researcher 🌱"])
@pytest.mark.parametrize("description", [None, "Investigate carefully.\nCite sources."])
def test_agent_optional_fields_roundtrip(
    state: CoreState, title: str | None, description: str | None
) -> None:
    """Preserve null and populated text without normalization."""
    agent = state.create_agent(
        flwr_aid="alice",
        name="Research Agent",
        app_spec="@flower/research==1.2.3",
        title=title,
        description=description,
    )

    assert agent is not None
    assert agent.title == title
    assert agent.description == description
    assert agent.app_spec == "@flower/research==1.2.3"
    assert state.list_agents("alice") == [agent]


def test_list_agents_filters_owner_and_allows_duplicate_names(state: CoreState) -> None:
    """Keep ownership isolated and return duplicate names in creation order."""
    assert state.list_agents("alice") == []
    first = state.create_agent(
        flwr_aid="alice", name="Research Agent", app_spec="@flower/research"
    )
    other = state.create_agent(
        flwr_aid="bob", name="Research Agent", app_spec="@flower/research"
    )
    second = state.create_agent(
        flwr_aid="alice", name="Research Agent", app_spec="@flower/research==1.2.3"
    )

    assert first is not None and other is not None and second is not None
    assert first.agent_id < other.agent_id < second.agent_id
    assert state.list_agents("alice") == [first, second]
    assert state.list_agents("bob") == [other]
    assert state.list_agents("unknown") == []
    assert state.list_agents("") == []


@pytest.mark.parametrize(
    "field", ["flwr_aid", "name", "app_spec", "title", "description"]
)
def test_create_agent_rejects_empty_fields(state: CoreState, field: str) -> None:
    """Reject invalid creation without storing an agent."""
    values = {
        "flwr_aid": "alice",
        "name": "Research Agent",
        "app_spec": "@flower/research",
    }
    values[field] = ""

    assert state.create_agent(**values) is None
    assert state.list_agents("alice") == []
    assert state.list_agents("") == []
