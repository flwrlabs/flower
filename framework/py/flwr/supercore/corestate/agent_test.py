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

from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    SpaceParticipant,
)
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from . import CoreState


@pytest.fixture(name="state", params=["in-memory", ":memory:", "sqlite"])
def corestate_fixture(request: pytest.FixtureRequest, tmp_path: Path) -> CoreState:
    """Provide in-memory and SQL state."""
    manager = NoOpFederationManager()
    object_store = ObjectStoreFactory().store()
    if request.param == "in-memory":
        return InMemoryLinkState(manager, object_store)
    state = SqlLinkState(
        database_path=(
            ":memory:" if request.param == ":memory:" else str(tmp_path / "state.db")
        ),
        federation_manager=manager,
        object_store=object_store,
    )
    state.initialize()
    return state


def test_create_and_list_agents(state: CoreState) -> None:
    """Store configuration and timestamps, allow duplicate names, and isolate owners."""
    assert state.list_agents("alice") == []
    config = AgentConfig(
        name="Research Agent",
        app_spec="@flower/research==1.2.3",
        title="Researcher",
        description="Investigate and cite sources.",
    )
    minimal_config = AgentConfig(name="Research Agent", app_spec="@flower/research")
    first = state.create_agent(flwr_aid="alice", config=config)
    other = state.create_agent(flwr_aid="bob", config=minimal_config)
    second = state.create_agent(flwr_aid="alice", config=config)

    assert first is not None and other is not None and second is not None
    assert 0 < first.id < other.id < second.id
    assert first.config == config
    assert other.config == minimal_config
    assert first.created_at and first.updated_at == first.created_at
    assert state.list_agents("alice") == [first, second]
    assert state.list_agents("bob") == [other]
    assert state.list_agents("unknown") == []
    assert state.list_agents("") == []


def test_create_agent_with_space(state: CoreState) -> None:
    """Each new agent gets an empty Space containing only itself and its creator."""
    config = AgentConfig(name="Research", app_spec="@flower/research")
    agents = [state.create_agent_with_space("alice", config) for _ in range(2)]
    spaces = state.get_spaces("alice")
    assert len(spaces) == 2 and spaces[0].id != spaces[1].id
    for agent in agents:
        assert agent is not None
        member = SpaceParticipant(agent_id=agent.id)
        space = next(space for space in spaces if member in space.participants)
        assert len(space.participants) == 2
        assert SpaceParticipant(flwr_aid="alice") in space.participants
        assert space.title == ""
        assert state.get_space_messages(UUID(space.id), "alice") == []
    assert state.get_spaces("bob") == []
    assert not state.get_tasks()


def test_agent_creation_rolls_back_when_space_creation_fails(state: CoreState) -> None:
    """Space creation failure must not leave an agent behind."""
    with patch.object(state, "create_space", side_effect=ValueError("Space failed")):
        with pytest.raises(ValueError, match="Space failed"):
            state.create_agent_with_space(
                "alice", AgentConfig(name="Research", app_spec="@flower/research")
            )
    assert state.list_agents("alice") == []
    assert state.get_spaces("alice") == []


@pytest.mark.parametrize(
    "field", ["flwr_aid", "name", "app_spec", "title", "description"]
)
def test_create_agent_rejects_empty_fields(state: CoreState, field: str) -> None:
    """Reject invalid creation without storing an agent."""
    values = {
        "name": "Research Agent",
        "app_spec": "@flower/research",
    }
    flwr_aid = "alice"
    if field == "flwr_aid":
        flwr_aid = ""
    else:
        values[field] = ""

    assert state.create_agent(flwr_aid=flwr_aid, config=AgentConfig(**values)) is None
    assert (
        state.create_agent_with_space(flwr_aid=flwr_aid, config=AgentConfig(**values))
        is None
    )
    assert state.list_agents("alice") == []
    assert state.list_agents("") == []
    assert state.get_spaces("alice") == []
