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

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import patch
from uuid import UUID

import pytest
from google.protobuf.json_format import MessageToDict

from flwr.proto.control_pb2 import Agent, AgentConfig  # pylint: disable=E0611
from flwr.proto.space_pb2 import SpaceParticipant  # pylint: disable=E0611
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import EventType
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.state.schema.corestate_models import AgentConfig as AgentConfigModel
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
        title="Researcher",
        description="Investigate and cite sources.",
        avatar="robot",
        color="sky",
    )
    minimal_config = AgentConfig(name="Research Agent")
    first = state.create_agent("alice", "@flower/research==1.2.3", config)
    other = state.create_agent("bob", "@flower/research", minimal_config)
    second = state.create_agent("alice", "@flower/research==1.2.3", config)

    assert first is not None and other is not None and second is not None
    assert 0 < first.id < other.id < second.id
    assert first.config == config
    assert first.app_spec == second.app_spec == "@flower/research==1.2.3"
    assert other.app_spec == "@flower/research"
    assert first.config_version == other.config_version == second.config_version == 1
    assert other.config == minimal_config
    assert first.created_at and first.updated_at == first.created_at
    assert state.list_agents("alice") == [first, second]
    assert state.list_agents("bob") == [other]
    assert state.get_agent("alice", first.id) == first
    assert state.get_agent("bob", first.id) is None
    assert state.get_agent("alice", other.id) is None
    assert state.list_agents("unknown") == []
    assert state.list_agents("") == []


def test_create_agent_creates_space(state: CoreState) -> None:
    """Each new agent gets an empty Space containing only itself and its creator."""
    config = AgentConfig(name="Research")
    agents = [state.create_agent("alice", "@flower/research", config) for _ in range(2)]
    spaces = state.get_spaces("alice")
    assert len(spaces) == 2 and spaces[0].id != spaces[1].id
    for agent in agents:
        assert agent is not None
        member = SpaceParticipant(agent_id=agent.id)
        space = next(space for space in spaces if member in space.participants)
        assert agent.default_space_id == space.id
        assert len(space.participants) == 2
        assert SpaceParticipant(flwr_aid="alice") in space.participants
        assert space.title == ""
        assert (
            state.get_space_messages(UUID(space.id), SpaceParticipant(flwr_aid="alice"))
            == []
        )
    assert state.get_spaces("bob") == []
    assert not state.get_tasks()


def test_get_default_space_is_owner_scoped(state: CoreState) -> None:
    """Resolve the recorded default even if an agent joins another Space."""
    agent = state.create_agent("alice", "@test/app", AgentConfig(name="Agent"))
    assert agent is not None
    default_space_id = UUID(state.get_spaces("alice")[0].id)
    state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(agent_id=agent.id),
        ]
    )

    assert state.get_default_space("alice", agent.id) == default_space_id
    assert state.get_default_space("bob", agent.id) is None
    assert state.get_default_space("alice", agent.id + 1) is None


@pytest.mark.parametrize(
    "field", ["flwr_aid", "name", "app_spec", "title", "description"]
)
def test_create_agent_rejects_empty_fields(state: CoreState, field: str) -> None:
    """Reject invalid creation without storing an agent."""
    values = {"name": "Research Agent"}
    flwr_aid = "alice"
    app_spec = "@flower/research"
    if field == "flwr_aid":
        flwr_aid = ""
    elif field == "app_spec":
        app_spec = ""
    else:
        values[field] = ""

    assert state.create_agent(flwr_aid, app_spec, AgentConfig(**values)) is None
    assert state.list_agents("alice") == []
    assert state.list_agents("") == []
    assert state.get_spaces("alice") == []


@pytest.fixture(name="agent")
def agent_fixture(state: CoreState) -> Agent:
    """Provide an owned agent with every editable field populated."""
    agent = state.create_agent(
        "alice",
        "@test/app",
        AgentConfig(
            name="Research",
            title="Researcher",
            description="Cite sources.",
            avatar="robot",
            color="sky",
        ),
    )
    assert agent is not None
    return agent


def test_update_agent(state: CoreState, agent: Agent) -> None:
    """Publish current metadata and an event while retaining identity and snapshots."""
    cursor = state.get_bootstrap("alice").cursor
    config = AgentConfig(
        name="Reviewer",
        title="Reviewer",
        description="Review changes.",
        avatar="owl",
        color="violet",
    )
    updated = state.update_agent("alice", agent.id, 1, config)

    assert updated is not None and updated.config_version == 2
    assert updated.config == config
    assert updated.id == agent.id
    assert updated.app_spec == agent.app_spec
    assert updated.default_space_id == agent.default_space_id
    assert state.list_agents("alice") == [updated]
    events = state.get_user_events("alice", cursor)
    assert len(events) == 1 and events[0].event.type == EventType.AGENT_UPDATED
    assert dict(events[0].event.payload) == MessageToDict(
        updated, preserving_proto_field_name=True
    )
    assert not state.get_user_events("bob")
    if isinstance(state, SqlLinkState):
        with state.session() as session:
            original = session.get(AgentConfigModel, (agent.id, 1))
            assert original is not None and original.name == agent.config.name
    else:
        assert isinstance(state, InMemoryLinkState)
        assert state.agent_config_store[(agent.id, 1)] == agent.config
    config = AgentConfig(name="Reviewer", title="", description="")
    cleared = state.update_agent("alice", agent.id, 2, config)
    assert cleared is not None and cleared.config_version == 3
    assert cleared.config == AgentConfig(name=config.name)
    cursor = state.get_bootstrap("alice").cursor
    with pytest.raises(FlowerError) as exc:
        state.update_agent("alice", agent.id, 2, config)
    assert exc.value.code == ApiErrorCode.AGENT_CONFIG_VERSION_CONFLICT
    assert state.get_agent("alice", agent.id) == cleared
    assert not state.get_user_events("alice", cursor)


def test_update_agent_rejects_invalid_writes(state: CoreState, agent: Agent) -> None:
    """Reject foreign agents, missing agents, and blank names."""
    cursor = state.get_bootstrap("alice").cursor
    assert state.update_agent("bob", agent.id, 2, agent.config) is None
    assert state.update_agent("alice", agent.id + 1, 1, agent.config) is None
    with pytest.raises(ValueError):
        state.update_agent("alice", agent.id, 1, AgentConfig(name=" \t"))
    assert state.get_agent("alice", agent.id) == agent
    assert not state.get_user_events("alice", cursor)
    assert not state.get_user_events("bob")


def test_update_agent_rolls_back_when_event_fails(
    state: CoreState, agent: Agent
) -> None:
    """Keep the prior head and discard the new snapshot if the event cannot be saved."""
    cursor = state.get_bootstrap("alice").cursor
    with patch.object(state, "store_event", side_effect=RuntimeError("event failed")):
        with pytest.raises(RuntimeError, match="event failed"):
            state.update_agent("alice", agent.id, 1, agent.config)

    assert state.get_agent("alice", agent.id) == agent
    assert not state.get_user_events("alice", cursor)
    updated = state.update_agent("alice", agent.id, 1, agent.config)
    assert updated is not None and updated.config_version == 2


@pytest.mark.parametrize("state", ["sqlite"], indirect=True)
def test_update_agent_concurrent_writers(state: SqlLinkState, agent: Agent) -> None:
    """Allow exactly one edit per expected version across SQL replicas."""
    replica = SqlLinkState(
        state.database_url, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    replica.initialize()
    barrier = Barrier(2)

    def edit(backend: CoreState) -> bool:
        barrier.wait(timeout=10)
        try:
            assert backend.update_agent("alice", agent.id, 1, agent.config) is not None
            return True
        except FlowerError as exc:
            assert exc.code == ApiErrorCode.AGENT_CONFIG_VERSION_CONFLICT
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(edit, (state, replica), timeout=15)) == [False, True]
