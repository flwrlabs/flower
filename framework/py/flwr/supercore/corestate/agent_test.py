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

from hashlib import sha256
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Agent,
    AgentConfig,
    StartRunRequest,
)
from flwr.proto.space_pb2 import SpaceParticipant  # pylint: disable=E0611
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import AutomationStatus, EventType, TaskType
from flwr.supercore.date import now
from flwr.supercore.fab import Fab
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.state.schema.corestate_models import Agent as AgentModel
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
        app_spec="@flower/research==1.2.3",
        title="Researcher",
        description="Investigate and cite sources.",
        avatar="robot",
        color="sky",
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
    assert state.get_agent("alice", first.id) == first
    assert state.get_agent("bob", first.id) is None
    assert state.get_agent("alice", other.id) is None
    assert state.list_agents("unknown") == []
    assert state.list_agents("") == []


def test_create_agent_creates_space(state: CoreState) -> None:
    """Each new agent gets an empty Space containing only itself and its creator."""
    config = AgentConfig(name="Research", app_spec="@flower/research")
    agents = [state.create_agent("alice", config) for _ in range(2)]
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
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@test/app"))
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
    assert state.list_agents("alice") == []
    assert state.list_agents("") == []
    assert state.get_spaces("alice") == []


@pytest.fixture(name="agent")
def agent_fixture(state: CoreState) -> Agent:
    """Provide an owned agent for deletion tests."""
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@test/app"))
    assert agent is not None
    return agent


def test_delete_agent(state: CoreState, agent: Agent) -> None:
    """Hide active data while preserving history and a single deletion event."""
    group = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(agent_id=agent.id),
        ]
    )
    message = state.create_space_message(
        UUID(group.id), SpaceParticipant(agent_id=agent.id), "History"
    )
    scope = f"agent-{agent.id}"
    state.put_file(scope, "memory", b"Retain", None)
    cursor = state.get_bootstrap("alice").cursor
    assert not state.delete_agent("bob", agent.id)
    assert not state.delete_agent("alice", agent.id + 1)
    assert state.delete_agent("alice", agent.id)
    assert state.get_agent("alice", agent.id) is None
    assert state.get_default_space("alice", agent.id) is None
    assert not state.get_bootstrap("alice").agents
    assert [space.id for space in state.get_spaces("alice")] == [group.id]
    assert state.get_agent("bob", agent.id) is None
    assert state.get_space_messages(
        UUID(group.id), SpaceParticipant(flwr_aid="alice")
    ) == [message]
    stored_file = state.get_file(scope, "memory")
    assert stored_file is not None and stored_file.data == b"Retain"
    events = state.get_user_events("alice", cursor)
    assert len(events) == 1 and events[0].event.type == EventType.AGENT_DELETED
    assert events[0].event.payload["id"] == str(agent.id)
    deleted_at = events[0].event.payload["updated_at"]
    assert state.delete_agent("alice", agent.id)
    assert state.get_user_events("alice", cursor) == events
    assert not state.get_user_events("bob")
    if isinstance(state, SqlLinkState):
        with state.session() as session:
            model = session.get(AgentModel, agent.id)
            assert model is not None and model.deleted_at is not None
            assert model.deleted_at.isoformat() == deleted_at
            assert session.get(AgentConfigModel, (agent.id, 1)) is not None
    else:
        assert isinstance(state, InMemoryLinkState)
        assert state.agent_store["alice"][agent.id].config == agent.config
        assert state.deleted_agent_store[agent.id].isoformat() == deleted_at


def test_delete_agent_blocks_future_activity(state: CoreState, agent: Agent) -> None:
    """Stop schedules and reject new runs and memberships for an archived agent."""
    assert isinstance(state, (InMemoryLinkState, SqlLinkState))
    automation = state.store_automation(
        flwr_aid="alice",
        agent_id=agent.id,
        start_run_request=StartRunRequest(),
        next_run_at=now().isoformat(),
        fixed_interval=60,
    )
    space_id = UUID(agent.default_space_id)
    state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), "Work")
    fab_hash = state.store_fab(Fab(sha256(b"app").hexdigest(), b"app", {}))
    assert state.get_agent_run_candidates()
    assert state.delete_agent("alice", agent.id)
    assert not state.get_agent_run_candidates()
    assert state.create_connector(None, "slack", "{}", "{}", "alice", agent.id) is None
    assert (
        state.create_run(
            "test/app",
            "1.0",
            fab_hash,
            {},
            None,
            None,
            None,
            TaskType.AGENT_APP,
            space_id=space_id,
            agent_id=agent.id,
        )
        == 0
    )
    schedules = state.list_automations(
        automation_ids=[automation.automation_id], order_by="updated_at"
    )
    assert schedules[0].status == AutomationStatus.STOPPED
    assert (
        state.claim_automation(
            automation.automation_id,
            previous_next_run_at=automation.next_run_at,
            next_run_at=None,
        )
        is None
    )
    with pytest.raises(ValueError, match="Agent participant"):
        state.create_space(participants=[SpaceParticipant(agent_id=agent.id)])
    with pytest.raises(ValueError, match="Agent not found"):
        state.store_automation(
            flwr_aid="alice",
            agent_id=agent.id,
            start_run_request=StartRunRequest(),
            next_run_at=now().isoformat(),
        )


def test_delete_agent_rolls_back(state: CoreState, agent: Agent) -> None:
    """Preserve the active agent and schedule when the deletion event fails."""
    automation = state.store_automation(
        flwr_aid="alice",
        agent_id=agent.id,
        start_run_request=StartRunRequest(),
        next_run_at=now().isoformat(),
    )
    cursor = state.get_bootstrap("alice").cursor
    with patch.object(state, "store_event", side_effect=RuntimeError("event failed")):
        with pytest.raises(RuntimeError, match="event failed"):
            state.delete_agent("alice", agent.id)
    assert state.get_agent("alice", agent.id) == agent
    assert not state.get_user_events("alice", cursor)
    schedules = state.list_automations(
        automation_ids=[automation.automation_id], order_by="updated_at"
    )
    assert schedules[0].status == AutomationStatus.ACTIVE
    assert state.delete_agent("alice", agent.id)
