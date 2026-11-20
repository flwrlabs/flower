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
from typing import Literal
from unittest.mock import patch
from uuid import UUID

import pytest
from google.protobuf.json_format import MessageToDict
from sqlalchemy import select

from flwr.common.constant import INT64_MAX_VALUE
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


def test_update_agent_preserves_snapshots_and_publishes_event(state: CoreState) -> None:
    """Advance metadata and its feed together without changing identity or history."""
    original = state.create_agent(
        "alice",
        AgentConfig(
            name="Research",
            app_spec="@flower/research",
            title="Researcher",
            description="Cite sources.",
            avatar="robot",
            color="sky",
        ),
    )
    assert original is not None
    cursor = state.get_bootstrap("alice").cursor

    updated = state.update_agent(
        "alice",
        original.id,
        original.config_version,
        AgentConfig(
            name="Reviewer",
            app_spec=original.config.app_spec,
            title="Reviewer",
            description="Review changes.",
            avatar="owl",
            color="violet",
        ),
    )

    assert updated is not None
    assert updated.config_version == 2
    assert updated.config == AgentConfig(
        name="Reviewer",
        app_spec=original.config.app_spec,
        title="Reviewer",
        description="Review changes.",
        avatar="owl",
        color="violet",
    )
    assert updated.id == original.id
    assert updated.default_space_id == original.default_space_id
    assert updated.created_at == original.created_at
    assert updated.updated_at > original.updated_at
    assert original.config.name == "Research" and original.config_version == 1
    assert state.get_agent("alice", original.id) == updated
    assert state.list_agents("alice") == [updated]
    assert state.get_bootstrap("alice").agents == [updated]
    events = state.get_user_events("alice", cursor)
    assert len(events) == 1
    assert events[0].event.type == EventType.AGENT_UPDATED
    assert dict(events[0].event.payload) == MessageToDict(
        updated, preserving_proto_field_name=True
    )
    assert not state.get_user_events("bob")

    if isinstance(state, SqlLinkState):
        with state.session() as session:
            snapshots = session.scalars(
                select(AgentConfigModel)
                .where(AgentConfigModel.agent_id == original.id)
                .order_by(AgentConfigModel.version)
            ).all()
            assert [(s.version, s.name, s.description) for s in snapshots] == [
                (1, "Research", "Cite sources."),
                (2, "Reviewer", "Review changes."),
            ]
    else:
        assert isinstance(state, InMemoryLinkState)
        assert state.agent_config_store[(original.id, 1)] == original.config
        assert state.agent_config_store[(original.id, 2)] == updated.config

    updated.config.name = "Unrecorded edit"
    updated.config_version = 100
    latest = state.get_agent("alice", original.id)
    assert latest is not None
    assert latest.config.name == "Reviewer" and latest.config_version == 2


@pytest.mark.parametrize("field", ["title", "description", "avatar", "color"])
@pytest.mark.parametrize("clear_value", [None, ""])
def test_update_agent_clears_optional_fields(
    state: CoreState,
    field: Literal["title", "description", "avatar", "color"],
    clear_value: str | None,
) -> None:
    """Clear absent or empty optional fields and allow setting them again."""
    original = state.create_agent(
        "alice",
        AgentConfig(
            name="Research",
            app_spec="@flower/research",
            title="Researcher",
            description="Cite sources.",
            avatar="robot",
            color="sky",
        ),
    )
    assert original is not None
    config = AgentConfig()
    config.CopyFrom(original.config)
    if clear_value is None:
        config.ClearField(field)
    else:
        setattr(config, field, clear_value)

    updated = state.update_agent("alice", original.id, 1, config)

    assert updated is not None and updated.config_version == 2
    expected = AgentConfig()
    expected.CopyFrom(original.config)
    expected.ClearField(field)
    assert updated.config == expected
    restored = state.update_agent("alice", original.id, 2, original.config)
    assert restored is not None and restored.config_version == 3
    assert restored.config == original.config


@pytest.mark.parametrize("expected_version", [1, 3])
def test_update_agent_rejects_version_conflicts(
    state: CoreState, expected_version: int
) -> None:
    """Reject stale and future versions without changing metadata or events."""
    original = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@test/app")
    )
    assert original is not None
    updated = state.update_agent(
        "alice", original.id, 1, AgentConfig(name="Updated", app_spec="@test/app")
    )
    cursor = state.get_bootstrap("alice").cursor

    with pytest.raises(FlowerError) as exc:
        state.update_agent(
            "alice",
            original.id,
            expected_version,
            AgentConfig(name="Lost update", app_spec="@test/app"),
        )

    assert exc.value.code == ApiErrorCode.AGENT_CONFIG_VERSION_CONFLICT
    assert state.get_agent("alice", original.id) == updated
    assert not state.get_user_events("alice", cursor)


def test_update_agent_is_owner_scoped(state: CoreState) -> None:
    """Treat another account's agents as absent, even with a mismatched version."""
    original = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@test/app")
    )
    assert original is not None
    cursor = state.get_bootstrap("alice").cursor

    assert (
        state.update_agent(
            "bob", original.id, 2, AgentConfig(name="Edit", app_spec="@test/app")
        )
        is None
    )
    assert (
        state.update_agent(
            "alice", original.id + 1, 1, AgentConfig(name="Edit", app_spec="@test/app")
        )
        is None
    )
    assert state.get_agent("alice", original.id) == original
    assert not state.get_user_events("alice", cursor)
    assert not state.get_user_events("bob")


@pytest.mark.parametrize(
    ("config", "version"),
    [
        (AgentConfig(), 1),
        (AgentConfig(name="", app_spec="@test/app"), 1),
        (AgentConfig(name=" \t", app_spec="@test/app"), 1),
        (AgentConfig(name="Updated"), 1),
        (AgentConfig(name="Updated", app_spec=" \t"), 1),
        (AgentConfig(name="Updated", app_spec="@test/app"), 0),
        (AgentConfig(name="Updated", app_spec="@test/app"), -1),
        (AgentConfig(name="Updated", app_spec="@test/app"), INT64_MAX_VALUE),
        (AgentConfig(name="Updated", app_spec="@test/app"), 2**64 - 1),
    ],
)
def test_update_agent_rejects_invalid_config(
    state: CoreState, config: AgentConfig, version: int
) -> None:
    """Reject invalid configurations and versions before changing any state."""
    original = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@test/app")
    )
    assert original is not None
    cursor = state.get_bootstrap("alice").cursor

    with pytest.raises(ValueError):
        state.update_agent("alice", original.id, version, config)

    assert state.get_agent("alice", original.id) == original
    assert not state.get_user_events("alice", cursor)


def test_update_agent_rejects_app_spec_changes(state: CoreState) -> None:
    """Keep the fixed AgentApp and configuration version when a replacement differs."""
    original = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@test/app")
    )
    assert original is not None
    cursor = state.get_bootstrap("alice").cursor

    with pytest.raises(ValueError, match="cannot be changed"):
        state.update_agent(
            "alice", original.id, 1, AgentConfig(name="Updated", app_spec="@other/app")
        )

    assert state.get_agent("alice", original.id) == original
    assert not state.get_user_events("alice", cursor)
    updated = state.update_agent("alice", original.id, 1, original.config)
    assert updated is not None and updated.config_version == 2


def test_update_agent_rolls_back_when_event_fails(state: CoreState) -> None:
    """Keep the prior head and discard the new snapshot if the event cannot be saved."""
    original = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@test/app")
    )
    assert original is not None
    cursor = state.get_bootstrap("alice").cursor
    config = AgentConfig(name="Updated", app_spec="@test/app")

    with patch.object(state, "store_event", side_effect=RuntimeError("event failed")):
        with pytest.raises(RuntimeError, match="event failed"):
            state.update_agent("alice", original.id, 1, config)

    assert state.get_agent("alice", original.id) == original
    assert not state.get_user_events("alice", cursor)
    updated = state.update_agent("alice", original.id, 1, config)
    assert updated is not None and updated.config_version == 2


@pytest.mark.parametrize("backend", ["in-memory", "sqlite"])
def test_update_agent_concurrent_writers(backend: str, tmp_path: Path) -> None:
    """Allow exactly one edit per expected version, including across SQL replicas."""
    if backend == "in-memory":
        first: CoreState = InMemoryLinkState(
            NoOpFederationManager(), ObjectStoreFactory().store()
        )
        second = first
    else:
        replicas = [
            SqlLinkState(
                str(tmp_path / "state.db"),
                NoOpFederationManager(),
                ObjectStoreFactory().store(),
            )
            for _ in range(2)
        ]
        for replica in replicas:
            replica.initialize()
        first, second = replicas
    original = first.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@test/app")
    )
    assert original is not None
    cursor = first.get_bootstrap("alice").cursor
    barrier = Barrier(2)

    def edit(state: CoreState, name: str) -> Agent | FlowerError | None:
        barrier.wait(timeout=10)
        try:
            return state.update_agent(
                "alice", original.id, 1, AgentConfig(name=name, app_spec="@test/app")
            )
        except FlowerError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(edit, first, "First"),
            executor.submit(edit, second, "Second"),
        ]
        results = [future.result(timeout=15) for future in futures]

    winners = [result for result in results if isinstance(result, Agent)]
    conflicts = [result for result in results if isinstance(result, FlowerError)]
    assert len(winners) == len(conflicts) == 1
    assert conflicts[0].code == ApiErrorCode.AGENT_CONFIG_VERSION_CONFLICT
    assert winners[0].config_version == 2
    assert (
        first.get_agent("alice", original.id)
        == second.get_agent("alice", original.id)
        == winners[0]
    )
    assert len(first.get_user_events("alice", cursor)) == 1
