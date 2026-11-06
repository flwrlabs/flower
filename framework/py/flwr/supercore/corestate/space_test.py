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
"""Space persistence tests."""

from uuid import UUID, uuid4

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


@pytest.fixture(name="state", params=["memory", "sqlite"])
def space_state(request: pytest.FixtureRequest) -> CoreState:
    """Provide in-memory and SQL state."""
    manager = NoOpFederationManager()
    object_store = ObjectStoreFactory().store()
    if request.param == "memory":
        return InMemoryLinkState(manager, object_store)
    state = SqlLinkState(":memory:", manager, object_store)
    state.initialize()
    return state


def test_create_and_list_spaces(state: CoreState) -> None:
    """Store explicit participants, deduplicate them, and filter listings."""
    agent = state.create_agent(
        flwr_aid="alice",
        config=AgentConfig(name="Research", app_spec="@flower/research"),
    )
    assert agent is not None
    personal = state.get_spaces("alice")[0]
    alice = SpaceParticipant(flwr_aid="alice")
    bob = SpaceParticipant(flwr_aid="bob")
    agent_participant = SpaceParticipant(agent_id=agent.id)
    first = state.create_space(participants=[bob, agent_participant, alice, bob])
    second = state.create_space(participants=[alice])
    empty = state.create_space(participants=[])
    assert not empty.participants
    assert first.title == ""
    assert len(first.participants) == 3
    assert all(
        member in first.participants for member in [alice, bob, agent_participant]
    )
    assert [space.id for space in state.get_spaces("alice")] == [
        second.id,
        first.id,
        personal.id,
    ]
    assert [space.id for space in state.get_spaces("bob")] == [first.id]
    assert not state.get_spaces(str(agent.id))
    assert all(
        member in state.get_spaces("bob")[0].participants
        for member in [alice, bob, agent_participant]
    )


def test_participation_controls_message_access(state: CoreState) -> None:
    """Only initial participants can read and publish space messages."""
    bob = SpaceParticipant(flwr_aid="bob")
    space = state.create_space(participants=[SpaceParticipant(flwr_aid="alice"), bob])
    space_id = UUID(space.id)
    outsider = SpaceParticipant(flwr_aid="charlie")
    assert state.get_space_messages(space_id, "charlie") is None
    assert state.create_space_message(space_id, outsider, "No access") is None
    message = state.create_space_message(space_id, bob, "Hello")
    assert message is not None and message.author == bob
    assert state.get_space_messages(space_id, "bob") == [message]
    assert state.get_space_messages(space_id, "alice") == [message]
    assert state.get_space_messages(space_id, "charlie") is None


def test_message_history_and_pagination(state: CoreState) -> None:
    """Keep messages ordered, preserve text, and isolate space history."""
    author = SpaceParticipant(flwr_aid="alice")
    space_id = UUID(state.create_space(participants=[author]).id)
    other_id = UUID(state.create_space(participants=[author]).id)
    first = state.create_space_message(space_id, author, "Hello, €100.\nNext.")
    other = state.create_space_message(other_id, author, "Other space")
    second = state.create_space_message(space_id, author, "Second")
    third = state.create_space_message(space_id, author, "Third")
    assert (
        first is not None
        and second is not None
        and third is not None
        and other is not None
    )
    assert first.content == "Hello, €100.\nNext."
    assert state.get_space_messages(space_id, "alice", limit=2) == [
        second,
        third,
    ]
    assert state.get_space_messages(space_id, "alice", before_id=second.id) == [first]
    assert state.get_space_messages(space_id, "alice", before_id=first.id) == []
    assert state.get_space_messages(other_id, "alice") == [other]
    assert state.get_space_messages(space_id, "alice", after_id=first.id, limit=1) == [
        second
    ]
    assert state.get_space_messages(
        space_id, "alice", after_id=first.id, before_id=third.id
    ) == [second]
    assert state.get_space_messages(space_id, "alice", after_id=0, limit=1) == [first]
    assert state.get_space_messages(space_id, "alice", after_id=2**64 - 1) == []


@pytest.mark.parametrize(
    "participant", [SpaceParticipant(), SpaceParticipant(agent_id=9999)]
)
def test_invalid_initial_participants(
    state: CoreState, participant: SpaceParticipant
) -> None:
    """Reject invalid membership without creating a partial space."""
    with pytest.raises(ValueError):
        state.create_space(
            participants=[SpaceParticipant(flwr_aid="bob"), participant],
        )
    assert not state.get_spaces("alice")
    assert not state.get_spaces("bob")


def test_empty_message(state: CoreState) -> None:
    """Reject blank content without appending history."""
    space_id = UUID(
        state.create_space(participants=[SpaceParticipant(flwr_aid="alice")]).id
    )
    with pytest.raises(ValueError, match="content"):
        state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), " \n")
    assert state.get_space_messages(space_id, "alice") == []


def test_unknown_space(state: CoreState) -> None:
    """Missing spaces cannot accumulate messages."""
    space_id = uuid4()
    author = SpaceParticipant(flwr_aid="alice")
    assert state.create_space_message(space_id, author, "Hello") is None
    assert state.get_space_messages(space_id, "alice") is None


@pytest.mark.parametrize("before_id,after_id", [(1, 1), (1, 2)])
def test_invalid_message_range(state: CoreState, before_id: int, after_id: int) -> None:
    """Reject equal or reversed exclusive message bounds."""
    space_id = UUID(
        state.create_space(participants=[SpaceParticipant(flwr_aid="alice")]).id
    )
    with pytest.raises(ValueError, match="after_id"):
        state.get_space_messages(
            space_id, "alice", before_id=before_id, after_id=after_id
        )
