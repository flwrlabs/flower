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

from uuid import UUID

import pytest

from flwr.proto.control_pb2 import AgentConfig  # pylint: disable=E0611
from flwr.proto.space_pb2 import SpaceParticipant  # pylint: disable=E0611
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import SpaceItemType
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


def test_latest_user_message(state: CoreState) -> None:
    """Summarize each space's newest user message item without consuming history."""
    agent = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@flower/research")
    )
    assert agent is not None
    personal = state.get_spaces("alice")[0]
    assert personal.latest_user_message == ""
    alice = SpaceParticipant(flwr_aid="alice")
    bob = SpaceParticipant(flwr_aid="bob")
    author = SpaceParticipant(agent_id=agent.id)
    state.store_space_item(
        UUID(personal.id), SpaceItemType.MESSAGE, "Agent-only space", author
    )
    shared = state.create_space(participants=[alice, bob, author])
    space_id = UUID(shared.id)
    assert state.get_spaces("bob")[0].latest_user_message == ""
    first = state.store_space_item(
        space_id, SpaceItemType.MESSAGE, "First prompt", alice
    )
    previous = state.get_spaces("bob")[0]
    prompt = "  Follow up: €100.\nNext line.  "
    latest = state.store_space_item(space_id, SpaceItemType.MESSAGE, prompt, bob)
    automation = state.store_space_item(space_id, SpaceItemType.AUTOMATION, "Scheduled")
    reply = state.store_space_item(
        space_id, SpaceItemType.MESSAGE, "Agent reply", author
    )
    other = state.create_space(participants=[alice])
    state.store_space_item(UUID(other.id), SpaceItemType.MESSAGE, "Other space", alice)
    assert {
        space.id: space.latest_user_message for space in state.get_spaces("alice")
    } == {personal.id: "", shared.id: prompt, other.id: "Other space"}
    assert previous.latest_user_message == "First prompt"
    assert state.get_space_items(space_id, alice, unread_only=True) == [
        first,
        latest,
        automation,
        reply,
    ]


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
