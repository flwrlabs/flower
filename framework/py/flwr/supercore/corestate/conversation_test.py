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
"""Conversation persistence tests."""

from uuid import UUID, uuid4

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    ConversationParticipant,
)
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from . import CoreState


@pytest.fixture(name="state", params=["memory", "sqlite"])
def conversation_state(request: pytest.FixtureRequest) -> CoreState:
    """Provide in-memory and SQL state."""
    manager = NoOpFederationManager()
    object_store = ObjectStoreFactory().store()
    if request.param == "memory":
        return InMemoryLinkState(manager, object_store)
    state = SqlLinkState(":memory:", manager, object_store)
    state.initialize()
    return state


def test_create_and_list_conversations(state: CoreState) -> None:
    """Store explicit participants, deduplicate them, and filter listings."""
    agent = state.create_agent(
        flwr_aid="alice",
        config=AgentConfig(name="Research", app_spec="@flower/research"),
    )
    assert agent is not None
    alice = ConversationParticipant(flwr_aid="alice")
    bob = ConversationParticipant(flwr_aid="bob")
    agent_participant = ConversationParticipant(agent_id=agent.id)
    first = state.create_conversation(participants=[bob, agent_participant, alice, bob])
    second = state.create_conversation(participants=[alice])
    empty = state.create_conversation(participants=[])
    assert not empty.participants
    assert first.title == ""
    assert len(first.participants) == 3
    assert all(
        member in first.participants for member in [alice, bob, agent_participant]
    )
    assert [conversation.id for conversation in state.get_conversations("alice")] == [
        second.id,
        first.id,
    ]
    assert [conversation.id for conversation in state.get_conversations("bob")] == [
        first.id
    ]
    assert not state.get_conversations(str(agent.id))
    assert all(
        member in state.get_conversations("bob")[0].participants
        for member in [alice, bob, agent_participant]
    )


def test_participation_controls_message_access(state: CoreState) -> None:
    """Only initial participants can read and publish conversation messages."""
    bob = ConversationParticipant(flwr_aid="bob")
    conversation = state.create_conversation(
        participants=[ConversationParticipant(flwr_aid="alice"), bob]
    )
    conversation_id = UUID(conversation.id)
    outsider = ConversationParticipant(flwr_aid="charlie")
    assert state.get_conversation_messages(conversation_id, "charlie") is None
    assert (
        state.create_conversation_message(conversation_id, outsider, "No access")
        is None
    )
    message = state.create_conversation_message(conversation_id, bob, "Hello")
    assert message is not None and message.author == bob
    assert state.get_conversation_messages(conversation_id, "bob") == [message]
    assert state.get_conversation_messages(conversation_id, "alice") == [message]
    assert state.get_conversation_messages(conversation_id, "charlie") is None


def test_message_history_and_pagination(state: CoreState) -> None:
    """Keep messages ordered, preserve text, and isolate conversation history."""
    author = ConversationParticipant(flwr_aid="alice")
    conversation_id = UUID(state.create_conversation(participants=[author]).id)
    other_id = UUID(state.create_conversation(participants=[author]).id)
    first = state.create_conversation_message(
        conversation_id, author, "Hello, €100.\nNext."
    )
    other = state.create_conversation_message(other_id, author, "Other conversation")
    second = state.create_conversation_message(conversation_id, author, "Second")
    third = state.create_conversation_message(conversation_id, author, "Third")
    assert (
        first is not None
        and second is not None
        and third is not None
        and other is not None
    )
    assert first.content == "Hello, €100.\nNext."
    assert state.get_conversation_messages(conversation_id, "alice", limit=2) == [
        second,
        third,
    ]
    assert state.get_conversation_messages(
        conversation_id, "alice", before_id=second.id
    ) == [first]
    assert (
        state.get_conversation_messages(conversation_id, "alice", before_id=first.id)
        == []
    )
    assert state.get_conversation_messages(other_id, "alice") == [other]
    assert state.get_conversation_messages(
        conversation_id, "alice", after_id=first.id, limit=1
    ) == [second]
    assert state.get_conversation_messages(
        conversation_id, "alice", after_id=first.id, before_id=third.id
    ) == [second]
    assert state.get_conversation_messages(
        conversation_id, "alice", after_id=0, limit=1
    ) == [first]
    assert (
        state.get_conversation_messages(conversation_id, "alice", after_id=2**64 - 1)
        == []
    )


@pytest.mark.parametrize(
    "participant", [ConversationParticipant(), ConversationParticipant(agent_id=9999)]
)
def test_invalid_initial_participants(
    state: CoreState, participant: ConversationParticipant
) -> None:
    """Reject invalid membership without creating a partial conversation."""
    with pytest.raises(ValueError):
        state.create_conversation(
            participants=[ConversationParticipant(flwr_aid="bob"), participant],
        )
    assert not state.get_conversations("alice")
    assert not state.get_conversations("bob")


def test_empty_message(state: CoreState) -> None:
    """Reject blank content without appending history."""
    conversation_id = UUID(
        state.create_conversation(
            participants=[ConversationParticipant(flwr_aid="alice")]
        ).id
    )
    with pytest.raises(ValueError, match="content"):
        state.create_conversation_message(
            conversation_id, ConversationParticipant(flwr_aid="alice"), " \n"
        )
    assert state.get_conversation_messages(conversation_id, "alice") == []


def test_unknown_conversation(state: CoreState) -> None:
    """Missing conversations cannot accumulate messages."""
    conversation_id = uuid4()
    author = ConversationParticipant(flwr_aid="alice")
    assert state.create_conversation_message(conversation_id, author, "Hello") is None
    assert state.get_conversation_messages(conversation_id, "alice") is None


@pytest.mark.parametrize("before_id,after_id", [(1, 1), (1, 2)])
def test_invalid_message_range(state: CoreState, before_id: int, after_id: int) -> None:
    """Reject equal or reversed exclusive message bounds."""
    conversation_id = UUID(
        state.create_conversation(
            participants=[ConversationParticipant(flwr_aid="alice")]
        ).id
    )
    with pytest.raises(ValueError, match="after_id"):
        state.get_conversation_messages(
            conversation_id, "alice", before_id=before_id, after_id=after_id
        )
