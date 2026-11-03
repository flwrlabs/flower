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

from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    Conversation,
    ConversationParticipant,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


@pytest.fixture(name="state", params=["memory", "sqlite", "sqlite-file"])
def conversation_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise both implementations and migrated on-disk databases."""
    object_store = ObjectStoreFactory().store()
    manager = NoOpFederationManager()
    if request.param == "memory":
        return InMemoryLinkState(manager, object_store)
    database_path = (
        ":memory:" if request.param == "sqlite" else str(tmp_path / "state.db")
    )
    state = SqlLinkState(database_path, manager, object_store)
    state.initialize()
    return state


def test_create_conversation(state: LinkState) -> None:
    """Create UUID metadata and creator participation without starting work."""
    conversation = state.create_conversation(flwr_aid="alice")
    assert str(UUID(conversation.id)) == conversation.id
    assert conversation.title == ""
    assert list(conversation.participants) == [
        ConversationParticipant(flwr_aid="alice")
    ]
    assert list(state.get_conversations("alice")) == [conversation]
    assert Conversation.FromString(conversation.SerializeToString()) == conversation
    assert state.get_conversation_messages(UUID(conversation.id), "alice") == []
    assert not state.get_run_series()
    assert not state.get_tasks()


def test_list_participating_conversations(state: LinkState) -> None:
    """List only participating conversations, newest first."""
    first = state.create_conversation(flwr_aid="alice")
    second = state.create_conversation(flwr_aid="alice")
    state.create_conversation(flwr_aid="bob")
    assert list(state.get_conversations("alice")) == [second, first]
    assert not state.get_conversations("unknown")


def test_participants_control_visibility(state: LinkState) -> None:
    """Membership grants visibility independently of the creator."""
    conversation = state.create_conversation(flwr_aid="alice")
    conversation_id = UUID(conversation.id)
    alice = ConversationParticipant(flwr_aid="alice")
    bob = ConversationParticipant(flwr_aid="bob")
    assert not state.get_conversations("bob")
    assert state.add_conversation_participant(conversation_id, bob)
    assert state.add_conversation_participant(conversation_id, bob)
    shared = state.get_conversations("bob")
    assert len(shared) == 1
    assert len(shared[0].participants) == 2
    assert alice in shared[0].participants and bob in shared[0].participants
    assert state.remove_conversation_participant(conversation_id, alice)
    assert not state.get_conversations("alice")
    assert list(state.get_conversations("bob")[0].participants) == [bob]
    assert not state.remove_conversation_participant(conversation_id, alice)
    assert state.remove_conversation_participant(conversation_id, bob)
    assert not state.get_conversations("bob")


def test_agent_and_user_participants_are_distinct(state: LinkState) -> None:
    """An agent identity cannot grant access to a similarly named user."""
    agent = state.create_agent(
        flwr_aid="alice",
        config=AgentConfig(name="Research", app_spec="@flower/research"),
    )
    assert agent is not None
    conversation_id = UUID(state.create_conversation(flwr_aid="alice").id)
    agent_author = ConversationParticipant(agent_id=agent.id)
    user_author = ConversationParticipant(flwr_aid=str(agent.id))
    assert state.add_conversation_participant(conversation_id, agent_author)
    assert state.add_conversation_participant(conversation_id, agent_author)
    assert state.get_conversation_messages(conversation_id, str(agent.id)) is None
    reply = state.create_conversation_message(
        conversation_id, agent_author, "Agent reply"
    )
    assert reply is not None and reply.author == agent_author
    assert state.add_conversation_participant(conversation_id, user_author)
    assert state.get_conversation_messages(conversation_id, str(agent.id)) == [reply]
    assert state.remove_conversation_participant(conversation_id, agent_author)
    assert (
        state.create_conversation_message(conversation_id, agent_author, "No") is None
    )
    assert state.get_conversation_messages(conversation_id, "alice") == [reply]


@pytest.mark.parametrize(
    "participant",
    [
        ConversationParticipant(),
        ConversationParticipant(flwr_aid=""),
        ConversationParticipant(agent_id=0),
        ConversationParticipant(agent_id=9999),
    ],
)
def test_invalid_participant(
    state: LinkState, participant: ConversationParticipant
) -> None:
    """Reject missing identities and unknown agents without changing membership."""
    conversation_id = UUID(state.create_conversation(flwr_aid="alice").id)
    assert not state.add_conversation_participant(conversation_id, participant)
    assert not state.remove_conversation_participant(conversation_id, participant)
    assert state.create_conversation_message(conversation_id, participant, "No") is None
    assert list(state.get_conversations("alice")[0].participants) == [
        ConversationParticipant(flwr_aid="alice")
    ]


def test_unknown_conversation(state: LinkState) -> None:
    """Missing conversations cannot accumulate messages or participants."""
    conversation_id = uuid4()
    author = ConversationParticipant(flwr_aid="alice")
    assert not state.add_conversation_participant(conversation_id, author)
    assert not state.remove_conversation_participant(conversation_id, author)
    assert state.create_conversation_message(conversation_id, author, "No") is None
    assert state.get_conversation_messages(conversation_id, "alice") is None


def test_message_history_and_pagination(state: LinkState) -> None:
    """Preserve text and chronological pages while isolating conversations."""
    conversation_id = UUID(state.create_conversation(flwr_aid="alice").id)
    other_id = UUID(state.create_conversation(flwr_aid="alice").id)
    author = ConversationParticipant(flwr_aid="alice")
    first = state.create_conversation_message(
        conversation_id, author, "Hello, €100.\n\nNext."
    )
    other = state.create_conversation_message(other_id, author, "Other conversation")
    second = state.create_conversation_message(conversation_id, author, "Second")
    third = state.create_conversation_message(conversation_id, author, "Third")
    assert (
        first is not None
        and other is not None
        and second is not None
        and third is not None
    )
    assert first.id < other.id < second.id < third.id
    assert first.content == "Hello, €100.\n\nNext."
    assert first.author == author and first.conversation_id == str(conversation_id)
    assert first.created_at
    assert state.get_conversation_messages(conversation_id, "alice") == [
        first,
        second,
        third,
    ]
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
    assert state.get_conversation_messages(
        conversation_id, "alice", before_id=2**64 - 1
    ) == [first, second, third]
    assert state.get_conversation_messages(other_id, "alice") == [other]
    assert not state.get_run_series() and not state.get_tasks()


def test_message_access_tracks_participation(state: LinkState) -> None:
    """Revoking access prevents reads and writes while preserving authored history."""
    conversation_id = UUID(state.create_conversation(flwr_aid="alice").id)
    bob = ConversationParticipant(flwr_aid="bob")
    assert state.get_conversation_messages(conversation_id, "bob") is None
    assert state.create_conversation_message(conversation_id, bob, "No") is None
    assert state.add_conversation_participant(conversation_id, bob)
    message = state.create_conversation_message(conversation_id, bob, "Shared reply")
    assert message is not None
    assert state.remove_conversation_participant(conversation_id, bob)
    assert state.get_conversation_messages(conversation_id, "bob") is None
    assert state.create_conversation_message(conversation_id, bob, "No") is None
    assert state.get_conversation_messages(conversation_id, "alice") == [message]


@pytest.mark.parametrize("content", ["", " ", "\n\t"])
def test_empty_message(state: LinkState, content: str) -> None:
    """Reject empty messages without appending history."""
    conversation_id = UUID(state.create_conversation(flwr_aid="alice").id)
    with pytest.raises(ValueError, match="content"):
        state.create_conversation_message(
            conversation_id, ConversationParticipant(flwr_aid="alice"), content
        )
    assert state.get_conversation_messages(conversation_id, "alice") == []


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_history_limit(state: LinkState, limit: int) -> None:
    """Reject limits that cannot select a bounded history page."""
    conversation_id = UUID(state.create_conversation(flwr_aid="alice").id)
    with pytest.raises(ValueError, match="limit"):
        state.get_conversation_messages(conversation_id, "alice", limit=limit)


def test_sql_creation_collision_preserves_existing_conversation() -> None:
    """A duplicate UUID cannot change an existing conversation's participants."""
    state = SqlLinkState(
        ":memory:", NoOpFederationManager(), ObjectStoreFactory().store()
    )
    state.initialize()
    original = state.create_conversation(flwr_aid="alice")
    with (
        patch(
            "flwr.supercore.corestate.sql_corestate.uuid4",
            return_value=UUID(original.id),
        ),
        pytest.raises(IntegrityError),
    ):
        state.create_conversation(flwr_aid="bob")
    assert state.get_conversations("alice") == [original]
    assert not state.get_conversations("bob")


def test_sql_reload_preserves_conversations_and_messages(tmp_path: Path) -> None:
    """Migrated file storage retains UUIDs, participation, and message history."""
    path = str(tmp_path / "persistent.db")
    state = SqlLinkState(path, NoOpFederationManager(), ObjectStoreFactory().store())
    state.initialize()
    conversation = state.create_conversation(flwr_aid="alice")
    conversation_id = UUID(conversation.id)
    bob = ConversationParticipant(flwr_aid="bob")
    assert state.add_conversation_participant(conversation_id, bob)
    message = state.create_conversation_message(conversation_id, bob, "Persist this")
    assert message is not None
    reloaded = SqlLinkState(path, NoOpFederationManager(), ObjectStoreFactory().store())
    reloaded.initialize()
    assert reloaded.get_conversations("alice") == state.get_conversations("alice")
    assert reloaded.get_conversations("bob") == state.get_conversations("bob")
    assert reloaded.get_conversation_messages(conversation_id, "alice") == [message]
    assert reloaded.get_conversation_messages(conversation_id, "bob") == [message]
