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

import pytest
from sqlalchemy.exc import IntegrityError

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Conversation,
    ConversationParticipant,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.date import now
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.state.schema.corestate_models import Agent as AgentModel
from flwr.supercore.utils import uint64_to_int64
from flwr.superlink.federation import NoOpFederationManager


@pytest.fixture(name="state", params=["memory", "sqlite", "sqlite-file"])
def conversation_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise both implementations and migrated on-disk databases."""
    object_store = ObjectStoreFactory().store()
    federation_manager = NoOpFederationManager()
    if request.param == "memory":
        return InMemoryLinkState(federation_manager, object_store)
    database_path = (
        ":memory:" if request.param == "sqlite" else str(tmp_path / "state.db")
    )
    sql_state = SqlLinkState(database_path, federation_manager, object_store)
    sql_state.initialize()
    return sql_state


def test_create_conversation(state: LinkState) -> None:
    """Create metadata without creating a run series or starting work."""
    conversation = state.create_conversation(flwr_aid="alice")
    assert conversation.title == ""
    assert list(conversation.participants) == [
        ConversationParticipant(flwr_aid="alice")
    ]
    assert not state.get_run_series()
    assert not state.get_tasks()
    assert list(state.get_conversations("alice")) == [conversation]
    assert Conversation.FromString(conversation.SerializeToString()) == conversation


def test_participation_filters_conversations(state: LinkState) -> None:
    """List only participating conversations, newest first."""
    first = state.create_conversation(flwr_aid="alice")
    second = state.create_conversation(flwr_aid="alice")
    state.create_conversation(flwr_aid="bob")

    assert list(state.get_conversations("alice")) == [second, first]
    assert not state.get_conversations("unknown")


def test_participants_control_visibility(state: LinkState) -> None:
    """Membership grants visibility independently of who created the conversation."""
    conversation = state.create_conversation(flwr_aid="alice")
    conversation_id = conversation.conversation_id
    alice = ConversationParticipant(flwr_aid="alice")
    bob = ConversationParticipant(flwr_aid="bob")

    assert not state.get_conversations("bob")
    assert state.add_conversation_participant(conversation_id, bob)
    assert state.add_conversation_participant(conversation_id, bob)
    shared = state.get_conversations("bob")
    assert len(shared) == 1
    assert list(shared[0].participants) == [alice, bob]

    assert state.remove_conversation_participant(conversation_id, alice)
    assert not state.get_conversations("alice")
    assert list(state.get_conversations("bob")[0].participants) == [bob]
    assert not state.remove_conversation_participant(conversation_id, alice)
    assert state.remove_conversation_participant(conversation_id, bob)
    assert not state.get_conversations("bob")


def test_agent_and_user_participants_are_distinct(state: LinkState) -> None:
    """An agent ID never grants access to a user with the same textual ID."""
    if isinstance(state, SqlLinkState):
        with state.session() as session:
            session.add(
                AgentModel(
                    agent_id=7,
                    name="Research",
                    app_spec="@flower/research",
                    flwr_aid="alice",
                    created_at=now(),
                    updated_at=now(),
                )
            )
    conversation = state.create_conversation(flwr_aid="alice")
    agent = ConversationParticipant(agent_id=7)
    user = ConversationParticipant(flwr_aid="7")

    assert state.add_conversation_participant(conversation.conversation_id, agent)
    assert state.add_conversation_participant(conversation.conversation_id, agent)
    assert not state.get_conversations("7")
    assert state.add_conversation_participant(conversation.conversation_id, user)
    assert list(state.get_conversations("7")[0].participants) == [
        ConversationParticipant(flwr_aid="alice"),
        agent,
        user,
    ]
    assert state.remove_conversation_participant(conversation.conversation_id, agent)
    assert len(state.get_conversations("7")[0].participants) == 2
    assert not state.get_run_series()
    assert not state.get_tasks()


@pytest.mark.parametrize(
    "participant",
    [
        ConversationParticipant(),
        ConversationParticipant(flwr_aid=""),
        ConversationParticipant(agent_id=0),
    ],
)
def test_invalid_participant_does_not_change_membership(
    state: LinkState, participant: ConversationParticipant
) -> None:
    """Reject missing or empty participant identities without changing visibility."""
    conversation = state.create_conversation(flwr_aid="alice")
    assert not state.add_conversation_participant(
        conversation.conversation_id, participant
    )
    assert not state.remove_conversation_participant(
        conversation.conversation_id, participant
    )
    assert list(state.get_conversations("alice")[0].participants) == [
        ConversationParticipant(flwr_aid="alice")
    ]


def test_unknown_conversation_cannot_gain_participants(state: LinkState) -> None:
    """Missing conversations cannot accumulate orphan participants."""
    participant = ConversationParticipant(flwr_aid="bob")
    assert not state.add_conversation_participant(42, participant)
    assert not state.remove_conversation_participant(42, participant)
    assert not state.get_conversations("bob")


def test_sql_creation_collision_preserves_existing_conversation() -> None:
    """A duplicate conversation ID cannot alter existing metadata or participants."""
    sql_state = SqlLinkState(
        ":memory:", NoOpFederationManager(), ObjectStoreFactory().store()
    )
    sql_state.initialize()
    original = sql_state.create_conversation(flwr_aid="alice")
    with (
        patch(
            "flwr.supercore.corestate.sql_corestate.generate_rand_int_from_bytes",
            return_value=original.conversation_id,
        ),
        pytest.raises(IntegrityError),
    ):
        sql_state.create_conversation(flwr_aid="bob")
    assert list(sql_state.get_conversations("alice")) == [original]
    assert not sql_state.get_conversations("bob")


def test_sql_reload_preserves_unsigned_ids(tmp_path: Path) -> None:
    """Migrated file storage retains full uint64 IDs and participants."""
    database_path = str(tmp_path / "persistent.db")
    sql_state = SqlLinkState(
        database_path, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    sql_state.initialize()
    with patch(
        "flwr.supercore.corestate.sql_corestate.generate_rand_int_from_bytes",
        return_value=2**63 + 42,
    ):
        conversation = sql_state.create_conversation(flwr_aid="alice")
    assert sql_state.add_conversation_participant(
        conversation.conversation_id, ConversationParticipant(flwr_aid="bob")
    )
    expected = sql_state.get_conversations("bob")[0]
    reloaded = SqlLinkState(
        database_path, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    reloaded.initialize()
    assert list(reloaded.get_conversations("alice")) == [expected]
    assert list(reloaded.get_conversations("bob")) == [expected]
    assert conversation.conversation_id == 2**63 + 42
    stored = reloaded.query(
        "SELECT conversation_id FROM conversation "
        "WHERE conversation_id = :conversation_id",
        {"conversation_id": uint64_to_int64(conversation.conversation_id)},
    )
    assert stored == [{"conversation_id": uint64_to_int64(2**63 + 42)}]
