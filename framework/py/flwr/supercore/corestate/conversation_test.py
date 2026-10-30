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
"""Conversation persistence and compatibility tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.exc import IntegrityError

from flwr.proto.control_pb2 import Conversation  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import NOOP_FEDERATION_ID, TaskType
from flwr.supercore.object_store import ObjectStoreFactory
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


def test_create_empty_conversation_and_reuse_series(state: LinkState) -> None:
    """Creation starts no work; the existing run path can reuse its series."""
    conversation = state.create_conversation(
        flwr_aid="alice", federation_id=NOOP_FEDERATION_ID
    )
    assert conversation.title == ""
    series = state.get_run_series(series_ids=[conversation.series_id], is_agent=True)
    assert len(series) == 1
    assert series[0].federation == NOOP_FEDERATION_ID
    assert not series[0].run_ids
    assert not state.get_tasks()
    assert state.get_run_series_context(conversation.series_id) is None
    assert state.get_conversation(conversation.conversation_id, "alice") == conversation
    assert Conversation.FromString(conversation.SerializeToString()) == conversation

    run_id = state.create_run(
        fab_id=None,
        fab_version=None,
        fab_hash=None,
        override_config={},
        federation_id=NOOP_FEDERATION_ID,
        federation_config=None,
        flwr_aid="alice",
        primary_task_type=TaskType.AGENT_APP,
        series_id=conversation.series_id,
        user_prompt="Hello",
    )
    assert run_id
    assert state.get_run_info(run_ids=[run_id])[0].series_id == conversation.series_id
    assert list(
        state.get_run_series(series_ids=[conversation.series_id])[0].run_ids
    ) == [run_id]


def test_owner_isolation_and_federation_filters(state: LinkState) -> None:
    """Reading and renaming cannot cross owners, even in the same federation."""
    own = state.create_conversation(flwr_aid="alice", federation_id="a")
    other_federation = state.create_conversation(flwr_aid="alice", federation_id="b")
    state.create_conversation(flwr_aid="bob", federation_id="a")

    assert len(state.get_conversations("alice")) == 2
    assert list(state.get_conversations("alice", ["a"])) == [own]
    assert list(state.get_conversations("alice", ["b"])) == [other_federation]
    assert not state.get_conversations("alice", [])
    assert not state.get_conversations("unknown")
    assert state.get_conversation(own.conversation_id, "bob") is None
    assert state.set_conversation_title(own.conversation_id, "bob", "Changed") is None
    assert state.get_conversation(0, "alice") is None
    assert state.set_conversation_title(0, "alice", "Changed") is None
    assert state.get_conversation(own.conversation_id, "alice") == own


def test_rename_order_noop_and_returned_message_isolation(state: LinkState) -> None:
    """Renaming updates metadata order; no-ops and caller mutations preserve state."""
    initial = datetime(2026, 10, 30, tzinfo=UTC)
    with (
        patch("flwr.supercore.corestate.sql_corestate.now", return_value=initial),
        patch("flwr.supercore.corestate.in_memory_corestate.now", return_value=initial),
    ):
        first = state.create_conversation(flwr_aid="alice", federation_id="a")
        second = state.create_conversation(flwr_aid="alice", federation_id="a")

    later = initial + timedelta(seconds=1)
    with (
        patch("flwr.supercore.corestate.sql_corestate.now", return_value=later),
        patch("flwr.supercore.corestate.in_memory_corestate.now", return_value=later),
    ):
        renamed = state.set_conversation_title(
            first.conversation_id, "alice", "Renamed"
        )
    assert renamed is not None
    assert renamed.created_at == first.created_at
    assert renamed.updated_at == later.isoformat()
    assert renamed.series_id == first.series_id
    assert [item.conversation_id for item in state.get_conversations("alice")] == [
        first.conversation_id,
        second.conversation_id,
    ]
    assert (
        state.set_conversation_title(first.conversation_id, "alice", "Renamed")
        == renamed
    )

    renamed.title = "Caller mutation"
    listed = state.get_conversations("alice")
    listed[0].title = "Another caller mutation"
    stored = state.get_conversation(first.conversation_id, "alice")
    assert stored is not None
    assert stored.title == "Renamed"


def test_sql_creation_rolls_back_backing_series_on_failure() -> None:
    """A failed conversation insert cannot leave an orphan backing series."""
    sql_state = SqlLinkState(
        ":memory:", NoOpFederationManager(), ObjectStoreFactory().store()
    )
    sql_state.initialize()
    original = sql_state.create_conversation(flwr_aid="alice", federation_id="a")
    with (
        patch(
            "flwr.supercore.corestate.sql_corestate.generate_rand_int_from_bytes",
            side_effect=[original.conversation_id, original.series_id ^ 1],
        ),
        pytest.raises(IntegrityError),
    ):
        sql_state.create_conversation(flwr_aid="alice", federation_id="a")
    assert len(sql_state.get_run_series()) == 1
    assert list(sql_state.get_conversations("alice")) == [original]


def test_sql_reload_preserves_unsigned_ids(tmp_path: Path) -> None:
    """Migrated file storage retains full uint64 conversation and series IDs."""
    database_path = str(tmp_path / "persistent.db")
    sql_state = SqlLinkState(
        database_path, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    sql_state.initialize()
    with patch(
        "flwr.supercore.corestate.sql_corestate.generate_rand_int_from_bytes",
        return_value=2**63 + 42,
    ):
        conversation = sql_state.create_conversation(
            flwr_aid="alice", federation_id="a"
        )
    reloaded = SqlLinkState(
        database_path, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    reloaded.initialize()
    assert (
        reloaded.get_conversation(conversation.conversation_id, "alice") == conversation
    )
    assert conversation.series_id == 2**63 + 42
    assert conversation.conversation_id == 2**63 + 42


@pytest.mark.parametrize("conversation_id", [42, 2**63 - 1, 2**63, 2**64 - 1])
def test_unsigned_conversation_ids_roundtrip(
    state: LinkState, conversation_id: int
) -> None:
    """IDs retain their uint64 value through storage, protobuf, reads, and edits."""
    with (
        patch(
            "flwr.supercore.corestate.sql_corestate.generate_rand_int_from_bytes",
            side_effect=[conversation_id, 17],
        ),
        patch(
            "flwr.supercore.corestate.in_memory_corestate.generate_rand_int_from_bytes",
            side_effect=[conversation_id, 17],
        ),
    ):
        conversation = state.create_conversation(flwr_aid="alice", federation_id="a")
    assert conversation.conversation_id == conversation_id
    assert Conversation.FromString(conversation.SerializeToString()) == conversation
    assert state.get_conversation(conversation_id, "alice") == conversation
    renamed = state.set_conversation_title(conversation_id, "alice", "Renamed")
    assert renamed is not None
    assert renamed.conversation_id == conversation_id
    assert renamed.title == "Renamed"
    assert list(state.get_conversations("alice")) == [renamed]


def test_equal_timestamp_order_uses_unsigned_ids(state: LinkState) -> None:
    """The deterministic tie-breaker agrees across signed SQL and in-memory IDs."""
    timestamp = datetime(2026, 10, 30, tzinfo=UTC)
    with (
        patch("flwr.supercore.corestate.sql_corestate.now", return_value=timestamp),
        patch(
            "flwr.supercore.corestate.in_memory_corestate.now", return_value=timestamp
        ),
        patch(
            "flwr.supercore.corestate.sql_corestate.generate_rand_int_from_bytes",
            side_effect=[42, 1, 2**63 + 42, 2],
        ),
        patch(
            "flwr.supercore.corestate.in_memory_corestate.generate_rand_int_from_bytes",
            side_effect=[42, 1, 2**63 + 42, 2],
        ),
    ):
        state.create_conversation(flwr_aid="alice", federation_id="a")
        state.create_conversation(flwr_aid="alice", federation_id="a")
    assert [item.conversation_id for item in state.get_conversations("alice")] == [
        2**63 + 42,
        42,
    ]
