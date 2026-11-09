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
"""Shared application events and ordered, durable user feeds."""

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from threading import Event as ThreadEvent
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Agent,
    AgentConfig,
    Event,
    SpaceParticipant,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState, SqlLinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


def make_event(**payload: str) -> Event:
    """Create an application event with a fresh identity."""
    event = Event(
        event_id=str(uuid4()),
        type="agent.updated",
        created_at=now().isoformat(),
    )
    event.payload.update(payload)
    return event


def sql_state(database: str) -> SqlLinkState:
    """Create an independently connected state using the requested database."""
    state = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    state.initialize()
    return state


@pytest.fixture(name="state", params=["memory", "sqlite", "file"])
def event_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise memory, metadata-created SQL, and migrated SQL."""
    if request.param == "memory":
        return InMemoryLinkState(NoOpFederationManager(), ObjectStoreFactory().store())
    return sql_state(
        ":memory:" if request.param == "sqlite" else str(tmp_path / "events.db")
    )


def test_event_is_shared_and_positions_are_per_user(state: LinkState) -> None:
    """Store shared content once and allocate independent recipient positions."""
    previous = make_event(id="1")
    shared = make_event(id="2")
    shared.created_at = (now() - timedelta(days=1)).isoformat()
    state.store_event(previous, ["bob"])
    state.store_event(shared, ["bob", "alice", "alice"])
    state.store_event(shared, ["alice", "bob"])
    alice = state.get_user_events("alice")
    bob = state.get_user_events("bob")
    assert [entry.seq for entry in alice] == [1]
    assert [entry.seq for entry in bob] == [1, 2]
    assert alice[0].event.event_id == bob[1].event.event_id == shared.event_id
    assert bob[0].event.created_at > bob[1].event.created_at
    assert not state.get_user_events("outsider")
    shared.type = "changed"
    with pytest.raises(ValueError, match="different content"):
        state.store_event(shared, ["alice"])
    shared.payload["id"] = "mutated input"
    alice[0].event.payload["id"] = "mutated result"
    persisted = state.get_user_events("alice")[0].event
    assert persisted.type == "agent.updated" and persisted.payload["id"] == "2"
    assert state.get_bootstrap("alice").cursor == 1


def test_domain_changes_publish_without_task_events(state: LinkState) -> None:
    """Publish Agent and Space changes to recipients while task traces stay separate."""
    agent = state.create_agent(
        "alice", AgentConfig(name="Research", app_spec="@flower/research")
    )
    assert agent is not None
    snapshot = state.get_bootstrap("alice")
    assert snapshot.agents == [agent] and len(snapshot.spaces) == 1
    assert snapshot.cursor == 2
    assert [entry.event.type for entry in state.get_user_events("alice")] == [
        "space.updated",
        "agent.updated",
    ]
    space = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(flwr_aid="bob"),
        ]
    )
    message = state.create_space_message(
        UUID(space.id), SpaceParticipant(flwr_aid="alice"), "Hello"
    )
    assert message is not None
    alice = state.get_user_events("alice", after_seq=3)[0]
    bob = state.get_user_events("bob", after_seq=1)[0]
    assert alice.event.event_id == bob.event.event_id
    assert alice.seq == 4 and bob.seq == 2
    assert alice.event.type == "message.created" and alice.event.space_id == space.id
    assert alice.event.payload["content"] == "Hello"
    assert (
        state.create_space_message(
            UUID(space.id), SpaceParticipant(flwr_aid="outsider"), "Private"
        )
        is None
    )
    task_id = state.create_task(TaskType.CONNECTOR, run_id=1, connector_ref="tool")
    assert task_id is not None
    assert state.store_task_events(
        [TaskEvent(run_id=1, task_id=task_id, event="model", data="{}")]
    )
    assert state.get_bootstrap("alice").cursor == 4
    assert not state.get_user_events("outsider")


def test_feed_keeps_all_events_replayable(state: LinkState) -> None:
    """Appending and replaying events preserves earlier deliveries and positions."""
    first = make_event()
    second = make_event()
    state.store_event(first, ["alice"])
    state.store_event(second, ["alice"])
    assert state.get_bootstrap("alice").cursor == 2
    assert not state.get_user_events("alice", after_seq=2)
    page = state.get_user_events("alice", after_seq=1, limit=1)
    assert len(page) == 1 and page[0].event.event_id == second.event_id
    third = make_event()
    state.store_event(third, ["alice"])
    state.store_event(first, ["alice"])
    entries = state.get_user_events("alice")
    assert [entry.seq for entry in entries] == [1, 2, 3]
    assert [entry.event.event_id for entry in entries] == [
        first.event_id,
        second.event_id,
        third.event_id,
    ]
    for cursor in (-1, 4, 2**64):
        with pytest.raises(FlowerError) as invalid:
            state.get_user_events("alice", after_seq=cursor)
        assert invalid.value.code == ApiErrorCode.INVALID_USER_EVENT_CURSOR


def test_sql_rollback_includes_mutation_fanout_and_counters(tmp_path: Path) -> None:
    """A failed fanout rolls back the Space message, event, and both feed heads."""
    state = sql_state(str(tmp_path / "rollback.db"))
    space = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(flwr_aid="bob"),
        ]
    )
    original = state.store_event

    def fail_after_publication(event: Event, recipients: Sequence[str]) -> None:
        original(event, recipients)
        raise RuntimeError("rollback")

    with patch.object(state, "store_event", side_effect=fail_after_publication):
        with pytest.raises(RuntimeError, match="rollback"):
            state.create_space_message(
                UUID(space.id), SpaceParticipant(flwr_aid="alice"), "Aborted"
            )
    assert state.get_space_messages(UUID(space.id), "alice") == []
    assert state.get_bootstrap("alice").cursor == state.get_bootstrap("bob").cursor == 1
    assert state.create_space_message(
        UUID(space.id), SpaceParticipant(flwr_aid="alice"), "Committed"
    )
    assert state.get_user_events("alice", after_seq=1)[0].seq == 2


def test_sql_feed_commit_order_and_restart(tmp_path: Path) -> None:
    """A later writer cannot commit a recipient position ahead of an earlier writer."""
    database = str(tmp_path / "commits.db")
    first, second, reader = [sql_state(database) for _ in range(3)]
    entered, release = ThreadEvent(), ThreadEvent()
    early = make_event(name="early")
    late = make_event(name="late")

    def write_first() -> None:
        with first.session():
            first.store_event(early, ["bob", "alice"])
            entered.set()
            assert release.wait(timeout=10)

    with ThreadPoolExecutor(2) as pool:
        before = pool.submit(write_first)
        try:
            assert entered.wait(timeout=10)
            after = pool.submit(second.store_event, late, ["alice", "bob"])
            assert not reader.get_user_events("alice")
            assert not after.done()
        finally:
            release.set()
        before.result(timeout=10)
        after.result(timeout=10)
    restarted = sql_state(database)
    assert [entry.event.event_id for entry in restarted.get_user_events("bob")] == [
        early.event_id,
        late.event_id,
    ]
    restarted.store_event(make_event(), ["alice"])
    assert [entry.seq for entry in restarted.get_user_events("alice")] == [1, 2, 3]
    assert restarted.get_user_events("alice", after_seq=2)[0].seq == 3


def test_bootstrap_snapshot_and_resume_have_no_gap(tmp_path: Path) -> None:
    """A mutation arriving between bootstrap reads appears entirely in replay."""
    database = str(tmp_path / "snapshot.db")
    reader, writer = sql_state(database), sql_state(database)
    entered, release = ThreadEvent(), ThreadEvent()
    list_agents = reader.list_agents

    def pause(account: str) -> Sequence[Agent]:
        agents = list_agents(account)
        entered.set()
        assert release.wait(timeout=10)
        return agents

    with patch.object(reader, "list_agents", side_effect=pause):
        with ThreadPoolExecutor(2) as pool:
            bootstrap = pool.submit(reader.get_bootstrap, "alice")
            try:
                assert entered.wait(timeout=10)
                mutation = pool.submit(
                    writer.create_agent,
                    "alice",
                    AgentConfig(name="Research", app_spec="@flower/research"),
                )
                assert not mutation.done()
            finally:
                release.set()
            snapshot = bootstrap.result(timeout=10)
            assert mutation.result(timeout=10) is not None
    assert not snapshot.agents and not snapshot.spaces and snapshot.cursor == 0
    entries = reader.get_user_events("alice", after_seq=snapshot.cursor)
    assert [entry.seq for entry in entries] == [1, 2]
    current = reader.get_bootstrap("alice")
    assert len(current.agents) == len(current.spaces) == 1 and current.cursor == 2
