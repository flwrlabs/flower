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
"""Tests for replayable, account-authorized Control events."""

# pylint: disable=protected-access

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, update

from flwr.common.constant import Status, SubStatus
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    SpaceParticipant,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.linkstate.in_memory_linkstate import InMemoryLinkState
from flwr.server.superlink.linkstate.sql_linkstate import SqlLinkState
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION_ID, TaskType
from flwr.supercore.date import now
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.state.schema.corestate_models import SpaceParticipantTable
from flwr.supercore.state.schema.corestate_models import Task as TaskModel
from flwr.supercore.state.schema.linkstate_models import Run as RunModel
from flwr.superlink.federation import NoOpFederationManager


@pytest.fixture(name="state", params=["memory", ":memory:", "sqlite"])
def event_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise memory state, metadata-only SQLite, and migrated SQLite."""
    database = {
        "memory": FLWR_IN_MEMORY_DB_NAME,
        ":memory:": ":memory:",
        "sqlite": str(tmp_path / "state.db"),
    }[request.param]
    return LinkStateFactory(
        database, NoOpFederationManager(), ObjectStoreFactory()
    ).state()


def test_visibility_and_replay(state: LinkState) -> None:
    """Replay only owned Agent changes and currently participating Spaces."""
    agent = state.create_agent(
        "alice", AgentConfig(name="Research", app_spec="@flwr/research")
    )
    assert agent is not None
    state.create_agent("bob", AgentConfig(name="Private", app_spec="@flwr/private"))
    alice = SpaceParticipant(flwr_aid="alice")
    shared = state.create_space(participants=[alice, SpaceParticipant(flwr_aid="bob")])
    private = state.create_space(participants=[alice])
    message = state.create_space_message(UUID(shared.id), alice, "Hello, €100.\nNext.")
    assert message is not None
    before = state.get_control_events("alice")
    assert [event.WhichOneof("event") for event in before] == [
        "agent_created",
        "space_created",
        "space_created",
        "message_created",
    ]
    assert before[0].agent_created == agent and before[-1].message_created == message
    bob = state.get_control_events("bob")
    assert [event.WhichOneof("event") for event in bob] == [
        "agent_created",
        "space_created",
        "message_created",
    ]
    assert all(event.space_id != private.id for event in bob)
    assert state.get_control_events("charlie") == []
    assert [event.id for event in before] == sorted({event.id for event in before})
    page = state.get_control_events("alice", limit=2)
    assert list(page) + list(
        state.get_control_events("alice", after_id=page[-1].id)
    ) == list(before)
    assert state.get_control_events("alice", after_id=2**64 - 1) == []
    assert (
        state.create_space_message(
            UUID(shared.id), SpaceParticipant(flwr_aid="charlie"), "Denied"
        )
        is None
    )
    with pytest.raises(ValueError):
        state.create_space(participants=[SpaceParticipant(agent_id=999)])
    assert state.get_control_events("alice") == before
    if isinstance(state, SqlLinkState) and not state._is_in_memory_sqlite:
        reopened = SqlLinkState(
            state.database_url, NoOpFederationManager(), ObjectStoreFactory().store()
        )
        reopened.initialize()
        assert reopened.get_control_events("alice") == before


def test_replay_checks_current_membership(state: LinkState) -> None:
    """Historical events become inaccessible when participation is removed."""
    bob = SpaceParticipant(flwr_aid="bob")
    space = state.create_space(participants=[SpaceParticipant(flwr_aid="alice"), bob])
    assert state.get_control_events("bob")
    if isinstance(state, InMemoryLinkState):
        state.space_store[UUID(space.id)].participants.remove(bob)
    else:
        assert isinstance(state, SqlLinkState)
        with state.session() as session:
            session.execute(
                delete(SpaceParticipantTable).where(
                    SpaceParticipantTable.c.space_id == UUID(space.id),
                    SpaceParticipantTable.c.flwr_aid == "bob",
                )
            )
    assert state.get_control_events("bob") == []


def test_space_task_events_and_lifecycle(state: LinkState) -> None:
    """Include Space run events and status while excluding legacy runs."""
    space = state.create_space(participants=[SpaceParticipant(flwr_aid="alice")])
    runs = [
        state.create_run(
            "", "", "hash", {}, NOOP_FEDERATION_ID, None, "alice", TaskType.AGENT_APP
        )
        for _ in range(2)
    ]
    run = state.get_run_info(run_ids=[runs[0]])[0]
    assert run.primary_task_id is not None
    # The nullable Run.space_id column is already on main. Space submission
    # and its run-creation mode live in a separate PR.
    if isinstance(state, InMemoryLinkState):
        state.run_ids[run.run_id].run.space_id = UUID(space.id)
    else:
        assert isinstance(state, SqlLinkState)
        with state.session() as session:
            session.execute(
                update(RunModel)
                .where(RunModel.run_id == run.run_id)
                .values(space_id=UUID(space.id))
            )
    state._on_task_status_changed(run.primary_task_id)
    assert state.claim_task(run.primary_task_id) is not None
    assert state.activate_task(run.primary_task_id)
    legacy_task_id = state.get_run_info(run_ids=[runs[1]])[0].primary_task_id
    assert legacy_task_id is not None
    assert state.store_task_events(
        [
            TaskEvent(
                run_id=run.run_id,
                task_id=run.primary_task_id,
                event="delta",
                data='{"text":"Visible"}',
            ),
            TaskEvent(
                run_id=runs[1],
                task_id=legacy_task_id,
                event="delta",
                data='{"text":"Legacy"}',
            ),
        ]
    )
    assert state.finish_task(run.primary_task_id, SubStatus.COMPLETED, "")
    events = state.get_control_events("alice")
    statuses = [
        event.run_status.status.status
        for event in events
        if event.HasField("run_status")
    ]
    assert statuses == [
        Status.PENDING,
        Status.STARTING,
        Status.RUNNING,
        Status.FINISHED,
    ]
    task_events = [event.task_event for event in events if event.HasField("task_event")]
    assert task_events == state.get_task_events(run_ids=[run.run_id])
    assert state.get_control_events("bob") == []


def test_journal_rollback(tmp_path: Path) -> None:
    """Domain rows, journal rows, and cursor allocation roll back together."""
    state = SqlLinkState(
        str(tmp_path / "state.db"),
        NoOpFederationManager(),
        ObjectStoreFactory().store(),
    )
    state.initialize()
    with pytest.raises(RuntimeError, match="rollback"):
        with state.session():
            state.create_agent(
                "alice", AgentConfig(name="Aborted", app_spec="@flwr/test")
            )
            raise RuntimeError("rollback")
    assert state.list_agents("alice") == [] and state.get_control_events("alice") == []
    state.create_agent("alice", AgentConfig(name="Committed", app_spec="@flwr/test"))
    assert state.get_control_events("alice")[0].id == 1


def test_sql_publication_is_atomic_and_ordered(tmp_path: Path) -> None:
    """Other replicas see only committed events; concurrent writers share a cursor."""
    database = str(tmp_path / "state.db")
    first = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    second = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    first.initialize()
    second.initialize()
    with ThreadPoolExecutor(max_workers=4) as pool:
        with first.session():
            first.create_agent(
                "alice", AgentConfig(name="First", app_spec="@flwr/test")
            )
            assert (
                pool.submit(second.get_control_events, "alice").result(timeout=5) == []
            )
        assert second.get_control_events("alice")[0].id == 1

        def create(index: int) -> None:
            state = first if index % 2 else second
            assert (
                state.create_agent(
                    "alice", AgentConfig(name=str(index), app_spec="@flwr/test")
                )
                is not None
            )

        list(pool.map(create, range(16)))
    events = second.get_control_events("alice")
    assert [event.id for event in events] == list(range(1, 18))
    assert {event.agent_created.config.name for event in events} == {
        "First",
        *(str(index) for index in range(16)),
    }


def test_expired_space_run_is_reported(state: LinkState) -> None:
    """Worker loss produces a terminal Space run event without a client write."""
    space = state.create_space(participants=[SpaceParticipant(flwr_aid="alice")])
    run_id = state.create_run(
        "", "", "hash", {}, NOOP_FEDERATION_ID, None, "alice", TaskType.AGENT_APP
    )
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None
    if isinstance(state, InMemoryLinkState):
        state.run_ids[run_id].run.space_id = UUID(space.id)
    else:
        assert isinstance(state, SqlLinkState)
        with state.session() as session:
            session.execute(
                update(RunModel)
                .where(RunModel.run_id == run_id)
                .values(space_id=UUID(space.id))
            )
    assert state.claim_task(task_id) and state.activate_task(task_id)
    expired_at = now() - timedelta(seconds=1)
    if isinstance(state, InMemoryLinkState):
        state.task_token_store[task_id].active_until = expired_at
    else:
        with state.session() as session:
            session.execute(
                update(TaskModel)
                .where(TaskModel.task_id == task_id)
                .values(active_until=expired_at)
            )
    state.get_tasks(task_ids=[task_id])
    event = state.get_control_events("alice")[-1]
    assert event.run_status.status.status == Status.FINISHED
    assert event.run_status.status.sub_status == SubStatus.FAILED
