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
"""Agent execution windows admitted during task acquisition."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import UUID

import pytest

from flwr.common.constant import Status, SubStatus
from flwr.proto.control_pb2 import Agent, AgentConfig  # pylint: disable=E0611
from flwr.proto.space_pb2 import SpaceParticipant  # pylint: disable=E0611
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState, SqlLinkState
from flwr.supercore.fab import Fab
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


@pytest.fixture(name="states", params=["memory", "sql-memory", "sqlite"])
def backends(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[LinkState, LinkState]:
    """Use one logical backend, with independent connections for file-based SQL."""
    if request.param == "memory":
        memory_state = InMemoryLinkState(
            NoOpFederationManager(), ObjectStoreFactory().store()
        )
        return memory_state, memory_state
    database = (
        ":memory:" if request.param == "sql-memory" else str(tmp_path / "state.db")
    )
    state = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    state.initialize()
    if request.param == "sql-memory":
        return state, state
    second = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    second.initialize()
    return state, second


@pytest.fixture(name="state")
def first_backend(states: tuple[LinkState, LinkState]) -> LinkState:
    """Expose the LinkState contract shared by all backends."""
    return states[0]


def _setup(state: LinkState) -> tuple[UUID, Agent, dict[str, str]]:
    agent = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@publisher/app")
    )
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    fab_hash = state.store_fab(Fab("", b"agent-fab", {}))
    return (
        space_id,
        agent,
        {"fab_id": "publisher/app", "fab_version": "1.0", "fab_hash": fab_hash},
    )


def test_execution_window_accepts_followups(state: LinkState) -> None:
    """One pending/starting/running invocation can read several user messages."""
    space_id, agent, app = _setup(state)
    user = SpaceParticipant(flwr_aid="alice")
    member = SpaceParticipant(agent_id=agent.id)
    first = state.create_space_message(space_id, user, "First")
    assert not state.get_run_info()
    assert state.get_agent_run_candidates() == [(space_id, agent)]
    run_id = state.create_agent_run(space_id, agent.id, **app)
    assert run_id is not None
    assert not state.get_agent_run_candidates()
    assert state.create_agent_run(space_id, agent.id, **app) is None
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.flwr_aid == "alice" and run.agent_id == agent.id
    assert run.space_id == space_id and not run.series_id and not run.federation_id
    task_id = run.primary_task_id
    assert task_id is not None
    assert state.claim_task(task_id)
    assert not state.get_agent_run_candidates()
    assert state.activate_task(task_id)
    assert state.get_run_info(run_ids=[run_id])[0].status.status == Status.RUNNING
    second = state.create_space_message(space_id, user, "Follow-up")
    assert not state.get_agent_run_candidates()
    assert state.get_space_messages(space_id, member, unread_only=True) == [
        first,
        second,
    ]
    assert state.finish_task(task_id, SubStatus.COMPLETED, "")
    state.create_space_message(space_id, member, "Agent output")
    assert not state.get_agent_run_candidates()
    state.create_space_message(space_id, user, "Next window")
    assert state.create_agent_run(space_id, agent.id, **app) not in (None, run_id)


@pytest.mark.parametrize("read_count", [0, 1, 2])
def test_failure_keeps_only_unread_input_eligible(
    state: LinkState, read_count: int
) -> None:
    """Retrieval governs scheduling; failed processing does not rewind the cursor."""
    space_id, agent, app = _setup(state)
    for text in ("First", "Second"):
        state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), text)
    run_id = state.create_agent_run(space_id, agent.id, **app)
    assert run_id is not None
    if read_count:
        assert state.get_space_messages(
            space_id,
            SpaceParticipant(agent_id=agent.id),
            unread_only=True,
            limit=read_count,
        )
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None and state.finish_task(
        task_id, SubStatus.FAILED, "Failed"
    )
    assert bool(state.get_agent_run_candidates()) == (read_count < 2)


def test_participation_and_unread_input_are_rechecked(state: LinkState) -> None:
    """Broadcast to participating agents only and recheck after FAB resolution."""
    _, first, app = _setup(state)
    _, second, _ = _setup(state)
    _, outsider, _ = _setup(state)
    user = SpaceParticipant(flwr_aid="bob")
    space_id = UUID(
        state.create_space(
            participants=[
                user,
                SpaceParticipant(agent_id=first.id),
                SpaceParticipant(agent_id=second.id),
            ]
        ).id
    )
    state.create_space_message(space_id, user, "For both agents")
    assert {agent.id for _, agent in state.get_agent_run_candidates()} == {
        first.id,
        second.id,
    }
    assert state.create_agent_run(space_id, outsider.id, **app) is None
    with pytest.raises(ValueError, match="stored"):
        state.create_agent_run(space_id, first.id, **dict(app, fab_hash="missing"))
    assert (
        state.create_agent_run(space_id, first.id, **dict(app, fab_id="other/app"))
        is None
    )
    state.get_space_messages(
        space_id, SpaceParticipant(agent_id=first.id), unread_only=True
    )
    assert state.create_agent_run(space_id, first.id, **app) is None
    assert state.create_agent_run(space_id, second.id, **app) is not None
    assert state.get_run_info()[0].flwr_aid == "alice"


@pytest.mark.parametrize("states", ["memory", "sqlite"], indirect=True)
def test_concurrent_admission_creates_one_run(
    states: tuple[LinkState, LinkState],
) -> None:
    """Concurrent schedulers cannot open duplicate windows for an agent/Space."""
    state, second = states
    space_id, agent, app = _setup(state)
    state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), "Input")
    barrier = Barrier(2)

    def start(backend: LinkState) -> int | None:
        barrier.wait(timeout=5)
        return backend.create_agent_run(space_id, agent.id, **app)

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(start, (state, second)))
    assert sum(run_id is not None for run_id in ids) == 1
    assert len(state.get_run_info()) == 1
