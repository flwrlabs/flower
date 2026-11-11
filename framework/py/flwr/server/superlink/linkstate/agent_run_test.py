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
"""Agent Run eligibility and concurrent admission tests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from uuid import UUID

import pytest

from flwr.proto.control_pb2 import Agent, AgentConfig  # pylint: disable=E0611
from flwr.proto.space_pb2 import SpaceParticipant  # pylint: disable=E0611
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState, SqlLinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.fab import Fab
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager


@pytest.fixture(name="states", params=["memory", "sqlite"])
def backends(
    request: pytest.FixtureRequest, tmp_path: Path
) -> tuple[LinkState, LinkState]:
    """Use one logical backend, with independent connections for file-based SQL."""
    if request.param == "memory":
        memory_state = InMemoryLinkState(
            NoOpFederationManager(), ObjectStoreFactory().store()
        )
        return memory_state, memory_state
    sql_states = [
        SqlLinkState(
            str(tmp_path / "state.db"),
            NoOpFederationManager(),
            ObjectStoreFactory().store(),
        )
        for _ in range(2)
    ]
    for state in sql_states:
        state.initialize()
    return sql_states[0], sql_states[1]


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


def test_unread_input_opens_one_run(states: tuple[LinkState, LinkState]) -> None:
    """Unread user input is eligible until a Run is open for the agent and Space."""
    state = states[0]
    space_id, agent, app = _setup(state)
    assert not state.get_agent_run_candidates()
    state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), "Input")
    assert state.get_agent_run_candidates() == [(space_id, agent)]
    run_ids = [
        state.create_run(
            fab_id=app["fab_id"],
            fab_version=app["fab_version"],
            fab_hash=app["fab_hash"],
            override_config={},
            federation_id=None,
            federation_config=None,
            flwr_aid=None,
            primary_task_type=TaskType.AGENT_APP,
            space_id=space_id,
            agent_id=agent.id,
        )
        for _ in range(2)
    ]
    assert run_ids[0]
    assert run_ids[1] == 0
    assert not state.get_agent_run_candidates()
    assert state.get_run_info(run_ids=[run_ids[0]])[0].flwr_aid == "alice"


def test_concurrent_admission_creates_one_run(
    states: tuple[LinkState, LinkState],
) -> None:
    """Concurrent schedulers cannot open duplicate windows for an agent/Space."""
    state, second = states
    space_id, agent, app = _setup(state)
    state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), "Input")
    barrier = Barrier(2)

    def start(backend: LinkState) -> int:
        barrier.wait(timeout=5)
        return backend.create_run(
            fab_id=app["fab_id"],
            fab_version=app["fab_version"],
            fab_hash=app["fab_hash"],
            override_config={},
            federation_id=None,
            federation_config=None,
            flwr_aid=None,
            primary_task_type=TaskType.AGENT_APP,
            space_id=space_id,
            agent_id=agent.id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        ids = list(pool.map(start, (state, second)))
    assert sum(bool(run_id) for run_id in ids) == 1
    assert len(state.get_run_info()) == 1
