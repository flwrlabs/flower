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
"""Tests for server-owned agent identity on Runs."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from flwr.common.constant import HEARTBEAT_DEFAULT_INTERVAL, Status, SubStatus
from flwr.proto.control_pb2 import AgentConfig  # pylint: disable=E0611
from flwr.proto.space_pb2 import SpaceParticipant  # pylint: disable=E0611
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION_ID, TaskType
from flwr.supercore.date import now
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from . import LinkState, LinkStateFactory


def _create_state(database: str) -> LinkState:
    """Create a state with an independent factory."""
    return LinkStateFactory(
        database, NoOpFederationManager(), ObjectStoreFactory()
    ).state()


@pytest.fixture(name="state", params=["memory", "sqlite"])
def fixture_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Provide both persistence backends."""
    database = (
        FLWR_IN_MEMORY_DB_NAME
        if request.param == "memory"
        else str(tmp_path / "state.db")
    )
    return _create_state(database)


def _create_run(state: LinkState, space_id: UUID, agent_id: int) -> int:
    """Create a Space AgentApp run with server-assigned identity."""
    return state.create_run(
        "",
        "",
        "hash",
        {},
        None,
        None,
        "alice",
        TaskType.AGENT_APP,
        space_id=space_id,
        agent_id=agent_id,
    )


def test_agent_identity_survives_task_lifecycle(state: LinkState) -> None:
    """Resolve the same executing agent before, during, and after a task claim."""
    other = state.create_agent("alice", AgentConfig(name="Other", app_spec="@flwr/app"))
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@flwr/app"))
    assert agent is not None and other is not None
    space_id = UUID(
        state.create_space(
            participants=[
                SpaceParticipant(agent_id=other.id),
                SpaceParticipant(agent_id=agent.id),
            ]
        ).id
    )
    run_id = _create_run(state, space_id, agent.id)
    assert run_id != 0
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.agent_id == agent.id and run.space_id == space_id
    assert run.primary_task_id is not None
    token = state.claim_task(run.primary_task_id)
    assert token is not None
    assert state.activate_task(run.primary_task_id)
    task = state.get_task_by_token(token)
    assert task is not None
    assert state.get_run_info(run_ids=[task.run_id])[0].agent_id == agent.id
    assert state.finish_task(task.task_id, SubStatus.COMPLETED, "")
    assert state.get_task_by_token(token) is None
    assert state.get_run_info(run_ids=[run_id])[0].agent_id == agent.id


def test_agent_identity_survives_startup_retry(state: LinkState) -> None:
    """Replacing an expired startup token preserves durable agent ownership."""
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@flwr/app"))
    assert agent is not None
    run_id = _create_run(state, UUID(state.get_spaces("alice")[0].id), agent.id)
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    expired_at = now() + timedelta(seconds=HEARTBEAT_DEFAULT_INTERVAL + 1)
    with (
        patch(
            "flwr.supercore.corestate.in_memory_corestate.now", return_value=expired_at
        ),
        patch("flwr.supercore.corestate.sql_corestate.now", return_value=expired_at),
    ):
        assert state.get_task_by_token(token) is None
        assert state.get_tasks(task_ids=[task_id])[0].status.status == Status.PENDING
        replacement = state.claim_task(task_id)
        assert replacement is not None and replacement != token
        task = state.get_task_by_token(replacement)
        assert task is not None
        assert state.get_run_info(run_ids=[task.run_id])[0].agent_id == agent.id


def test_agent_identity_survives_database_reopen(tmp_path: Path) -> None:
    """Read the binding from persistent storage after reopening SQLite."""
    database = str(tmp_path / "state.db")
    state = _create_state(database)
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@flwr/app"))
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    run_id = _create_run(state, space_id, agent.id)
    restored = _create_state(database).get_run_info(run_ids=[run_id])[0]
    assert restored.agent_id == agent.id and restored.space_id == space_id


def test_run_rejects_agent_outside_space(state: LinkState) -> None:
    """Reject unknown agents and nonparticipants without creating orphan tasks."""
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@flwr/app"))
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    other_space_id = UUID(state.create_space(participants=[]).id)
    assert _create_run(state, other_space_id, agent.id) == 0
    assert _create_run(state, space_id, agent.id + 1) == 0
    assert state.get_run_info() == []
    assert state.get_tasks() == [] and state.get_run_series() == []


@pytest.mark.parametrize("agent_id", [0, -1, 2**63, 2**64 - 1])
def test_run_rejects_invalid_agent_id(state: LinkState, agent_id: int) -> None:
    """Reject IDs outside the agent table's supported range before writes."""
    space_id = UUID(state.create_space(participants=[]).id)
    with pytest.raises(ValueError, match="agent_id"):
        _create_run(state, space_id, agent_id)
    assert state.get_run_info() == [] and state.get_tasks() == []


def test_agent_binding_requires_space_run(state: LinkState) -> None:
    """Legacy federation runs cannot acquire an agent binding."""
    with pytest.raises(ValueError, match="agent_id requires a Space run"):
        state.create_run(
            "",
            "",
            "hash",
            {},
            NOOP_FEDERATION_ID,
            None,
            "alice",
            TaskType.AGENT_APP,
            agent_id=1,
        )
    assert state.get_run_info() == []
    assert state.get_tasks() == [] and state.get_run_series() == []
