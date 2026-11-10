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

from uuid import UUID

import pytest

from flwr.proto.control_pb2 import AgentConfig  # pylint: disable=E0611
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION_ID, TaskType
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from . import LinkState, LinkStateFactory


@pytest.fixture(name="state", params=[FLWR_IN_MEMORY_DB_NAME, ":memory:"])
def fixture_state(request: pytest.FixtureRequest) -> LinkState:
    """Provide both persistence backends."""
    return LinkStateFactory(
        request.param, NoOpFederationManager(), ObjectStoreFactory()
    ).state()


def _create_run(state: LinkState, space_id: UUID | None, agent_id: int) -> int:
    """Create a Space AgentApp run with server-assigned identity."""
    return state.create_run(
        "",
        "",
        "hash",
        {},
        NOOP_FEDERATION_ID if space_id is None else None,
        None,
        "alice",
        TaskType.AGENT_APP,
        space_id=space_id,
        agent_id=agent_id,
    )


def test_run_binds_only_participating_agent(state: LinkState) -> None:
    """Store a participant's identity and reject unrelated agents before writes."""
    agent = state.create_agent("alice", AgentConfig(name="Agent", app_spec="@flwr/app"))
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    other_space_id = UUID(state.create_space(participants=[]).id)
    assert _create_run(state, other_space_id, agent.id) == 0
    assert _create_run(state, space_id, agent.id + 1) == 0
    assert state.get_run_info() == []
    assert state.get_tasks() == [] and state.get_run_series() == []
    run_id = _create_run(state, space_id, agent.id)
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.agent_id == agent.id and run.space_id == space_id


@pytest.mark.parametrize(
    ("agent_id", "space_run"), [(0, True), (-1, True), (2**63, True), (1, False)]
)
def test_run_rejects_invalid_binding(
    state: LinkState, agent_id: int, space_run: bool
) -> None:
    """Reject unsupported agent IDs and bindings on federation runs."""
    space_id = UUID(state.create_space(participants=[]).id) if space_run else None
    with pytest.raises(ValueError, match="agent_id"):
        _create_run(state, space_id, agent_id)
    assert state.get_run_info() == []
    assert state.get_tasks() == [] and state.get_run_series() == []
