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
"""CoreState queue persistence independent of Task lifecycle."""

from uuid import UUID

import pytest

from flwr.app import Message
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    SpaceParticipant,
)
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.corestate.sql_corestate import SqlCoreState
from flwr.supercore.object_store import ObjectStoreFactory


class StandaloneCoreState(SqlCoreState):
    """Supply unrelated abstract methods without loading LinkState."""

    def get_node_id(self) -> int:
        """Return the local node identity."""
        return SUPERLINK_NODE_ID

    def store_message_and_object_tree(
        self, message: Message, object_tree: ObjectTree, session_id: str
    ) -> tuple[bool, list[str]]:
        """Leave message publication outside this persistence test."""
        raise NotImplementedError


def test_agent_queue_is_independent_of_tasks() -> None:
    """Persist and deduplicate queue entries without changing generic Tasks."""
    state = StandaloneCoreState(":memory:", ObjectStoreFactory().store())
    state.initialize()
    assert "run" not in state.get_metadata().tables
    agent = state.create_agent(
        "alice", AgentConfig(name="Research", app_spec="@flower/research")
    )
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    first = state.create_space_message(
        space_id, SpaceParticipant(flwr_aid="alice"), "One"
    )
    second = state.create_space_message(
        space_id, SpaceParticipant(flwr_aid="alice"), "Two"
    )
    assert first is not None and second is not None
    head = state.create_agent_queue_entry(space_id, agent.id, first.id, run_id=1)
    tail = state.create_agent_queue_entry(space_id, agent.id, second.id, run_id=2)
    assert head.id < tail.id
    assert not state.get_tasks()
    assert state.get_agent_queue_entry(agent.id, first.id) == head
    assert (
        state.create_agent_queue_entry(space_id, agent.id, first.id, run_id=1) == head
    )
    with pytest.raises(ValueError, match="already assigned"):
        state.create_agent_queue_entry(space_id, agent.id, first.id, run_id=3)

    # The orchestration boundary supplies eligibility; generic Tasks ignore queues.
    first_task = state.create_task(TaskType.AGENT_APP, run_id=1)
    second_task = state.create_task(TaskType.AGENT_APP, run_id=2)
    assert first_task is not None and second_task is not None
    assert state.claim_task(second_task)
    assert state.claim_task(first_task)
