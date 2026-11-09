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
"""Atomic agent submissions and ordered execution."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Barrier
from typing import cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from flwr.app import Message
from flwr.common.constant import SUPERLINK_NODE_ID, Status, SubStatus
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    SpaceParticipant,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    PullTaskInputRequest,
)
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState, SqlLinkState
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    TaskType,
)
from flwr.supercore.date import now
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_utils import pull_and_inflate_object_from_tree
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.servicer.runtime.runtime_handlers import acquire_task
from flwr.supercore.typing import AgentRunTarget
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.runtime.runtime_handlers import pull_task_input


@pytest.fixture(name="state", params=["memory", "sqlite"])
def queue_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise both persistence backends."""
    manager = NoOpFederationManager()
    objects = ObjectStoreFactory().store()
    if request.param == "memory":
        return InMemoryLinkState(manager, objects)
    state = SqlLinkState(str(tmp_path / "state.db"), manager, objects)
    state.initialize()
    return state


def setup_agent(state: LinkState, owner: str = "alice") -> tuple[UUID, AgentRunTarget]:
    """Create an agent with its default Space and stored FAB."""
    agent = state.create_agent(
        owner, AgentConfig(name="Research", app_spec="@flower/research")
    )
    assert agent is not None
    target = AgentRunTarget(
        agent.id, "flower/research", "1.0.0", state.store_fab(Fab("", b"research", {}))
    )
    return UUID(state.get_spaces(owner)[0].id), target


def primary_task(state: LinkState, run_id: int) -> int:
    """Return a Run's primary Task ID."""
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None
    return task_id


def test_input_survives_queue_wait(state: LinkState) -> None:
    """Start delivery after a 13-hour wait, using the accepted text and config."""
    space_id, target = setup_agent(state)
    target.override_config["temperature"] = 0.5
    accepted_at = now()
    message, runs = state.submit_agent_message(
        space_id, "alice", "First prompt", [target]
    )
    target.override_config["temperature"] = 0.9
    assert state.get_space_messages(space_id, "alice") == [message]
    assert state.get_run_info(run_ids=runs)[0].status.status == Status.PENDING
    assert not state.get_messages(dst_node_id=SUPERLINK_NODE_ID, run_id=runs[0])
    with patch("datetime.datetime") as clock:
        clock.now.return_value = accepted_at + timedelta(hours=13)
        acquired = acquire_task(
            AcquireTaskRequest(agentapp_fab_hashes=[target.fab_hash]), state
        )
        response = pull_task_input(PullTaskInputRequest(), state, acquired.task)
        assert response.run.run_id == runs[0]
        prompts = state.get_messages(dst_node_id=SUPERLINK_NODE_ID, run_id=runs[0])
        assert len(prompts) == 1
        prompt = pull_and_inflate_object_from_tree(
            state.object_store.get_object_tree(prompts[0].metadata.message_id),
            lambda object_id: cast(bytes, state.object_store.get(object_id)),
            lambda _: None,
            return_type=Message,
        )
        assert (
            prompt.content[AGENT_MESSAGE_CONTENT_RECORD_KEY][AGENT_MESSAGE_TEXT_KEY]
            == "First prompt"
        )
        assert not state.activate_task(acquired.task.task_id)
    assert state.get_run_info(run_ids=runs)[0].override_config == {"temperature": 0.5}
    assert not state.get_run_series()


@pytest.mark.parametrize(
    "terminal", [SubStatus.COMPLETED, SubStatus.FAILED, SubStatus.STOPPED]
)
def test_ordering_and_child_completion(state: LinkState, terminal: str) -> None:
    """Later messages wait through claim, execution, and child-task cleanup."""
    space_id, target = setup_agent(state)
    first, head_runs = state.submit_agent_message(space_id, "alice", "First", [target])
    second, tail_runs = state.submit_agent_message(
        space_id, "alice", "Second", [target]
    )
    head, tail = primary_task(state, head_runs[0]), primary_task(state, tail_runs[0])
    assert first.id < second.id
    assert state.claim_task(tail) is None
    assert state.claim_task(head)
    assert state.claim_task(tail) is None
    assert state.activate_task(head)
    child = state.create_task(TaskType.CONNECTOR, head_runs[0])
    assert child is not None and state.claim_task(child)
    assert state.claim_task(tail) is None
    assert state.finish_task(head, terminal, "")
    assert all(
        task.status.status == Status.FINISHED
        for task in state.get_tasks(run_ids=head_runs)
    )
    assert state.claim_task(tail)


def test_independent_scopes(state: LinkState) -> None:
    """Other agents and Spaces can execute while the first agent is busy."""
    space_id, target = setup_agent(state)
    _, first = state.submit_agent_message(space_id, "alice", "Personal", [target])
    assert state.claim_task(primary_task(state, first[0]))
    _, other = setup_agent(state, "bob")
    shared = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(agent_id=target.agent_id),
            SpaceParticipant(agent_id=other.agent_id),
        ]
    )
    _, runs = state.submit_agent_message(
        UUID(shared.id), "alice", "Shared", [target, other]
    )
    assert all(state.claim_task(primary_task(state, run_id)) for run_id in runs)


def test_invalid_submissions_leave_no_work(state: LinkState) -> None:
    """Reject invalid inputs, unauthorized users, and unassigned agents."""
    space_id, target = setup_agent(state)
    _, other = setup_agent(state, "bob")
    for scope, account, content, targets in (
        (space_id, "alice", " ", [target]),
        (space_id, "outsider", "Prompt", [target]),
        (uuid4(), "alice", "Prompt", [target]),
        (space_id, "alice", "Prompt", [target, target]),
        (space_id, "alice", "Prompt", [other]),
        (space_id, "alice", "Prompt", [replace(target, fab_id="flower/other")]),
        (space_id, "alice", "Prompt", [replace(target, fab_hash="missing")]),
    ):
        with pytest.raises(ValueError):
            state.submit_agent_message(scope, account, content, targets)
        assert state.get_space_messages(space_id, "alice") == []
        assert not state.get_run_info() and not state.get_tasks()


def test_empty_targets_only_publish_message(state: LinkState) -> None:
    """Publication alone does not trigger agents."""
    space_id, _ = setup_agent(state)
    message, runs = state.submit_agent_message(space_id, "alice", "Chat only", [])
    assert not runs and not state.get_tasks()
    assert state.get_space_messages(space_id, "alice") == [message]


def test_batch_failure_rolls_back_every_run(state: LinkState) -> None:
    """A later failure rolls back the message and partially initialized Runs."""
    _, target = setup_agent(state)
    _, other = setup_agent(state, "bob")
    space = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(agent_id=target.agent_id),
            SpaceParticipant(agent_id=other.agent_id),
        ]
    )
    space_id = UUID(space.id)
    create_entry = state.create_agent_queue_entry

    def fail_entry(scope: UUID, agent_id: int, message_id: int, run_id: int) -> None:
        create_entry(scope, agent_id, message_id, run_id)
        if agent_id == other.agent_id:
            raise RuntimeError("queue insert")

    with patch.object(state, "create_agent_queue_entry", side_effect=fail_entry):
        with pytest.raises(RuntimeError, match="queue insert"):
            state.submit_agent_message(space_id, "alice", "Rollback", [target, other])
    assert not state.get_space_messages(space_id, "alice")
    assert not state.get_run_info() and not state.get_tasks()
    assert state.get_agent_queue_entry(target.agent_id, 1) is None
    assert state.get_agent_queue_entry(other.agent_id, 1) is None
    with patch.object(
        state.object_store, "put", side_effect=AssertionError("unexpected upload")
    ):
        message, runs = state.submit_agent_message(
            space_id, "alice", "Retry", [target, other]
        )
    entry = state.get_agent_queue_entry(target.agent_id, message.id)
    assert entry is not None and entry.run_id == runs[0]


def test_warm_worker_cannot_skip_queue_head(state: LinkState) -> None:
    """An available FAB for a later message cannot bypass earlier work."""
    space_id, target = setup_agent(state)
    _, first = state.submit_agent_message(space_id, "alice", "First", [target])
    other = replace(target, fab_hash=state.store_fab(Fab("", b"other", {})))
    _, second = state.submit_agent_message(space_id, "alice", "Second", [other])
    acquire = AcquireTaskRequest(agentapp_fab_hashes=[other.fab_hash])
    assert not acquire_task(acquire, state).HasField("task")
    assert state.finish_task(primary_task(state, first[0]), SubStatus.STOPPED, "")
    assert acquire_task(acquire, state).task.task_id == primary_task(state, second[0])


def test_concurrent_submissions_survive_restart(tmp_path: Path) -> None:
    """Replicas commit complete submissions and retain message order on restart."""
    database = str(tmp_path / "concurrent.db")
    states = [
        SqlLinkState(database, NoOpFederationManager(), ObjectStoreFactory().store())
        for _ in range(2)
    ]
    for state in states:
        state.initialize()
    space_id, target = setup_agent(states[0])
    barrier = Barrier(2)

    def submit(state: LinkState) -> tuple[int, int]:
        barrier.wait(timeout=10)
        message, runs = state.submit_agent_message(
            space_id, "alice", "Prompt", [target]
        )
        return message.id, runs[0]

    with ThreadPoolExecutor(2) as pool:
        submissions = sorted(pool.map(submit, states))
    restarted = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    restarted.initialize()
    assert len(restarted.get_space_messages(space_id, "alice") or []) == 2
    head, tail = [primary_task(restarted, run_id) for _, run_id in submissions]
    assert restarted.claim_task(tail) is None

    def claim(item: tuple[LinkState, int]) -> str | None:
        barrier.wait(timeout=10)
        return item[0].claim_task(item[1])

    with ThreadPoolExecutor(2) as pool:
        tokens = list(pool.map(claim, [(restarted, head), (states[0], tail)]))
    assert tokens[0] and tokens[1] is None
    assert states[0].claim_task(head) is None
    assert states[1].claim_task(tail) is None


@pytest.mark.parametrize("expired", [False, True])
def test_sql_completion_and_children_are_atomic(tmp_path: Path, expired: bool) -> None:
    """A failed child cleanup rolls back primary completion, including expiry."""
    database = str(tmp_path / "completion.db")
    state = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    state.initialize()
    space_id, target = setup_agent(state)
    _, first = state.submit_agent_message(space_id, "alice", "First", [target])
    _, second = state.submit_agent_message(space_id, "alice", "Second", [target])
    head = primary_task(state, first[0])
    assert state.claim_task(head) and state.activate_task(head)
    child = state.create_task(TaskType.CONNECTOR, first[0])
    assert child is not None and state.claim_task(child)
    expired_at = now() + timedelta(hours=1)
    with patch.object(state, "_finish_run_tasks", side_effect=RuntimeError("cleanup")):
        with pytest.raises(RuntimeError, match="cleanup"):
            if expired:
                with patch("datetime.datetime") as clock:
                    clock.now.return_value = expired_at
                    state.get_tasks()
            else:
                state.finish_task(head, SubStatus.FAILED, "")
    restarted = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory().store()
    )
    restarted.initialize()
    assert restarted.get_tasks(task_ids=[head])[0].status.status == Status.RUNNING
    assert restarted.claim_task(primary_task(restarted, second[0])) is None
    assert restarted.finish_task(head, SubStatus.FAILED, "")
    assert all(
        task.status.status == Status.FINISHED
        for task in restarted.get_tasks(run_ids=first)
    )
    assert restarted.claim_task(primary_task(restarted, second[0]))
