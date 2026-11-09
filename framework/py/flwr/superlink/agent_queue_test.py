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
"""Agent acceptance and task eligibility across state backends."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from threading import Barrier, Event
from typing import cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from flwr.app import Message
from flwr.common.constant import (
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_PATIENCE,
    SUPERLINK_NODE_ID,
    Status,
    SubStatus,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AgentConfig,
    SpaceMessage,
    SpaceParticipant,
)
from flwr.proto.runtime_pb2 import AcquireTaskRequest  # pylint: disable=E0611
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
from flwr.supercore.object_store.in_memory_object_store import InMemoryObjectStore
from flwr.supercore.object_store.object_store import NoObjectInStoreError
from flwr.supercore.servicer.runtime.runtime_handlers import acquire_task
from flwr.superlink.agent_queue import (
    AgentRunRequest,
    AgentRunTarget,
    enqueue_agent_run,
    submit_agent_message,
)
from flwr.superlink.federation import NoOpFederationManager


@pytest.fixture(name="state", params=["memory", "sqlite", "sqlite-file"])
def orchestration_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise memory, metadata-created SQL, and migrated SQL state."""
    manager = NoOpFederationManager()
    objects = ObjectStoreFactory().store()
    if request.param == "memory":
        return InMemoryLinkState(manager, objects)
    database = ":memory:" if request.param == "sqlite" else str(tmp_path / "state.db")
    state = SqlLinkState(database, manager, objects)
    state.initialize()
    return state


def make_request(state: LinkState) -> AgentRunRequest:
    """Prepare a stored FAB, agent, Space, and user message."""
    agent = state.create_agent(
        "alice", AgentConfig(name="Research", app_spec="@flower/research")
    )
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    source = state.create_space_message(
        space_id, SpaceParticipant(flwr_aid="alice"), "First prompt"
    )
    assert source is not None
    fab_hash = state.store_fab(Fab("", b"research-fab", {}))
    return AgentRunRequest(
        space_id, agent.id, source.id, "alice", "flower/research", "1.0.0", fab_hash
    )


def follow_up(state: LinkState, request: AgentRunRequest) -> AgentRunRequest:
    """Publish another user message without scheduling it implicitly."""
    source = state.create_space_message(
        request.space_id, SpaceParticipant(flwr_aid=request.flwr_aid), "Next prompt"
    )
    assert source is not None
    return replace(request, message_id=source.id)


def primary_task(state: LinkState, run_id: int) -> int:
    """Return the primary task created for a Run."""
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None
    return task_id


def run_target(request: AgentRunRequest) -> AgentRunTarget:
    """Keep only the resolved target fields when accepting a new message."""
    return AgentRunTarget(
        request.agent_id,
        request.fab_id,
        request.fab_version,
        request.fab_hash,
        request.override_config,
    )


def shared_targets(state: LinkState) -> tuple[UUID, list[AgentRunTarget]]:
    """Prepare explicit targets for two agents in one initially empty Space."""
    request = make_request(state)
    other = state.create_agent(
        "alice", AgentConfig(name="Other", app_spec="@flower/research")
    )
    assert other is not None
    space = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(agent_id=request.agent_id),
            SpaceParticipant(agent_id=other.id),
        ]
    )
    first = run_target(request)
    return UUID(space.id), [first, replace(first, agent_id=other.id)]


def read_prompt(state: LinkState, run_id: int) -> str:
    """Inflate the persisted input, including SQL's metadata-only messages."""
    messages = state.get_messages(dst_node_id=SUPERLINK_NODE_ID, run_id=run_id)
    assert len(messages) == 1
    message = pull_and_inflate_object_from_tree(
        state.object_store.get_object_tree(messages[0].metadata.message_id),
        lambda object_id: cast(bytes, state.object_store.get(object_id)),
        lambda _: None,
        return_type=Message,
    )
    return cast(
        str, message.content[AGENT_MESSAGE_CONTENT_RECORD_KEY][AGENT_MESSAGE_TEXT_KEY]
    )


def test_acceptance_pins_input_and_deduplicates(state: LinkState) -> None:
    """A replay returns the accepted Run even after failure or new config."""
    request = replace(make_request(state), override_config={"temperature": 0.5})
    first = enqueue_agent_run(state, request)
    assert read_prompt(state, first) == "First prompt"
    assert state.finish_task(primary_task(state, first), SubStatus.FAILED, "failed")
    request.override_config["temperature"] = 0.9
    assert enqueue_agent_run(state, replace(request, fab_version="2.0.0")) == first
    assert state.get_run_info(run_ids=[first])[0].override_config == {
        "temperature": 0.5
    }

    second = enqueue_agent_run(state, follow_up(state, request))
    assert second != first
    assert read_prompt(state, second) == "Next prompt"
    assert len(state.get_tasks()) == 2
    assert state.get_run_series() == []


@pytest.mark.parametrize(
    "terminal", [SubStatus.COMPLETED, SubStatus.FAILED, SubStatus.STOPPED]
)
def test_fifo_and_terminal_advancement(state: LinkState, terminal: str) -> None:
    """Pending, starting, and running heads block the next primary task."""
    request = make_request(state)
    first = primary_task(state, enqueue_agent_run(state, request))
    second = primary_task(state, enqueue_agent_run(state, follow_up(state, request)))
    assert state.claim_task(second) is None
    assert state.claim_task(first)
    assert state.claim_task(second) is None
    assert state.activate_task(first)
    assert state.claim_task(second) is None
    assert state.finish_task(first, terminal, "")
    assert state.claim_task(second)


def test_parallel_scopes_and_child_tasks(state: LinkState) -> None:
    """The gate affects only primary tasks for the same agent and Space."""
    request = make_request(state)
    run_id = enqueue_agent_run(state, request)
    assert state.claim_task(primary_task(state, run_id))
    child = state.create_task(TaskType.CONNECTOR, run_id, connector_ref="tool")
    assert child is not None and state.claim_task(child)

    other_agent = state.create_agent(
        "bob", AgentConfig(name="Other", app_spec="@flower/research")
    )
    assert other_agent is not None
    space = state.create_space(
        participants=[
            SpaceParticipant(flwr_aid="alice"),
            SpaceParticipant(agent_id=request.agent_id),
            SpaceParticipant(agent_id=other_agent.id),
        ]
    )
    shared = follow_up(state, replace(request, space_id=UUID(space.id)))
    other_run = enqueue_agent_run(state, shared)
    assert state.claim_task(primary_task(state, other_run))
    # Alice can invoke Bob's agent because it participates in the shared Space.
    parallel = enqueue_agent_run(state, replace(shared, agent_id=other_agent.id))
    assert state.claim_task(primary_task(state, parallel))


def test_child_cleanup_precedes_advancement(state: LinkState) -> None:
    """A successor cannot start between root completion and child cleanup."""
    request = make_request(state)
    run_id = enqueue_agent_run(state, request)
    head = primary_task(state, run_id)
    tail = primary_task(state, enqueue_agent_run(state, follow_up(state, request)))
    assert state.claim_task(head)
    child = state.create_task(TaskType.CONNECTOR, run_id)
    assert child is not None and state.claim_task(child)
    assert isinstance(state, (InMemoryLinkState, SqlLinkState))
    finish_children = state._finish_run_tasks  # pylint: disable=protected-access

    def finish_after_check(
        run_primary_pairs: list[tuple[int, int]], sub_status: str, details: str
    ) -> None:
        assert state.claim_task(tail) is None
        finish_children(run_primary_pairs, sub_status, details)

    with patch.object(state, "_finish_run_tasks", side_effect=finish_after_check):
        assert state.finish_task(head, SubStatus.FAILED, "failed")
    assert state.claim_task(tail)


def test_warm_worker_cannot_skip_unavailable_head(state: LinkState) -> None:
    """Runtime acquisition respects FIFO even if only the next FAB is warm."""
    request = make_request(state)
    head = primary_task(state, enqueue_agent_run(state, request))
    next_request = replace(
        follow_up(state, request),
        fab_hash=state.store_fab(Fab("", b"next-fab", {})),
        fab_version="2.0.0",
    )
    tail = primary_task(state, enqueue_agent_run(state, next_request))
    acquire = AcquireTaskRequest(agentapp_fab_hashes=[next_request.fab_hash])
    assert not acquire_task(acquire, state).HasField("task")
    assert state.finish_task(head, SubStatus.STOPPED, "cancelled")
    assert acquire_task(acquire, state).task.task_id == tail


def test_reject_ineligible_triggers(state: LinkState) -> None:
    """Only an assigned agent and a caller-authored user message are eligible."""
    request = make_request(state)
    agent_message = state.create_space_message(
        request.space_id, SpaceParticipant(agent_id=request.agent_id), "Agent output"
    )
    assert agent_message is not None
    other = state.create_agent(
        "alice", AgentConfig(name="Other", app_spec="@flower/research")
    )
    assert other is not None
    for invalid in (
        replace(request, flwr_aid="outsider"),
        replace(request, agent_id=other.id),
        replace(request, space_id=uuid4()),
        replace(request, message_id=agent_message.id),
        replace(request, fab_id="flower/different"),
        replace(request, fab_hash="missing"),
    ):
        with pytest.raises(ValueError):
            enqueue_agent_run(state, invalid)
    assert not state.get_tasks()


def test_prompt_failure_rolls_back_acceptance(state: LinkState) -> None:
    """Failed prompt uploads leave neither queued tasks nor orphan objects."""
    request = make_request(state)
    with patch.object(state.object_store, "put", side_effect=RuntimeError("upload")):
        with pytest.raises(RuntimeError, match="upload"):
            enqueue_agent_run(state, request)
    assert not state.get_run_info()
    assert not state.get_tasks()
    assert state.get_agent_queue_entry(request.agent_id, request.message_id) is None
    assert not state.get_messages(dst_node_id=SUPERLINK_NODE_ID)
    assert isinstance(state.object_store, InMemoryObjectStore)
    assert not state.object_store.store
    assert not state.object_store.run_objects_mapping
    assert len(state.get_space_messages(request.space_id, "alice") or []) == 1
    assert state.claim_task(primary_task(state, enqueue_agent_run(state, request)))


@pytest.mark.parametrize("shared", [False, True])
def test_sql_objectstore_rollback(tmp_path: Path, shared: bool) -> None:
    """Roll back two full uploads with shared and independent SQL stores."""
    database = str(tmp_path / "state.db")
    object_database = database if shared else ":memory:"
    objects = ObjectStoreFactory(object_database).store()
    state = SqlLinkState(database, NoOpFederationManager(), objects)
    state.initialize()
    space_id, targets = shared_targets(state)
    store_prompt = state._store_generated_message  # pylint: disable=protected-access
    prompt_ids = []

    def fail_after_upload(message: Message) -> None:
        store_prompt(message)
        prompt_ids.append(message.metadata.message_id)
        if len(prompt_ids) == 2:
            raise RuntimeError("after upload")

    with patch.object(state, "_store_generated_message", side_effect=fail_after_upload):
        with pytest.raises(RuntimeError, match="after upload"):
            submit_agent_message(state, space_id, "alice", "First prompt", targets)
    assert not state.get_run_info() and not state.get_tasks()
    assert not state.get_space_messages(space_id, "alice")
    assert len(prompt_ids) == 2
    for prompt_id in prompt_ids:
        assert objects.get(prompt_id) is None
        with pytest.raises(NoObjectInStoreError):
            objects.get_object_tree(prompt_id)
    _, runs = submit_agent_message(state, space_id, "alice", "First prompt", targets)
    assert all(read_prompt(state, run_id) == "First prompt" for run_id in runs)


@pytest.mark.parametrize("active", [False, True])
def test_worker_expiry_preserves_queue_contract(state: LinkState, active: bool) -> None:
    """Startup expiry retries the head; execution expiry fails and advances it."""
    request = make_request(state)
    first = primary_task(state, enqueue_agent_run(state, request))
    second = primary_task(state, enqueue_agent_run(state, follow_up(state, request)))
    fixed_now = now()
    with patch("datetime.datetime") as clock:
        clock.now.return_value = fixed_now
        token = state.claim_task(first)
        assert token
        if active:
            assert state.activate_task(first)
        clock.now.return_value = fixed_now + timedelta(
            seconds=HEARTBEAT_DEFAULT_INTERVAL * HEARTBEAT_PATIENCE + 1
        )
        assert state.get_task_by_token(token) is None
        if active:
            assert (
                state.get_tasks(task_ids=[first])[0].status.sub_status
                == SubStatus.FAILED
            )
            assert state.claim_task(second)
        else:
            assert state.get_tasks(task_ids=[first])[0].status.status == Status.PENDING
            assert state.claim_task(second) is None
            assert state.claim_task(first)


@pytest.mark.parametrize("backend", ["memory", "sql"])
def test_concurrent_acceptance_and_claim(tmp_path: Path, backend: str) -> None:
    """Concurrent deliveries deduplicate; competing workers cannot skip FIFO."""
    if backend == "memory":
        first: LinkState = InMemoryLinkState(
            NoOpFederationManager(), ObjectStoreFactory().store()
        )
        second = first
    else:
        database = str(tmp_path / "shared.db")
        first = SqlLinkState(
            database, NoOpFederationManager(), ObjectStoreFactory().store()
        )
        second = SqlLinkState(
            database, NoOpFederationManager(), ObjectStoreFactory().store()
        )
        first.initialize()
        second.initialize()
    request = make_request(first)
    barrier = Barrier(2)

    def accept(state: LinkState) -> int:
        barrier.wait(timeout=10)
        return enqueue_agent_run(state, request)

    with ThreadPoolExecutor(2) as pool:
        runs = list(pool.map(accept, [first, second]))
    assert runs[0] == runs[1]
    assert len(first.get_tasks()) == 1
    head = primary_task(first, runs[0])
    tail = primary_task(first, enqueue_agent_run(first, follow_up(first, request)))

    def claim(item: tuple[LinkState, int]) -> str | None:
        barrier.wait(timeout=10)
        return item[0].claim_task(item[1])

    with ThreadPoolExecutor(2) as pool:
        tokens = list(pool.map(claim, [(first, head), (second, tail)]))
    assert tokens[0] and tokens[1] is None
    assert second.claim_task(head) is None


def test_queue_order_survives_sql_restart(tmp_path: Path) -> None:
    """Execute in message order even when queue entries arrive in reverse."""
    database = str(tmp_path / "durable.db")
    state = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory(database).store()
    )
    state.initialize()
    earlier_message = make_request(state)
    later_message = follow_up(state, earlier_message)
    first_run = enqueue_agent_run(state, later_message)
    second_run = enqueue_agent_run(state, earlier_message)
    head = primary_task(state, second_run)
    tail = primary_task(state, first_run)

    restarted = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory(database).store()
    )
    restarted.initialize()
    assert enqueue_agent_run(restarted, earlier_message) == second_run
    assert restarted.claim_task(tail) is None
    assert restarted.claim_task(head)
    assert read_prompt(restarted, first_run) == "Next prompt"
    assert read_prompt(restarted, second_run) == "First prompt"


def test_timestamp_ties_preserve_message_order(state: LinkState) -> None:
    """Tied timestamps still use message IDs rather than queue insertion order."""
    earlier = make_request(state)
    later = follow_up(state, earlier)
    fixed_now = now()
    with patch("datetime.datetime") as clock:
        clock.now.return_value = fixed_now
        later_task = primary_task(state, enqueue_agent_run(state, later))
        earlier_task = primary_task(state, enqueue_agent_run(state, earlier))
        assert state.claim_task(later_task) is None
        assert state.claim_task(earlier_task)
        assert state.claim_task(later_task) is None
        assert state.finish_task(earlier_task, SubStatus.STOPPED, "")
        assert state.claim_task(later_task)


def test_backdated_submission_waits_for_active_execution(state: LinkState) -> None:
    """Clock skew cannot let a newer submission overlap a claimed execution."""
    request = make_request(state)
    later = follow_up(state, request)
    fixed_now = now()
    with patch("datetime.datetime") as clock:
        clock.now.return_value = fixed_now
        head = primary_task(state, enqueue_agent_run(state, request))
        assert state.claim_task(head)
        clock.now.return_value = fixed_now - timedelta(seconds=10)
        tail = primary_task(state, enqueue_agent_run(state, later))
        assert state.claim_task(tail) is None
        assert state.finish_task(head, SubStatus.STOPPED, "")
        assert state.claim_task(tail)


def test_submission_creates_message_and_pending_runs(state: LinkState) -> None:
    """One acceptance publishes a message and records all pending executions."""
    space_id, targets = shared_targets(state)
    message, runs = submit_agent_message(
        state, space_id, "alice", "Both agents", targets
    )
    assert state.get_space_messages(space_id, "alice") == [message]
    assert len(runs) == 2
    for target, run_id in zip(targets, runs, strict=True):
        entry = state.get_agent_queue_entry(target.agent_id, message.id)
        assert entry is not None and entry.run_id == run_id
        run = state.get_run_info(run_ids=[run_id])[0]
        assert run.status.status == Status.PENDING
        assert (
            state.get_tasks(task_ids=[primary_task(state, run_id)])[0].status.status
            == Status.PENDING
        )
        assert read_prompt(state, run_id) == "Both agents"


def test_empty_targets_do_not_implicitly_trigger_participants(state: LinkState) -> None:
    """An explicit empty target list stores only the message."""
    request = make_request(state)
    message, runs = submit_agent_message(
        state, request.space_id, "alice", "Chat only", []
    )
    assert not runs and not state.get_tasks()
    assert (state.get_space_messages(request.space_id, "alice") or [])[-1] == message


@pytest.mark.parametrize("after_upload", [False, True])
def test_submission_rolls_back_every_target(
    state: LinkState, after_upload: bool
) -> None:
    """A later failure removes the message, all runs/tasks, queues, and objects."""
    space_id, targets = shared_targets(state)
    store_prompt = state._store_generated_message  # pylint: disable=protected-access
    uploads = 0

    def upload_then_fail(message: Message) -> None:
        nonlocal uploads
        store_prompt(message)
        uploads += 1
        if after_upload and uploads == 2:
            raise RuntimeError("after second upload")

    failing = (
        targets
        if after_upload
        else [targets[0], replace(targets[1], fab_hash="missing")]
    )
    with patch.object(state, "_store_generated_message", side_effect=upload_then_fail):
        with pytest.raises((ValueError, RuntimeError)):
            submit_agent_message(state, space_id, "alice", "Rollback", failing)
    assert uploads == (2 if after_upload else 1)
    assert not state.get_space_messages(space_id, "alice")
    assert not state.get_run_info() and not state.get_tasks()
    assert isinstance(state.object_store, InMemoryObjectStore)
    assert not state.object_store.store and not state.object_store.run_objects_mapping
    if isinstance(state, InMemoryLinkState):
        assert not state.agent_queue
    message, runs = submit_agent_message(state, space_id, "alice", "Retry", targets)
    assert state.get_space_messages(space_id, "alice") == [message]
    assert len(runs) == 2


@pytest.mark.parametrize("finished", [False, True])
def test_overtaken_assignment_is_rejected(state: LinkState, finished: bool) -> None:
    """A late assignment cannot insert work before an execution already started."""
    older = make_request(state)
    newer = follow_up(state, older)
    run_id = enqueue_agent_run(state, newer)
    task = primary_task(state, run_id)
    assert state.claim_task(task)
    if finished:
        assert state.finish_task(task, SubStatus.FAILED, "")
    with pytest.raises(ValueError, match="newer message"):
        enqueue_agent_run(state, older)
    assert state.get_agent_queue_entry(older.agent_id, older.message_id) is None
    assert len(state.get_tasks()) == 1
    assert enqueue_agent_run(state, newer) == run_id


# pylint: disable-next=too-many-locals
def test_concurrent_submission_is_atomic_and_ordered(
    tmp_path: Path,
) -> None:
    """An uncommitted message is invisible and cannot be overtaken by a replica."""
    database = str(tmp_path / "submissions.db")
    first = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory(database).store()
    )
    second = SqlLinkState(
        database, NoOpFederationManager(), ObjectStoreFactory(database).store()
    )
    first.initialize()
    second.initialize()
    request = make_request(first)
    targets = [run_target(request)]
    entered, release = Event(), Event()
    create_message = first.create_space_message

    def pause_after_insert(
        space_id: UUID, author: SpaceParticipant, content: str
    ) -> SpaceMessage | None:
        message = create_message(space_id, author, content)
        entered.set()
        assert release.wait(timeout=10)
        return message

    with patch.object(first, "create_space_message", side_effect=pause_after_insert):
        with ThreadPoolExecutor(2) as pool:
            earlier = pool.submit(
                submit_agent_message,
                first,
                request.space_id,
                "alice",
                "Earlier",
                targets,
            )
            try:
                assert entered.wait(timeout=10)
                history = second.get_space_messages(request.space_id, "alice")
                assert history is not None and len(history) == 1
                later = pool.submit(
                    submit_agent_message,
                    second,
                    request.space_id,
                    "alice",
                    "Later",
                    targets,
                )
                assert not later.done()
            finally:
                release.set()
            earlier_message, earlier_runs = earlier.result(timeout=10)
            later_message, later_runs = later.result(timeout=10)
    assert earlier_message.id < later_message.id
    head = primary_task(first, earlier_runs[0])
    tail = primary_task(second, later_runs[0])
    assert second.claim_task(tail) is None
    assert first.claim_task(head)
    assert second.claim_task(tail) is None
