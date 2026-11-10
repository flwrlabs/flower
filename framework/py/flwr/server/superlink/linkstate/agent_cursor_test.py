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
"""Cursor-based agent execution windows."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import cast
from unittest.mock import patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from flwr.common.constant import SubStatus
from flwr.proto.control_pb2 import AgentConfig  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    AcquireTaskRequest,
    GetCompactedContextsRequest,
    GetRunSeriesEventsRequest,
    PullTaskInputRequest,
)
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    GetSpaceMessagesRequest,
    GetSpaceMessagesResponse,
    SpaceParticipant,
)
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState, SqlLinkState
from flwr.supercore.constant import TASK_TOKEN_HEADER, TaskType
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.typing import AgentRunTarget
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.routers.runtime.router_test import _create_app
from flwr.superlink.servicer.runtime import runtime_handlers


@pytest.fixture(name="state", params=["memory", "sqlite"])
def agent_state(request: pytest.FixtureRequest, tmp_path: Path) -> LinkState:
    """Exercise both backends, using persistent SQLite for concurrent sessions."""
    objects = ObjectStoreFactory().store()
    if request.param == "memory":
        return InMemoryLinkState(NoOpFederationManager(), objects)
    state = SqlLinkState(str(tmp_path / "state.db"), NoOpFederationManager(), objects)
    state.initialize()
    return state


def _setup(state: LinkState) -> tuple[UUID, AgentRunTarget]:
    agent = state.create_agent(
        "alice", AgentConfig(name="Agent", app_spec="@publisher/app")
    )
    assert agent is not None
    space_id = UUID(state.get_spaces("alice")[0].id)
    fab_hash = state.store_fab(Fab("", b"test-fab", {}))
    return space_id, AgentRunTarget(agent.id, "publisher/app", "1.0", fab_hash)


def _start(state: LinkState, target: AgentRunTarget) -> tuple[int, str]:
    candidate = next(
        c for c in state.get_pending_agent_runs() if c.agent_id == target.agent_id
    )
    run_id = state.create_agent_run(candidate, target)
    assert run_id is not None
    task_id = state.get_run_info(run_ids=[run_id])[0].primary_task_id
    assert task_id is not None
    token = state.claim_task(task_id)
    assert token is not None
    assert state.activate_task(task_id)
    return task_id, token


def test_run_window_accepts_followups(state: LinkState) -> None:
    """History is read-only; consuming reads share one window and preserve order."""
    space_id, target = _setup(state)
    user = SpaceParticipant(flwr_aid="alice")
    first = state.create_space_message(space_id, user, "First")
    second = state.create_space_message(space_id, user, "Second")
    assert not state.get_run_info()  # Submission does not create a Run.
    task_id, token = _start(state, target)
    assert not state.get_pending_agent_runs()
    assert state.get_agent_space_messages(token, limit=1) == [second]
    assert state.get_agent_space_messages(token, consume=True, limit=1) == [first]
    third = state.create_space_message(space_id, user, "Follow-up")
    assert state.get_agent_space_messages(token, consume=True) == [second, third]
    assert state.get_agent_space_messages(token, consume=True) == []
    assert len(state.get_run_info()) == 1
    assert state.finish_task(task_id, SubStatus.COMPLETED, "")
    assert not state.get_pending_agent_runs()
    state.create_space_message(
        space_id, SpaceParticipant(agent_id=target.agent_id), "Reply"
    )
    assert not state.get_pending_agent_runs()  # Agent output does not trigger work.
    state.create_space_message(space_id, user, "Next window")
    _start(state, target)
    assert len(state.get_run_info()) == 2


@pytest.mark.parametrize("consume", [False, True])
def test_failure_reschedules_only_unseen_input(state: LinkState, consume: bool) -> None:
    """Seen is retrieval, not successful processing; stale tokens cannot advance."""
    space_id, target = _setup(state)
    state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), "Input")
    task_id, token = _start(state, target)
    if consume:
        assert state.get_agent_space_messages(token, consume=True)
    assert state.finish_task(task_id, SubStatus.FAILED, "Agent failed")
    with pytest.raises(PermissionError):
        state.get_agent_space_messages(token, consume=True)
    assert bool(state.get_pending_agent_runs()) is not consume


def test_broadcast_has_independent_agent_cursors(state: LinkState) -> None:
    """Each participating agent sees the same user message independently."""
    _, first = _setup(state)
    _, second = _setup(state)
    user = SpaceParticipant(flwr_aid="alice")
    space = state.create_space(
        participants=[
            user,
            SpaceParticipant(agent_id=first.agent_id),
            SpaceParticipant(agent_id=second.agent_id),
        ]
    )
    message = state.create_space_message(UUID(space.id), user, "For both agents")
    for target in (first, second):
        _, token = _start(state, target)
        assert state.get_agent_space_messages(token, consume=True) == [message]
    assert not state.get_pending_agent_runs()


def test_concurrent_schedulers_and_reads_serialize(state: LinkState) -> None:
    """Independent schedulers open one window and consume distinct ordered pages."""
    space_id, target = _setup(state)
    messages = [
        state.create_space_message(space_id, SpaceParticipant(flwr_aid="alice"), str(i))
        for i in range(2)
    ]
    candidate = state.get_pending_agent_runs()[0]
    other = state
    if isinstance(state, SqlLinkState):
        other = SqlLinkState(
            state.database_url, NoOpFederationManager(), ObjectStoreFactory().store()
        )
        other.initialize()
    barrier = Barrier(2)

    def create(backend: LinkState) -> int | None:
        barrier.wait(timeout=5)
        return backend.create_agent_run(candidate, target)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, (state, other)))
    assert sum(run_id is not None for run_id in results) == 1
    assert len(state.get_run_info()) == 1
    assert not state.get_pending_agent_runs()

    run = state.get_run_info()[0]
    assert run.primary_task_id is not None
    token = state.claim_task(run.primary_task_id)
    assert token is not None and state.activate_task(run.primary_task_id)

    def consume(backend: LinkState) -> object:
        barrier.wait(timeout=5)
        return backend.get_agent_space_messages(token, consume=True, limit=1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        pages = list(pool.map(consume, (state, other)))
    assert pages in ([messages[:1], messages[1:]], [messages[1:], messages[:1]])
    assert state.finish_task(run.primary_task_id, SubStatus.COMPLETED, "")
    assert (
        not other.get_pending_agent_runs()
    )  # The cursor is shared across SQL instances.


def test_runtime_consumes_with_task_identity(state: LinkState) -> None:
    """AcquireTask opens a window and Runtime authorizes consumption by task token."""
    space_id, _ = _setup(state)
    message = state.create_space_message(
        space_id, SpaceParticipant(flwr_aid="alice"), "Input"
    )
    config = {
        "project": {"name": "app", "version": "1.0"},
        "tool": {
            "flwr": {
                "app": {"publisher": "publisher", "components": {"agentapp": "app:app"}}
            }
        },
    }
    with (
        patch(
            "flwr.superlink.servicer.control.control_handlers._get_remote_fab",
            return_value=(b"test-fab", {}, ""),
        ),
        patch(
            "flwr.superlink.servicer.control.control_handlers.get_fab_config",
            return_value=config,
        ),
    ):
        acquired = runtime_handlers.acquire_task(
            AcquireTaskRequest(supported_task_types=[TaskType.AGENT_APP]), state
        )
    assert acquired.token
    runtime_handlers.pull_task_input(PullTaskInputRequest(), state, acquired.task)
    client = TestClient(_create_app(state))
    client.headers.update(
        {TASK_TOKEN_HEADER: acquired.token, "content-type": PROTOBUF_MEDIA_TYPE}
    )
    path = "/v1/runtime/get-space-messages"

    def post(request: GetSpaceMessagesRequest) -> Response:
        return cast(Response, client.post(path, content=request.SerializeToString()))

    denied = post(GetSpaceMessagesRequest(space_id=str(UUID(int=1)), consume=True))
    assert denied.status_code == 404
    invalid = post(GetSpaceMessagesRequest(consume=True, after_id=0))
    assert invalid.status_code == 400
    response = post(GetSpaceMessagesRequest(consume=True))
    assert response.status_code == 200
    assert GetSpaceMessagesResponse.FromString(response.content).messages == [message]
    # Existing AgentApp trace reads work without a RunSeries; summaries never use zero.
    assert not runtime_handlers.get_run_series_events(
        GetRunSeriesEventsRequest(), state, acquired.task
    ).events
    with pytest.raises(FlowerError) as error:
        runtime_handlers.get_compacted_contexts(
            GetCompactedContextsRequest(), state, acquired.task
        )
    assert error.value.code == ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE
    assert state.finish_task(acquired.task.task_id, SubStatus.COMPLETED, "")
    assert post(GetSpaceMessagesRequest(consume=True)).status_code == 401
