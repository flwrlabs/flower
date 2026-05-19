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
"""Tests for the `flwr-agentapp` executor."""


from __future__ import annotations

import json
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.common import ConfigRecord, Context, RecordDict
from flwr.common.constant import SubStatus
from flwr.common.typing import Fab, Run
from flwr.proto.appio_pb2 import (
    PullTaskInputResponse,
    PushConversationItemsResponse,
    PushRunEventsRequest,
    PushTaskOutputRequest,
)
from flwr.supercore.executors.run_agentapp import (
    AGENT_COMPLETED_EVENT,
    AGENT_FAILED_EVENT,
    AGENT_STARTED_EVENT,
    AgentAppTaskInput,
    _run_agentapp_task,
)

TASK_ID = 101


class FakeAgentAppExecutorStub:
    """Fake ServerAppIo stub for AgentApp executor tests."""

    def __init__(self) -> None:
        self.run_events: list[tuple[str, dict[str, object]]] = []
        self.conversation_items: list[tuple[str, list[str]]] = []
        self.task_outputs: list[PushTaskOutputRequest] = []

    def PullTaskInput(self, request: object) -> PullTaskInputResponse:
        """Satisfy the executor stub protocol."""
        del request
        return PullTaskInputResponse(task_id=TASK_ID)

    def PushRunEvents(self, request: PushRunEventsRequest) -> object:
        """Capture run events."""
        for event in request.events:
            self.run_events.append((event.event, json.loads(event.data)))
        return object()

    def PushTaskOutput(self, request: PushTaskOutputRequest) -> object:
        """Capture task output."""
        self.task_outputs.append(request)
        return object()

    def PushConversationItems(self, request: Any) -> PushConversationItemsResponse:
        """Capture conversation items."""
        self.conversation_items.append(
            (
                request.conversation_id,
                [item.item_json for item in request.items],
            )
        )
        return PushConversationItemsResponse(
            item_indices=range(len(request.items))
        )


def test_run_agentapp_task_completes_successfully() -> None:
    """Test AgentApp task success flow."""
    stub = FakeAgentAppExecutorStub()
    seen_sessions: list[AgentSession] = []
    app = AgentApp()

    @app.main()
    def main(session: AgentSession) -> dict[str, object]:
        seen_sessions.append(session)
        return {
            "id": "resp-1",
            "output_text": "hello",
            "model": "gpt-4.1-mini",
        }

    completed = _run_agentapp_task(
        stub,
        _task_input(),
        lambda fab_id, fab_version, fab_hash: app,
    )

    assert completed
    assert seen_sessions[0].task_id == TASK_ID
    assert [event for event, _ in stub.run_events] == [
        AGENT_STARTED_EVENT,
        AGENT_COMPLETED_EVENT,
    ]
    assert stub.conversation_items == [
        (
            "conv-1",
            [
                (
                    '{"role":"assistant","content":"hello","response_id":"resp-1",'
                    '"model":"gpt-4.1-mini"}'
                )
            ],
        )
    ]
    assert stub.task_outputs[-1].sub_status == SubStatus.COMPLETED
    assert stub.task_outputs[-1].details == ""


def test_run_agentapp_task_fails_on_agent_exception() -> None:
    """Test AgentApp task failure flow."""
    stub = FakeAgentAppExecutorStub()
    app = AgentApp()

    @app.main()
    def main(_: AgentSession) -> dict[str, object]:
        raise ValueError("boom")

    completed = _run_agentapp_task(
        stub,
        _task_input(),
        lambda fab_id, fab_version, fab_hash: app,
    )

    assert not completed
    assert [event for event, _ in stub.run_events] == [
        AGENT_STARTED_EVENT,
        AGENT_FAILED_EVENT,
    ]
    assert stub.run_events[-1][1]["error"] == {
        "type": "ValueError",
        "message": "boom",
    }
    assert stub.task_outputs[-1].sub_status == SubStatus.FAILED
    assert stub.task_outputs[-1].details == "ValueError: boom"


def _task_input() -> AgentAppTaskInput:
    """Create AgentApp task input for tests."""
    run = Run.create_empty(1)
    run.fab_id = "test-agent"
    run.fab_version = "0.1.0"
    run.fab_hash = "fab-hash"
    return AgentAppTaskInput(
        task_id=TASK_ID,
        context=_context(),
        run=run,
        fab=Fab("fab-hash", b"fab-content", {}),
    )


def _context() -> Context:
    """Create a Context carrying agent.start state."""
    return Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(
            {
                "agent.start": ConfigRecord(
                    {
                        "agent_ref": "test-agent",
                        "conversation_id": "conv-1",
                        "input_json": "[]",
                        "model": "gpt-4.1-mini",
                    }
                )
            }
        ),
        run_config={},
    )
