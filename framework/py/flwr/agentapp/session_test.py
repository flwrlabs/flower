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
"""Tests for AgentSession."""


from __future__ import annotations

import json
from typing import Any

import pytest

from flwr.agentapp import AgentModelError, AgentModelTimeoutError, AgentSession
from flwr.common import ConfigRecord, Context, RecordDict
from flwr.common.serde import message_from_proto, message_to_proto
from flwr.common.typing import Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskResponse,
    PullTaskMessageResponse,
    PushConversationItemsResponse,
    PushRunEventsRequest,
    PushTaskMessageResponse,
)
from flwr.supercore.task_message import ModelTaskMessage, ModelTaskResultMessage

AGENT_TASK_ID = 101
MODEL_TASK_ID = 202
REQUEST_MESSAGE_ID = "request-message-id"


class FakeAgentStub:
    """Fake ServerAppIo stub for AgentSession tests."""

    def __init__(self, messages: list[Any] | None = None) -> None:
        self.messages = messages or []
        self.created_tasks: list[Any] = []
        self.pushed_messages: list[Any] = []
        self.pull_limits: list[int] = []
        self.run_events: list[tuple[str, dict[str, object]]] = []
        self.conversation_items: list[tuple[str, list[str]]] = []

    def CreateTask(self, request: Any) -> CreateTaskResponse:
        """Capture child task creation."""
        self.created_tasks.append(request)
        return CreateTaskResponse(task_id=MODEL_TASK_ID)

    def PushTaskMessage(self, request: Any) -> PushTaskMessageResponse:
        """Capture pushed task messages."""
        self.pushed_messages.append(request.message)
        return PushTaskMessageResponse(message_id=REQUEST_MESSAGE_ID)

    def PullTaskMessage(self, request: Any) -> PullTaskMessageResponse:
        """Return configured task messages."""
        self.pull_limits.append(request.limit)
        return PullTaskMessageResponse(messages=self.messages)

    def PushRunEvents(self, request: PushRunEventsRequest) -> object:
        """Capture run events."""
        for event in request.events:
            self.run_events.append((event.event, json.loads(event.data)))
        return object()

    def PushConversationItems(self, request: Any) -> PushConversationItemsResponse:
        """Capture conversation items."""
        self.conversation_items.append(
            (request.conversation_id, [item.item_json for item in request.items])
        )
        return PushConversationItemsResponse(item_indices=range(len(request.items)))


def test_agent_session_parses_start_state() -> None:
    """Test AgentSession parses agent.start state."""
    stub = FakeAgentStub()
    session = _session(stub)

    assert session.task_id == AGENT_TASK_ID
    assert session.agent_ref == "test-agent"
    assert session.conversation_id == "conv-1"
    assert session.invocation.input_items == [{"role": "user", "content": "hi"}]
    assert session.invocation.agent_ref == "test-agent"
    assert session.invocation.conversation_id == "conv-1"
    assert session.model.default_model == "gpt-4.1-mini"


def test_agent_session_emits_run_events() -> None:
    """Test AgentSession emits compact run events with base metadata."""
    stub = FakeAgentStub()
    session = _session(stub)

    session.emit_event("agent.custom", {"value": 1})

    assert stub.run_events == [
        (
            "agent.custom",
            {"task_id": AGENT_TASK_ID, "conversation_id": "conv-1", "value": 1},
        )
    ]


def test_conversation_client_adds_items() -> None:
    """Test conversation items are pushed as strict JSON payloads."""
    stub = FakeAgentStub()
    session = _session(stub)

    indices = session.conversation.add_items(
        [{"role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}]
    )

    assert indices == [0]
    assert stub.conversation_items == [
        (
            "conv-1",
            ['{"role":"assistant","content":[{"type":"output_text","text":"hi"}]}'],
        )
    ]


def test_model_client_creates_task_and_returns_result() -> None:
    """Test model response flow over task messages."""
    result_message = _model_result_proto(
        response={
            "id": "resp-1",
            "output": [{"type": "message", "content": "hello"}],
            "usage": {"input_tokens": 1},
        }
    )
    stub = FakeAgentStub([result_message])
    session = _session(stub)

    result = session.model.response(
        input_items=[{"role": "user", "content": "hi"}],
        stream=True,
        timeout=0.1,
        poll_interval=0.0,
    )

    assert stub.created_tasks[0].type == "flwr-model"
    assert stub.created_tasks[0].model_ref == "gpt-4.1-mini"
    pushed = ModelTaskMessage.from_message(message_from_proto(stub.pushed_messages[0]))
    assert pushed.dst_task_id == MODEL_TASK_ID
    assert pushed.payload["input"] == [{"role": "user", "content": "hi"}]
    assert result == {
        "id": "resp-1",
        "output": [{"type": "message", "content": "hello"}],
        "usage": {"input_tokens": 1},
    }
    assert stub.run_events == []


def test_model_client_raises_structured_model_errors() -> None:
    """Test model result errors become AgentModelError."""
    stub = FakeAgentStub(
        [
            _model_result_proto(
                response={"status": "failed"},
                error={"type": "provider_error", "message": "down"},
            )
        ]
    )
    session = _session(stub)

    with pytest.raises(AgentModelError) as err:
        session.model.response(
            input_items=[],
            timeout=0.1,
            poll_interval=0.0,
        )

    assert err.value.error == {"type": "provider_error", "message": "down"}


def test_model_client_times_out_without_matching_result() -> None:
    """Test model response polling times out without a matching result."""
    stub = FakeAgentStub([])
    session = _session(stub)

    with pytest.raises(AgentModelTimeoutError):
        session.model.response(
            input_items=[],
            timeout=0.001,
            poll_interval=0.0,
        )


def _session(stub: FakeAgentStub) -> AgentSession:
    """Create an AgentSession for tests."""
    return AgentSession.from_context(
        stub=stub,
        task_id=AGENT_TASK_ID,
        context=_context(),
        run=Run.create_empty(1),
        model_response_timeout=0.1,
        model_poll_interval=0.0,
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
                        "input_json": '[{"role":"user","content":"hi"}]',
                        "model": "gpt-4.1-mini",
                    }
                )
            }
        ),
        run_config={},
    )


def _model_result_proto(
    *,
    response: dict[str, object],
    error: dict[str, object] | None = None,
) -> Any:
    """Create a model result Message proto."""
    response_id = response.get("id")
    usage = response.get("usage")
    message = ModelTaskResultMessage.create(
        dst_task_id=AGENT_TASK_ID,
        response=response,  # type: ignore[arg-type]
        response_id=response_id if isinstance(response_id, str) else None,
        usage=usage if isinstance(usage, dict) else None,
        output=response.get("output"),  # type: ignore[arg-type]
        error=error,  # type: ignore[arg-type]
        reply_to_message_id=REQUEST_MESSAGE_ID,
    ).to_message()
    return message_to_proto(message)
