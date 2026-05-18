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
"""Tests for the built-in GPT chat AgentApp."""


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import pytest

from flwr.agentapp import AgentApp, AgentModelError, AgentModelResponse, AgentSession
from flwr.agentapp.builtin import gpt_chat
from flwr.supercore.task_message import JsonObject, JsonValue


class FakeModelClient:
    """Fake model client for built-in AgentApp tests."""

    def __init__(
        self,
        *,
        result: AgentModelResponse | None = None,
        error: AgentModelError | None = None,
        default_model: str = "gpt-4.1-mini",
    ) -> None:
        self.default_model = default_model
        self.result = result or _model_response()
        self.error = error
        self.calls: list[JsonObject] = []

    def response(
        self,
        *,
        input: JsonValue,  # pylint: disable=redefined-builtin
        model: str | None = None,
        stream: bool = True,
    ) -> AgentModelResponse:
        """Capture the model request and return the configured result."""
        self.calls.append({"input": input, "model": model, "stream": stream})
        if self.error is not None:
            raise self.error
        return self.result


class FakeConversationClient:
    """Fake conversation client for built-in AgentApp tests."""

    def __init__(self) -> None:
        self.items: list[JsonObject] = []

    def add_items(self, items: Sequence[JsonObject]) -> list[int]:
        """Capture conversation items."""
        self.items.extend(items)
        return list(range(len(items)))


@dataclass
class FakeSession:
    """Fake session carrying only the fields used by gpt_chat."""

    input: JsonValue
    model: FakeModelClient = field(default_factory=FakeModelClient)
    conversation: FakeConversationClient = field(default_factory=FakeConversationClient)
    events: list[tuple[str, JsonObject]] = field(default_factory=list)

    def emit_event(self, event: str, data: JsonObject) -> None:
        """Capture emitted events."""
        self.events.append((event, data))


def test_gpt_chat_exports_agentapp() -> None:
    """Test the built-in module exposes an AgentApp."""
    assert isinstance(gpt_chat.app, AgentApp)


def test_gpt_chat_passes_list_input_to_model() -> None:
    """Test gpt-chat sends list input to the model task."""
    session = FakeSession(input=[{"role": "user", "content": "hello"}])

    gpt_chat.app(cast(AgentSession, session))

    assert session.model.calls == [
        {
            "input": [{"role": "user", "content": "hello"}],
            "model": "gpt-4.1-mini",
            "stream": True,
        }
    ]
    assert session.conversation.items == [
        {
            "role": "assistant",
            "content": "Hello",
            "response_id": "resp-1",
            "model": "gpt-4.1-mini",
        }
    ]
    assert session.events == [
        (gpt_chat.GPT_CHAT_STARTED_EVENT, {"model": "gpt-4.1-mini"}),
        (
            gpt_chat.GPT_CHAT_COMPLETED_EVENT,
            {"model": "gpt-4.1-mini", "response_id": "resp-1"},
        ),
    ]


def test_gpt_chat_wraps_object_input() -> None:
    """Test gpt-chat wraps a single input object for the model task."""
    session = FakeSession(input={"role": "user", "content": "hello"})

    gpt_chat.app(cast(AgentSession, session))

    assert session.model.calls[0]["input"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "input_value",
    [
        "hello",
        [{"role": "user", "content": "hello"}, "invalid"],
    ],
)
def test_gpt_chat_rejects_non_object_input(input_value: JsonValue) -> None:
    """Test gpt-chat rejects input shapes that are not model input items."""
    session = FakeSession(input=input_value)

    with pytest.raises(ValueError, match="JSON object or list of JSON objects"):
        gpt_chat.app(cast(AgentSession, session))

    assert not session.model.calls
    assert not session.conversation.items
    assert not session.events


def test_gpt_chat_persists_full_response_when_output_text_is_empty() -> None:
    """Test gpt-chat stores the response payload when text extraction is empty."""
    response: JsonObject = {"id": "resp-1", "output": []}
    session = FakeSession(
        input=[{"role": "user", "content": "hello"}],
        model=FakeModelClient(result=_model_response(response)),
    )

    gpt_chat.app(cast(AgentSession, session))

    assert session.conversation.items == [
        {
            "role": "assistant",
            "content": "",
            "response_id": "resp-1",
            "model": "gpt-4.1-mini",
            "response": response,
        }
    ]


def test_gpt_chat_propagates_model_errors() -> None:
    """Test gpt-chat leaves model errors to the AgentApp executor."""
    error = AgentModelError({"type": "provider_error", "message": "down"})
    session = FakeSession(
        input=[{"role": "user", "content": "hello"}],
        model=FakeModelClient(error=error),
    )

    with pytest.raises(AgentModelError) as err:
        gpt_chat.app(cast(AgentSession, session))

    assert err.value is error
    assert not session.conversation.items
    assert session.events == [
        (gpt_chat.GPT_CHAT_STARTED_EVENT, {"model": "gpt-4.1-mini"})
    ]


def _model_response(response: JsonObject | None = None) -> AgentModelResponse:
    """Create a model response for tests."""
    return AgentModelResponse(
        response=response or {"id": "resp-1", "output_text": "Hello"},
        response_id="resp-1",
        output=None,
        usage=None,
        events=[],
    )
