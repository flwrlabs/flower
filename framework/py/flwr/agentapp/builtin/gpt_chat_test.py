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

from dataclasses import dataclass, field
from typing import cast

import pytest

from flwr.agentapp import AgentApp, AgentModelError, AgentSession, DEFAULT_MODEL_NAME
from flwr.agentapp.builtin import gpt_chat
from flwr.supercore.task_message import JsonObject


class FakeModelClient:
    """Fake model client for built-in AgentApp tests."""

    def __init__(
        self,
        *,
        result: JsonObject | None = None,
        error: AgentModelError | None = None,
    ) -> None:
        self.result = result or {"id": "resp-1", "output_text": "Hello"}
        self.error = error
        self.calls: list[dict[str, object]] = []

    def response(
        self,
        *,
        model: str | None = None,
        input_items: list[JsonObject],
        stream: bool = True,
    ) -> JsonObject:
        """Capture the model request and return the configured result."""
        self.calls.append(
            {"model": model, "input_items": input_items, "stream": stream}
        )
        if self.error is not None:
            raise self.error
        return self.result


@dataclass(frozen=True)
class FakeInvocation:
    """Fake invocation object carrying input items."""

    input_items: list[JsonObject]


@dataclass
class FakeSession:
    """Fake session carrying only the fields used by gpt_chat."""

    invocation: FakeInvocation
    model: FakeModelClient = field(default_factory=FakeModelClient)


def test_gpt_chat_exports_agentapp() -> None:
    """Test the built-in module exposes an AgentApp."""
    assert isinstance(gpt_chat.app, AgentApp)


def test_gpt_chat_passes_invocation_input_items_to_model() -> None:
    """Test gpt-chat sends invocation input items to the model task."""
    input_items: list[JsonObject] = [{"role": "user", "content": "hello"}]
    session = FakeSession(invocation=FakeInvocation(input_items=input_items))

    result = gpt_chat.app(cast(AgentSession, session))

    assert result == {"id": "resp-1", "output_text": "Hello"}
    assert session.model.calls == [
        {
            "model": DEFAULT_MODEL_NAME,
            "input_items": input_items,
            "stream": True,
        }
    ]


def test_gpt_chat_propagates_model_errors() -> None:
    """Test gpt-chat leaves model errors to the AgentApp executor."""
    error = AgentModelError({"type": "provider_error", "message": "down"})
    session = FakeSession(
        invocation=FakeInvocation([{"role": "user", "content": "hello"}]),
        model=FakeModelClient(error=error),
    )

    with pytest.raises(AgentModelError) as err:
        gpt_chat.app(cast(AgentSession, session))

    assert err.value is error
