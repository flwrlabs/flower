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

import pytest

from flwr.agentapp import AgentAppError, AgentSession
from flwr.common import ConfigRecord, ConfigRecordValues, Context, RecordDict
from flwr.common.typing import Run


def test_agent_session_parses_start_state() -> None:
    """Test AgentSession parses agent.start state."""
    session = AgentSession.from_context(
        task_id=101,
        run=Run.create_empty(1),
        context=_context(
            {
                "agent_ref": "test-agent",
                "conversation_id": "conv-1",
                "input_json": '[{"role":"user","content":"hi"}]',
            }
        ),
    )

    assert session.task_id == 101
    assert session.run.run_id == 1
    assert session.agent_ref == "test-agent"
    assert session.conversation_id == "conv-1"
    assert session.input_items == [{"role": "user", "content": "hi"}]


def test_agent_session_normalizes_single_input_object() -> None:
    """Test AgentSession accepts one JSON object as input."""
    session = AgentSession.from_context(
        task_id=101,
        run=Run.create_empty(1),
        context=_context(
            {
                "agent_ref": "test-agent",
                "conversation_id": "conv-1",
                "input_json": '{"role":"user","content":"hi"}',
            }
        ),
    )

    assert session.input_items == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "record, expected",
    [
        ({}, "requires `agent_ref`"),
        (
            {
                "agent_ref": "test-agent",
                "conversation_id": "conv-1",
            },
            "requires `input_json`",
        ),
        (
            {
                "agent_ref": "test-agent",
                "input_json": "[]",
            },
            "requires `conversation_id`",
        ),
        (
            {
                "agent_ref": "test-agent",
                "conversation_id": "conv-1",
                "input_json": "not-json",
            },
            "must contain valid JSON",
        ),
        (
            {
                "agent_ref": "test-agent",
                "conversation_id": "conv-1",
                "input_json": '["not-object"]',
            },
            "JSON object or list of JSON objects",
        ),
    ],
)
def test_agent_session_rejects_invalid_start_state(
    record: dict[str, ConfigRecordValues], expected: str
) -> None:
    """Test invalid agent.start state raises AgentAppError."""
    with pytest.raises(AgentAppError, match=expected):
        AgentSession.from_context(
            task_id=101,
            run=Run.create_empty(1),
            context=_context(record),
        )


def test_agent_session_rejects_missing_start_record() -> None:
    """Test AgentSession requires an agent.start record."""
    context = Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config={},
    )

    with pytest.raises(AgentAppError, match="missing `agent.start`"):
        AgentSession.from_context(
            task_id=101,
            run=Run.create_empty(1),
            context=context,
        )


def test_agent_session_rejects_invalid_task_id() -> None:
    """Test AgentSession requires a positive task ID."""
    with pytest.raises(ValueError, match="task_id"):
        AgentSession.from_context(
            task_id=0,
            run=Run.create_empty(1),
            context=_context(
                {
                    "agent_ref": "test-agent",
                    "conversation_id": "conv-1",
                    "input_json": "[]",
                }
            ),
        )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"agent_ref": ""}, "`agent_ref` must be a non-empty string"),
        ({"conversation_id": ""}, "`conversation_id` must be a non-empty string"),
        (
            {"input_items": ["not-object"]},
            "`input_items` must be a list of JSON objects",
        ),
    ],
)
def test_agent_session_rejects_invalid_constructor_arguments(
    kwargs: dict[str, object], expected: str
) -> None:
    """Test AgentSession validates its public constructor arguments."""
    values = {
        "task_id": 101,
        "run": Run.create_empty(1),
        "agent_ref": "test-agent",
        "conversation_id": "conv-1",
        "input_items": [],
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=expected):
        AgentSession(**values)  # type: ignore[arg-type]


def _context(record: dict[str, ConfigRecordValues]) -> Context:
    """Create a Context carrying agent.start state."""
    return Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict({"agent.start": ConfigRecord(record)}),
        run_config={},
    )
