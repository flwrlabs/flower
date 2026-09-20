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
"""Tests for the AgentApp process environment."""

import os
from unittest.mock import Mock

import pytest

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
)
from flwr.supercore.task_identity import TaskIdentity

from .run_agentapp import _set_runtime_environment, message_to_prompt, pull_prompt


@pytest.fixture(autouse=True)
def task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the task identity required to construct test messages."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 123)
    monkeypatch.setattr(TaskIdentity, "_run_id", 456)
    monkeypatch.setattr(TaskIdentity, "_node_id", 789)


def _payload_message(src_node_id: int, message_type: str = "query") -> Message:
    """Build a Grid message carrying a JSON payload record."""
    message = Message(
        RecordDict(
            {
                AGENT_MESSAGE_CONTENT_RECORD_KEY: ConfigRecord(
                    {AGENT_MESSAGE_TEXT_KEY: "hello world!"}
                )
            }
        ),
        dst_node_id=0,
        message_type=message_type,
    )
    message.metadata.__dict__["_message_id"] = "message-1"
    message.metadata.__dict__["_src_node_id"] = src_node_id
    return message


@pytest.mark.parametrize(("insecure", "scheme"), [(True, "http"), (False, "https")])
def test_set_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, insecure: bool, scheme: str
) -> None:
    """Expose the Runtime Responses base URL and AgentApp task token."""
    monkeypatch.delenv("FLWR_RUNTIME_BASE_URL", raising=False)
    monkeypatch.delenv("FLWR_RUNTIME_API_KEY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    _set_runtime_environment(
        "runtime.example:9092",
        "task-token",
        insecure=insecure,
        root_certificates_path="/path/to/runtime-ca.pem",
    )

    assert os.environ["FLWR_RUNTIME_BASE_URL"] == (
        f"{scheme}://runtime.example:9092/v1/runtime"
    )
    assert os.environ["FLWR_RUNTIME_API_KEY"] == "task-token"
    assert os.environ["SSL_CERT_FILE"] == "/path/to/runtime-ca.pem"


@pytest.mark.parametrize(
    ("message_type", "msg_src_node_id", "expected"),
    [
        (SYSTEM_MESSAGE_TYPE, 789, "hello world!"),
        (
            "query",
            789,
            '{"message_id":"message-1","src_node_id":"789","payload":"hello world!"}',
        ),
        (
            "query",
            99,
            '{"message_id":"message-1","src_node_id":"99","payload":"hello world!"}',
        ),
    ],
)
def test_message_to_prompt(
    message_type: str, msg_src_node_id: int, expected: str
) -> None:
    """Return system instructions as text and other messages as JSON."""
    assert (
        message_to_prompt(_payload_message(msg_src_node_id, message_type)) == expected
    )


def test_pull_prompt_requires_instruction() -> None:
    """Fail when the run has no initial instruction."""
    grid = Mock()
    grid.pull_messages.return_value = []

    with pytest.raises(RuntimeError, match="exactly one"):
        pull_prompt(grid)
    grid.pull_messages.assert_called_once_with([])


def test_pull_prompt_serializes_instruction() -> None:
    """Pull and serialize the run's initial instruction."""
    grid = Mock()
    grid.pull_messages.return_value = [_payload_message(789, SYSTEM_MESSAGE_TYPE)]

    assert pull_prompt(grid) == "hello world!"


def test_pull_prompt_rejects_multiple_instructions() -> None:
    """Reject ambiguous initial instructions for the singular prompt API."""
    grid = Mock()
    grid.pull_messages.return_value = [
        _payload_message(789),
        _payload_message(789),
    ]

    with pytest.raises(RuntimeError, match="exactly one"):
        pull_prompt(grid)
