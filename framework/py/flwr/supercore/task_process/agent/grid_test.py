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
"""Runtime AgentGrid tests."""


from unittest.mock import Mock

import pytest

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
)
from flwr.supercore.task_identity import TaskIdentity

from .grid import RuntimeAgentGrid


@pytest.fixture(autouse=True)
def task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the task identity used by Agent Grid messages."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 123)
    monkeypatch.setattr(TaskIdentity, "_run_id", 456)
    monkeypatch.setattr(TaskIdentity, "_node_id", 789)


def test_runtime_agent_grid_tools() -> None:
    """Grid tools should sample nodes, send content, and return serialized replies."""
    grid = Mock()
    grid.get_node_ids.return_value = [11, 22]
    grid.push_messages.return_value = ["message-1", ""]
    reply = Message(
        RecordDict(
            {
                AGENT_MESSAGE_CONTENT_RECORD_KEY: ConfigRecord(
                    {AGENT_MESSAGE_TEXT_KEY: "done"}
                )
            }
        ),
        dst_node_id=0,
        message_type="query",
    )
    reply.metadata.__dict__["_message_id"] = "reply-1"
    reply.metadata.__dict__["_src_node_id"] = 11
    reply.metadata.__dict__["_reply_to_message_id"] = "message-1"
    grid.pull_messages.return_value = [reply]
    events = Mock()
    agent_grid = RuntimeAgentGrid(grid, events)

    assert [tool["name"] for tool in agent_grid.tools()] == [
        "get_nodes",
        "push_messages",
        "pull_messages",
    ]
    all_nodes = agent_grid.call(
        {"name": "get_nodes", "call_id": "call-0", "arguments": {}}
    )
    assert all_nodes["output"] == '{"node_ids":["11","22"],"num_available":2}'

    get_nodes = agent_grid.call(
        {
            "name": "get_nodes",
            "call_id": "call-1",
            "arguments": '{"sample_size":1}',
        }
    )
    assert get_nodes["output"] in (
        '{"node_ids":["11"],"num_available":2}',
        '{"node_ids":["22"],"num_available":2}',
    )

    pushed = agent_grid.call(
        {
            "name": "push_messages",
            "call_id": "call-2",
            "arguments": {
                "messages": [
                    {
                        "dst_node_id": "11",
                        "payload": "hi",
                    },
                    {
                        "dst_node_id": "22",
                        "payload": "hello",
                        "reply_to_message_id": "message-0",
                    },
                ]
            },
        }
    )
    assert pushed["output"] == (
        '{"results":[{"message_id":"message-1","error":null},'
        '{"message_id":null,"error":"Message was not accepted."}]}'
    )
    grid.create_message.assert_not_called()
    sent, second = list(grid.push_messages.call_args.args[0])
    assert sent.metadata.dst_node_id == 11
    assert second.metadata.dst_node_id == 22
    assert sent.metadata.message_type == "query"
    assert sent.metadata.group_id == ""
    assert sent.metadata.reply_to_message_id == ""
    assert second.metadata.reply_to_message_id == "message-0"
    assert (
        sent.content[AGENT_MESSAGE_CONTENT_RECORD_KEY][AGENT_MESSAGE_TEXT_KEY] == "hi"
    )

    pulled = agent_grid.call(
        {
            "name": "pull_messages",
            "call_id": "call-3",
            "arguments": {"message_ids": ["message-1"], "timeout": 0},
        }
    )
    assert pulled["output"] == (
        '{"messages":[{"message_id":"reply-1",'
        '"reply_to_message_id":"message-1","src_node_id":"11",'
        '"payload":"done","error":null}],"pending_message_ids":[]}'
    )
    assert events.emit.call_count == 8
