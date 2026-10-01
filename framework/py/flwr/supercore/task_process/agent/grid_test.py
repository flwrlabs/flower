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


from typing import cast
from unittest.mock import Mock

import pytest

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
)
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.typing import JSONObject

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
    grid.get_nodes.return_value = [
        NodeInfo(node_id=11, name="London", location="51.5072,-0.1276"),
        NodeInfo(node_id=22),
    ]
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
    agent_grid = RuntimeAgentGrid(grid, events, SUPERLINK_NODE_ID)

    tools = agent_grid.tools()
    assert [tool["name"] for tool in tools] == [
        "get_nodes",
        "push_messages",
        "pull_messages",
    ]
    assert all(tool["strict"] is True for tool in tools)
    all_nodes = agent_grid.call(
        {
            "name": "get_nodes",
            "call_id": "call-0",
            "arguments": {"sample_size": None},
        }
    )
    assert all_nodes["output"] == (
        '{"nodes":[{"id":"11","name":"London","location":"51.5072,-0.1276"},'
        '{"id":"22","name":null,"location":null}],"num_available":2}'
    )

    get_nodes = agent_grid.call(
        {
            "name": "get_nodes",
            "call_id": "call-1",
            "arguments": '{"sample_size":1}',
        }
    )
    assert get_nodes["output"] in (
        '{"nodes":[{"id":"11","name":"London",'
        '"location":"51.5072,-0.1276"}],"num_available":2}',
        '{"nodes":[{"id":"22","name":null,"location":null}],"num_available":2}',
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
                        "reply_to_message_id": None,
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
    assert second.metadata.ttl == 21600
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


def test_supernode_agent_grid_only_pushes_one_reply() -> None:
    """Reply using the received instruction's IDs, not model-supplied IDs."""
    grid = Mock()
    grid.push_messages.return_value = ["reply-1"]
    instruction = Message(
        RecordDict(
            {
                AGENT_MESSAGE_CONTENT_RECORD_KEY: ConfigRecord(
                    {AGENT_MESSAGE_TEXT_KEY: "request"}
                )
            }
        ),
        dst_node_id=789,
        message_type="query",
    )
    instruction.metadata.__dict__["_message_id"] = "request-1"
    instruction.metadata.__dict__["_src_node_id"] = 11
    agent_grid = RuntimeAgentGrid(
        grid, Mock(), node_id=789, instruction_metadata=instruction.metadata
    )

    tools = agent_grid.tools()
    assert [tool["name"] for tool in tools] == ["push_reply_message"]
    parameters = cast(JSONObject, tools[0]["parameters"])
    assert parameters["required"] == ["payload"]
    assert list(cast(JSONObject, parameters["properties"])) == ["payload"]
    with pytest.raises(ValueError, match="Unsupported Grid tool 'push_messages'"):
        agent_grid.call({"name": "push_messages", "call_id": "call-0", "arguments": {}})
    grid.push_messages.assert_not_called()

    pushed = agent_grid.call(
        {
            "name": "push_reply_message",
            "call_id": "call-1",
            "arguments": {"payload": "done"},
        }
    )

    assert pushed["output"] == '{"message_id":"reply-1","error":null}'
    grid.push_messages.assert_called_once()
    sent = list(grid.push_messages.call_args.args[0])
    assert len(sent) == 1
    assert sent[0].metadata.dst_node_id == 11
    assert sent[0].metadata.reply_to_message_id == "request-1"
    assert (
        sent[0].content[AGENT_MESSAGE_CONTENT_RECORD_KEY][AGENT_MESSAGE_TEXT_KEY]
        == "done"
    )

    repeated = agent_grid.call(
        {
            "name": "push_reply_message",
            "call_id": "call-2",
            "arguments": {"payload": "again"},
        }
    )
    assert repeated["output"] == (
        '{"message_id":null,"error":"No instruction message to reply to. '
        'You may have already replied to it once."}'
    )
    grid.push_messages.assert_called_once()
