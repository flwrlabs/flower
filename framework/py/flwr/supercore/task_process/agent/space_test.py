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
"""Tests for the AgentApp Space capability."""

from unittest.mock import Mock

import pytest

from flwr.agentapp import AgentSpace
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    GetSpaceMessagesRequest,
    GetSpaceMessagesResponse,
    SendSpaceMessageRequest,
    SendSpaceMessageResponse,
    SpaceMessage,
    SpaceParticipant,
)
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_loads

from .space import RuntimeAgentSpace


def test_get_messages_forwards_filters_and_preserves_authors() -> None:
    """Read through the Runtime API with both user and agent authors."""
    stub = Mock()
    stub.GetSpaceMessages.return_value = GetSpaceMessagesResponse(
        messages=[
            SpaceMessage(id=1, author=SpaceParticipant(flwr_aid="alice"), content="Hi"),
            SpaceMessage(id=2, author=SpaceParticipant(agent_id=42), content="Hello"),
        ]
    )
    space: AgentSpace = RuntimeAgentSpace(stub, Mock())

    messages = space.get_messages(limit=2, before_id=3, after_id=0, unread_only=True)

    stub.GetSpaceMessages.assert_called_once_with(
        GetSpaceMessagesRequest(limit=2, before_id=3, after_id=0, unread_only=True)
    )
    assert [message["author"] for message in messages] == [
        {"flwr_aid": "alice"},
        {"agent_id": 42},
    ]
    assert [message["content"] for message in messages] == ["Hi", "Hello"]


@pytest.mark.parametrize("arguments", ['{"content":"Hello"}', {"content": "Hello"}])
def test_send_tool_publishes_and_emits_output(arguments: str | JSONObject) -> None:
    """Expose sending as a tool, using the server-assigned message identity."""
    stub, events = Mock(), Mock()
    stub.SendSpaceMessage.return_value = SendSpaceMessageResponse(
        message=SpaceMessage(
            id=7,
            space_id="space",
            author=SpaceParticipant(agent_id=42),
            created_at="2026-11-11T12:00:00+00:00",
            content="Hello",
        )
    )
    space = RuntimeAgentSpace(stub, events)
    assert [tool["name"] for tool in space.tools()] == ["send_space_message"]

    output = space.call(
        {"name": "send_space_message", "call_id": "call-1", "arguments": arguments}
    )

    stub.SendSpaceMessage.assert_called_once_with(
        SendSpaceMessageRequest(content="Hello")
    )
    assert output["type"] == "function_call_output" and output["call_id"] == "call-1"
    assert strict_json_loads(str(output["output"])) == {
        "id": 7,
        "space_id": "space",
        "author": {"agent_id": 42},
        "created_at": "2026-11-11T12:00:00+00:00",
        "content": "Hello",
    }
    assert events.emit.call_count == 2
    assert events.emit.call_args_list[0].args[0]["type"] == "function_call"
    events.emit.assert_called_with(output)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("get_space_messages", {}),
        ("send_space_message", []),
        ("send_space_message", {"content": 1}),
    ],
)
def test_invalid_tool_call_does_not_publish(name: str, arguments: JSONValue) -> None:
    """Reject unsupported tools and malformed content before calling Runtime."""
    stub = Mock()
    space = RuntimeAgentSpace(stub, Mock())
    tool_call: JSONObject = {"name": name, "call_id": "call-1", "arguments": arguments}

    with pytest.raises(ValueError):
        space.call(tool_call)

    stub.SendSpaceMessage.assert_not_called()
