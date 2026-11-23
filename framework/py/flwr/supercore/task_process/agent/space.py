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
"""Executor-bound AgentSpace implementation."""

from typing import cast

from flwr.agentapp import AgentEvents, AgentSpace
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    GetSpaceMessagesRequest,
    SendSpaceMessageRequest,
    SpaceMessage,
)
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_process.connector.tool_schema import (
    function_tool,
    string_property,
)
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps, strict_json_loads


def _message_to_json(message: SpaceMessage) -> JSONObject:
    """Serialize a Space message without exposing protobufs to AgentApps."""
    author: JSONObject = (
        {"agent_id": message.author.agent_id}
        if message.author.HasField("agent_id")
        else {"flwr_aid": message.author.flwr_aid}
    )
    return {
        "id": message.id,
        "space_id": message.space_id,
        "author": author,
        "created_at": message.created_at,
        "content": message.content,
    }


class RuntimeAgentSpace(AgentSpace):
    """Expose Run-bound Space reads and a model tool for publishing."""

    def __init__(self, stub: RuntimeHttpClient, events: AgentEvents) -> None:
        self._stub = stub
        self._events = events

    def get_messages(
        self,
        *,
        limit: int = 100,
        before_id: int | None = None,
        after_id: int | None = None,
        unread_only: bool = False,
    ) -> list[JSONObject]:
        """Read messages and advance the agent's read cursor."""
        response = self._stub.GetSpaceMessages(
            GetSpaceMessagesRequest(
                limit=limit,
                before_id=before_id,
                after_id=after_id,
                unread_only=unread_only,
            )
        )
        return [_message_to_json(message) for message in response.messages]

    def tools(self) -> list[JSONObject]:
        """Return the send_space_message tool schema."""
        return [
            function_tool(
                "send_space_message",
                "Send a message to the current Space as this agent.",
                properties={"content": string_property("Message text to publish.")},
                required=["content"],
                strict=True,
            )
        ]

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute a Space send and emit its function call and output events."""
        name = tool_call["name"]
        if name != "send_space_message":
            raise ValueError(f"Unsupported Space tool '{name}'.")
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            arguments = strict_json_loads(arguments)
        content = arguments.get("content") if isinstance(arguments, dict) else None
        if not isinstance(content, str):
            raise ValueError("Space message tool requires a string 'content' field.")
        call_id = cast(str, tool_call["call_id"])
        self._events.emit(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": strict_json_dumps(arguments, compact=True),
            }
        )
        response = self._stub.SendSpaceMessage(SendSpaceMessageRequest(content=content))
        output: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(
                _message_to_json(response.message), compact=True
            ),
        }
        self._events.emit(output)
        return output
