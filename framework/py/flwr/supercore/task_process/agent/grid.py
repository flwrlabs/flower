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
"""Executor-bound AgentGrid implementation."""


from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flwr.agentapp import AgentGrid
from flwr.supercore.task_process.connector.tool_schema import (
    function_tool,
    string_property,
)
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_loads

if TYPE_CHECKING:
    from .session import AgentRuntime


def _grid_tools() -> list[JSONObject]:
    """Return model-facing federation Grid tool schemas."""
    return [
        function_tool(
            "get_nodes",
            (
                "Return all available SuperNodes, or a random sample if requested. "
                "A SuperNode is a node in a federation that sits next to data, "
                "performs operations on it, and returns results."
            ),
            properties={
                "sample_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional maximum number of SuperNodes to return.",
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": string_property(
                            "Selected SuperNode uint64 ID as a decimal string."
                        ),
                        "description": "All or a random sample of available nodes.",
                    },
                    "num_available": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Total number of available SuperNodes.",
                    },
                },
                "required": ["node_ids", "num_available"],
                "additionalProperties": False,
            },
        ),
        function_tool(
            "push_messages",
            (
                "Send messages to SuperNodes and return one result per message in "
                "the same order. Pass accepted message IDs to pull_messages if "
                "replies are required."
            ),
            properties={
                "messages": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "dst_node_id": string_property(
                                "Destination SuperNode uint64 ID as a decimal string "
                                "to preserve precision."
                            ),
                            "payload": string_property("String payload to send."),
                            "reply_to_message_id": string_property(
                                "ID of the message being replied to. Required when "
                                "replying to another message; otherwise, this field "
                                "must not be set."
                            ),
                            "ttl": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "description": "Optional round-trip TTL in seconds.",
                            },
                        },
                        "required": ["dst_node_id", "payload"],
                        "additionalProperties": False,
                    },
                },
            },
            required=["messages"],
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "description": "One result per input message, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "message_id": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "Accepted message ID, or null if rejected."
                                    ),
                                },
                                "error": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "Failure reason, or null if accepted."
                                    ),
                                },
                            },
                            "required": ["message_id", "error"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["results"],
                "additionalProperties": False,
            },
        ),
        function_tool(
            "pull_messages",
            "Wait for replies to message IDs returned by push_messages.",
            properties={
                "message_ids": {
                    "type": "array",
                    "items": string_property("Message ID returned by push_messages."),
                    "minItems": 1,
                    "description": "Message IDs whose replies are awaited.",
                },
                "timeout": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 300,
                    "description": "Maximum wait in seconds; zero checks once.",
                },
            },
            required=["message_ids", "timeout"],
            output_schema={
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "message_id": string_property("Reply message ID."),
                                "reply_to_message_id": string_property(
                                    "ID of the message this replies to."
                                ),
                                "src_node_id": string_property(
                                    "Source SuperNode uint64 ID as a decimal string."
                                ),
                                "payload": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "Reply payload, or null for an error reply."
                                    ),
                                },
                                "error": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "Error reason, or null for a content reply."
                                    ),
                                },
                            },
                            "required": [
                                "message_id",
                                "reply_to_message_id",
                                "src_node_id",
                                "payload",
                                "error",
                            ],
                            "additionalProperties": False,
                        },
                        "description": "Replies received before the timeout.",
                    },
                    "pending_message_ids": {
                        "type": "array",
                        "items": string_property(
                            "Requested message ID with no reply before the timeout."
                        ),
                    },
                },
                "required": ["messages", "pending_message_ids"],
                "additionalProperties": False,
            },
        ),
    ]


class RuntimeAgentGrid(AgentGrid):
    """Expose selected Grid operations as model tools."""

    def __init__(self, agent_runtime: AgentRuntime) -> None:
        self._agent_runtime = agent_runtime

    def tools(self) -> list[JSONObject]:
        """Return model-facing Grid tool schemas."""
        return _grid_tools()

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute one Grid function_call and return a function_call_output item."""
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            arguments = strict_json_loads(arguments)
        name = cast(str, tool_call["name"])
        call_id = cast(str, tool_call["call_id"])
        arguments_obj = cast(JSONObject, arguments)
        return self._agent_runtime.call_grid_with_events(
            name=name,
            call_id=call_id,
            arguments=arguments_obj,
        )
