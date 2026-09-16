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

import random
import time
from typing import cast

from flwr.agentapp import AgentEvents, AgentGrid
from flwr.agentapp.constants import (
    AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY,
    AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY,
)
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid
from flwr.supercore.task_process.connector.tool_schema import (
    function_tool,
    string_property,
)
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps, strict_json_loads

_GRID_TOOL_NAMES = {"get_nodes", "push_messages", "pull_messages"}


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

    def __init__(self, grid: Grid, events: AgentEvents) -> None:
        self._grid = grid
        self._events = events

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
        if name not in _GRID_TOOL_NAMES:
            raise ValueError(f"Unsupported Grid tool '{name}'.")

        arguments_obj = cast(JSONObject, arguments)
        self._events.emit(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": strict_json_dumps(arguments_obj, compact=True),
            }
        )
        output = cast(JSONObject, getattr(self, f"_{name}")(**arguments_obj))
        output_item: JSONObject = {
            "type": "function_call_output",
            "call_id": call_id,
            "output": strict_json_dumps(output, compact=True),
        }
        self._events.emit(output_item)
        return output_item

    def _get_nodes(self, sample_size: int | None = None) -> JSONObject:
        node_ids = list(self._grid.get_node_ids())
        if sample_size is not None and sample_size < 1:
            raise ValueError("Grid sample size must be positive.")
        selected = (
            node_ids
            if sample_size is None
            else random.sample(node_ids, min(sample_size, len(node_ids)))
        )
        return {
            "node_ids": [str(node_id) for node_id in selected],
            "num_available": len(node_ids),
        }

    def _push_messages(self, messages: list[JSONObject]) -> JSONObject:
        if not messages:
            raise ValueError("At least one message is required.")

        outgoing = []
        for item in messages:
            ttl = cast(float | None, item.get("ttl"))
            if ttl is not None and ttl <= 0:
                raise ValueError("Grid message TTL must be positive.")
            config_record = ConfigRecord(
                {AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY: cast(str, item["payload"])}
            )
            content = RecordDict({AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY: config_record})

            message = Message(
                content,
                dst_node_id=int(cast(str, item["dst_node_id"])),
                message_type="query",  # Replace with an AgentGrid message type.
                group_id="",
                ttl=ttl,
            )
            outgoing.append(message)

        message_ids = list(self._grid.push_messages(outgoing))
        if len(message_ids) != len(outgoing):
            raise RuntimeError("Grid returned an unexpected number of message IDs.")
        return {
            "results": [
                {
                    "message_id": message_id or None,
                    "error": None if message_id else "Message was not accepted.",
                }
                for message_id in message_ids
            ]
        }

    def _pull_messages(self, message_ids: list[str], timeout: float) -> JSONObject:
        if not 0 <= timeout <= 300:
            raise ValueError("Grid pull timeout must be between 0 and 300 seconds.")
        pending = set(message_ids)
        replies: list[Message] = []
        deadline = time.monotonic() + timeout
        while pending:
            pulled = list(self._grid.pull_messages(pending))
            replies.extend(pulled)
            pending.difference_update(
                message.metadata.reply_to_message_id for message in pulled
            )
            remaining = deadline - time.monotonic()
            if not pending or remaining <= 0:
                break
            time.sleep(min(0.25, remaining))

        messages: list[JSONObject] = []
        for message in replies:
            payload = None
            error = None
            if message.has_error():
                error = message.error.reason
            else:
                payload = cast(
                    str,
                    message.content[AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY][
                        AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY
                    ],
                )
            messages.append(
                {
                    "message_id": message.metadata.message_id,
                    "reply_to_message_id": message.metadata.reply_to_message_id,
                    "src_node_id": str(message.metadata.src_node_id),
                    "payload": payload,
                    "error": error,
                }
            )

        return {
            "messages": messages,
            "pending_message_ids": sorted(pending),
        }
