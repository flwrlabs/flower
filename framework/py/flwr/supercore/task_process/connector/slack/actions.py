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
"""Slack action definitions."""

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import integer_property, string_property

SLACK_CONVERSATION_TYPES = ("public_channel", "private_channel", "mpim", "im")
_CURSOR = string_property("Cursor returned by the previous Slack response.")
_MESSAGE_LIMIT = integer_property(
    "Maximum number of messages to return.", minimum=1, maximum=15
)

ACTIONS = (
    ActionDefinition(
        name="search_messages",
        description="Search messages visible to the connected Slack user.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property("Slack message search query."),
                "limit": integer_property(
                    "Maximum number of matches to return.", minimum=1, maximum=15
                ),
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="list_conversations",
        description="List Slack channels and direct-message conversations.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "limit": integer_property(
                    "Maximum number of conversations to return.",
                    minimum=1,
                    maximum=50,
                ),
                "cursor": _CURSOR,
                "types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(SLACK_CONVERSATION_TYPES),
                    },
                    "minItems": 1,
                    "description": "Conversation types to include.",
                },
                "exclude_archived": {
                    "type": "boolean",
                    "description": "Whether to exclude archived conversations.",
                },
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_conversation_history",
        description="Read recent messages from one Slack conversation.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "conversation_id": string_property("Slack conversation ID."),
                "limit": _MESSAGE_LIMIT,
                "cursor": _CURSOR,
            },
            "required": ["conversation_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_thread_replies",
        description="Read a Slack thread's parent message and replies.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "conversation_id": string_property("Slack conversation ID."),
                "thread_ts": string_property(
                    "Timestamp of the thread's parent message."
                ),
                "limit": _MESSAGE_LIMIT,
                "cursor": _CURSOR,
            },
            "required": ["conversation_id", "thread_ts"],
            "additionalProperties": False,
        },
    ),
)
