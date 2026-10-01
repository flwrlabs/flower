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

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import integer_property, string_property

SLACK_CONVERSATION_TYPES = ("public_channel", "private_channel", "mpim", "im")
SLACK_LIST_CONVERSATIONS_MAX_LIMIT = 999
SLACK_MESSAGE_MAX_LIMIT = 15
SLACK_SEARCH_MAXIMUM = 100
_CURSOR: JSONObject = {
    "type": "string",
    "description": "The Slack pagination cursor.",
}
_MESSAGE_LIMIT = integer_property(
    "The maximum number of messages to return.",
    minimum=1,
    maximum=SLACK_MESSAGE_MAX_LIMIT,
)

ACTIONS = (
    ActionDefinition(
        name="search_messages",
        description=(
            "Search Slack messages visible to the connected user. Supports Slack "
            "search modifiers such as in:channel_name and from:<@UserID>."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property("The Slack search query."),
                "count": integer_property(
                    "The number of results to return per page.",
                    minimum=1,
                    maximum=SLACK_SEARCH_MAXIMUM,
                ),
                "page": integer_property(
                    "The Slack page number to fetch.",
                    minimum=1,
                    maximum=SLACK_SEARCH_MAXIMUM,
                ),
                "cursor": {
                    "type": "string",
                    "description": (
                        "The Slack cursor for cursormark pagination. Use '*' for the "
                        "first request."
                    ),
                },
                "highlight": {
                    "type": "boolean",
                    "description": (
                        "Whether Slack should mark query terms in matching text."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["score", "timestamp"],
                    "description": "How Slack should sort search results.",
                },
                "sort_dir": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "The sort direction for Slack search results.",
                },
                "team_id": {
                    "type": "string",
                    "description": (
                        "The encoded team ID to search when using an org-level token."
                    ),
                },
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
                    "Maximum number of conversations to return. Omit to use "
                    "Slack's default of 100.",
                    minimum=1,
                    maximum=SLACK_LIST_CONVERSATIONS_MAX_LIMIT,
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
                "team_id": {
                    "type": "string",
                    "description": (
                        "The encoded team ID to list. Required when using an "
                        "org-level token; omit when using a workspace-level token."
                    ),
                },
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_conversation_history",
        description="Get recent messages from a Slack conversation.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": string_property("The Slack conversation or channel ID."),
                "limit": _MESSAGE_LIMIT,
                "cursor": _CURSOR,
            },
            "required": ["channel_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_conversation_replies",
        description="Get messages in a Slack thread.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": string_property("The Slack conversation or channel ID."),
                "thread_ts": string_property("The timestamp of the parent message."),
                "limit": _MESSAGE_LIMIT,
                "cursor": _CURSOR,
            },
            "required": ["channel_id", "thread_ts"],
            "additionalProperties": False,
        },
    ),
)
