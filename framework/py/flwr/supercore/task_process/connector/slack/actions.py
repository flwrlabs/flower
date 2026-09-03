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
"""Slack action definitions aligned with Open Connector."""

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition

SLACK_CONVERSATION_TYPES = ("public_channel", "private_channel", "im", "mpim")
_CHANNEL_ID: JSONObject = {
    "type": "string",
    "minLength": 1,
    "description": "The Slack conversation or channel ID.",
}
_MESSAGE: JSONObject = {
    "type": "object",
    "properties": {
        "ts": {"type": "string", "description": "The message timestamp identifier."},
        "userId": {
            "type": "string",
            "description": "The user ID of the message author.",
        },
        "text": {
            "type": "string",
            "description": "The text content of the message.",
        },
    },
    "additionalProperties": True,
    "description": "A Slack message record.",
}
_HAS_MORE: JSONObject = {
    "type": "boolean",
    "description": "Whether more messages are available beyond this page.",
}
_MESSAGES_OUTPUT: JSONObject = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "items": _MESSAGE,
            "description": "The list of messages in the conversation.",
        },
        "hasMore": _HAS_MORE,
    },
    "additionalProperties": False,
    "required": ["messages", "hasMore"],
    "description": "The output payload for this action.",
}
_SEARCH_MATCH: JSONObject = {
    "type": "object",
    "properties": {
        "matchId": {
            "type": "string",
            "description": "Slack's search result item identifier.",
        },
        "channelId": {
            "type": "string",
            "description": "The conversation identifier containing the message.",
        },
        "channelName": {
            "anyOf": [
                {
                    "type": "string",
                    "description": "The conversation name when Slack returns one.",
                },
                {"type": "null"},
            ]
        },
        "ts": {"type": "string", "description": "The message timestamp identifier."},
        "userId": {
            "type": "string",
            "description": "The user ID of the message author.",
        },
        "username": {
            "type": "string",
            "description": "The username of the message author when Slack returns one.",
        },
        "text": {"type": "string", "description": "The matching message text."},
        "permalink": {
            "type": "string",
            "description": "A Slack permalink for the matching message.",
        },
        "teamId": {
            "type": "string",
            "description": "The Slack team ID returned for the match.",
        },
        "type": {"type": "string", "description": "The Slack result type."},
    },
    "additionalProperties": True,
    "description": "A normalized Slack message search match.",
}
_CONVERSATION: JSONObject = {
    "type": "object",
    "properties": {
        "channelId": {
            "type": "string",
            "description": "The unique identifier of the conversation.",
        },
        "name": {
            "anyOf": [
                {
                    "type": "string",
                    "description": "The name of the conversation when available.",
                },
                {"type": "null"},
            ]
        },
        "type": {
            "type": "string",
            "enum": ["public_channel", "private_channel", "im", "mpim", "unknown"],
            "description": "The normalized Slack conversation type.",
        },
        "isArchived": {
            "anyOf": [
                {
                    "type": "boolean",
                    "description": "Whether the conversation is archived.",
                },
                {"type": "null"},
            ]
        },
        "isPrivate": {
            "anyOf": [
                {
                    "type": "boolean",
                    "description": "Whether the conversation is private.",
                },
                {"type": "null"},
            ]
        },
        "isMember": {
            "anyOf": [
                {
                    "type": "boolean",
                    "description": (
                        "Whether the connected Slack identity is a member."
                    ),
                },
                {"type": "null"},
            ]
        },
        "memberCount": {
            "type": "integer",
            "description": "The member count when Slack provides it.",
        },
        "topic": {
            "anyOf": [
                {"type": "string", "description": "The conversation topic."},
                {"type": "null"},
            ]
        },
        "purpose": {
            "anyOf": [
                {"type": "string", "description": "The conversation purpose."},
                {"type": "null"},
            ]
        },
        "userId": {
            "type": "string",
            "description": "The linked user identifier for IM conversations.",
        },
        "locale": {
            "type": "string",
            "description": "The locale returned by Slack when requested.",
        },
    },
    "additionalProperties": False,
    "required": [
        "channelId",
        "name",
        "type",
        "isArchived",
        "isPrivate",
        "isMember",
        "topic",
        "purpose",
    ],
    "description": "A normalized Slack conversation record.",
}

ACTIONS = (
    ActionDefinition(
        name="get_channel_messages",
        description="Get recent messages from a Slack conversation.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channelId": _CHANNEL_ID,
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "The maximum number of messages to return.",
                },
            },
            "additionalProperties": False,
            "required": ["channelId"],
            "description": "Input parameters for reading Slack conversation history.",
        },
        output_schema=_MESSAGES_OUTPUT,
    ),
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
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The Slack search query.",
                },
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "The number of results to return per page.",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "The Slack page number to fetch.",
                },
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
                "sortDir": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "The sort direction for Slack search results.",
                },
                "teamId": {
                    "type": "string",
                    "description": (
                        "The encoded team ID to search when using an org-level token."
                    ),
                },
            },
            "additionalProperties": False,
            "required": ["query"],
            "description": "Input parameters for searching Slack messages.",
        },
        output_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query Slack executed.",
                },
                "matches": {
                    "type": "array",
                    "items": _SEARCH_MATCH,
                    "description": "The matching Slack messages.",
                },
                "total": {
                    "type": "integer",
                    "description": "The total number of matches Slack reports.",
                },
                "pagination": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Slack pagination metadata when returned.",
                },
                "paging": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Slack legacy paging metadata when returned.",
                },
                "nextCursor": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": (
                                "The cursor for the next page when Slack returns one."
                            ),
                        },
                        {"type": "null"},
                    ]
                },
            },
            "additionalProperties": False,
            "required": [
                "query",
                "matches",
                "total",
                "pagination",
                "paging",
                "nextCursor",
            ],
            "description": "The output payload for this action.",
        },
    ),
    ActionDefinition(
        name="get_thread",
        description="Get messages in a Slack thread.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channelId": _CHANNEL_ID,
                "threadTs": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The timestamp of the parent message.",
                },
            },
            "additionalProperties": False,
            "required": ["channelId", "threadTs"],
            "description": "Input parameters for reading a Slack thread.",
        },
        output_schema={
            **_MESSAGES_OUTPUT,
            "properties": {
                "messages": {
                    "type": "array",
                    "items": _MESSAGE,
                    "description": "The list of messages in the thread.",
                },
                "hasMore": _HAS_MORE,
            },
        },
    ),
    ActionDefinition(
        name="list_conversations",
        description=(
            "List Slack conversations visible to the connected Slack identity."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "description": "The maximum number of conversations to return.",
                },
                "cursor": {
                    "type": "string",
                    "description": "The Slack pagination cursor.",
                },
                "types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": list(SLACK_CONVERSATION_TYPES),
                        "description": "A Slack conversation type.",
                    },
                    "description": "Conversation types to include.",
                    "minItems": 1,
                },
                "excludeArchived": {
                    "type": "boolean",
                    "description": (
                        "Whether archived conversations should be excluded."
                    ),
                },
            },
            "additionalProperties": False,
            "description": "Input parameters for listing Slack conversations.",
        },
        output_schema={
            "type": "object",
            "properties": {
                "conversations": {
                    "type": "array",
                    "items": _CONVERSATION,
                    "description": "The list of Slack conversations.",
                },
                "nextCursor": {
                    "anyOf": [
                        {
                            "type": "string",
                            "description": "The cursor for the next page.",
                        },
                        {"type": "null"},
                    ]
                },
            },
            "additionalProperties": False,
            "required": ["conversations", "nextCursor"],
            "description": "The output payload for this action.",
        },
    ),
)
