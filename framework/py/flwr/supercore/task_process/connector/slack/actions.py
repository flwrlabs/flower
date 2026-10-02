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

_SEARCH_PROPERTIES: JSONObject = {
    "context_channel_id": {
        "type": "string",
        "description": (
            "Context channel ID to support boosting the search results for "
            "a channel when applicable"
        ),
    },
    "cursor": {
        "type": "string",
        "description": (
            "The cursor returned by the API. Leave this blank for the "
            "first request, and use this to get the next page of results"
        ),
    },
    "limit": {
        "type": "integer",
        "description": (
            "Number of results to return, up to a max of 20. Defaults to 20."
        ),
    },
    "after": {
        "type": "string",
        "description": ("Only messages after this Unix timestamp (inclusive)"),
    },
    "before": {
        "type": "string",
        "description": ("Only messages before this Unix timestamp (inclusive)"),
    },
    "include_bots": {
        "type": "boolean",
        "description": "Include bot messages (default: false)",
    },
    "sort": {
        "type": "string",
        "description": (
            "Sort order: 'score' (relevance, default) or 'timestamp' "
            "(newest first unless sort_dir is 'asc')."
        ),
        "enum": ["score", "timestamp"],
    },
    "sort_dir": {
        "type": "string",
        "description": ("Sort direction (default: 'desc'). Options: 'asc', 'desc'"),
        "enum": ["asc", "desc"],
    },
    "response_format": {
        "type": "string",
        "description": (
            "Level of detail (default: 'detailed'). Options: 'detailed', 'concise'"
        ),
        "enum": ["detailed", "concise"],
    },
    "include_context": {
        "type": "boolean",
        "description": (
            "Include surrounding context messages for each result "
            "(default: true). Set to false to reduce response size."
        ),
    },
    "max_context_length": {
        "type": "integer",
        "description": (
            "Max character length for each context message. Longer "
            "messages are truncated."
        ),
    },
    "keywords": {
        "type": "array",
        "description": (
            "Array of lexical search terms. Each element MUST be a single "
            "word (no spaces) OR an exact phrase in quotes. All elements "
            "are AND'd (every element must match). Use the author test: "
            "only include words the author would naturally write."
        ),
        "items": {
            "type": "string",
        },
    },
    "filters": {
        "type": "string",
        "description": (
            "Slack search modifiers (e.g., 'from:<@U123> in:<#C456> "
            "after:2025-01-01'). Use for people, channels, dates, and "
            "content type constraints."
        ),
    },
    "natural_language_query": {
        "type": "string",
        "description": (
            "The user's question restated in conversational tone. Used for "
            "semantic reranking. Do not include filter-like content "
            "(people, channels, dates) — those belong in filters. Pass an "
            "empty string when the query is purely structural (only "
            "filters, no semantic question)."
        ),
    },
}

ACTIONS = (
    ActionDefinition(
        name="search_public",
        description=(
            "Searches for messages, files in public Slack channels ONLY. \n"
            "`slack_search_public` does NOT generally require user consent "
            "for use, whereas you should request and wait for user consent to "
            "use `slack_search_public_and_private`.\n"
            "\n"
            "---\n"
            "Split your search query into 3 fields:\n"
            "\n"
            "1. `keywords` — Lexical terms that must appear in content. Each "
            'element should be a single word or "quoted phrase". All AND\'d. '
            "Use nouns identifying subject matter. People/channels/dates go "
            "in filters.\n"
            "\n"
            "2. `filters` — Slack search modifiers to constrain results:\n"
            "   in:<#C123456> | in:@username | from:<@U123456> | "
            "from:username | with:<@U123456> | creator:@user\n"
            "   has:pin | has:link | has:file | has:reaction | has::emoji: | "
            "hasmy::emoji: | is:thread | is:saved | is:dm\n"
            "   before:YYYY-MM-DD | after:YYYY-MM-DD | on:YYYY-MM-DD | "
            "during:month\n"
            "   Same modifier repeated = OR (except with/has = AND).\n"
            "\n"
            "3. `natural_language_query` —  User's question in conversational "
            "tone. Preserve original phrasing; on follow-ups incorporate "
            'prior context. Don\'t include filter-like content. Pass "" for '
            "filter-only queries with no semantic content.\n"
            "\n"
            "Require at least one of `keywords` or `filters`.\n"
            "\n"
            "<examples>\n"
            "User: What's the latest on Project Unicorn?\n"
            '> keywords: ["Project", "Unicorn"], natural_language_query: '
            "What's the latest on Project Unicorn?\n"
            "\n"
            "User: What did <@U0123456ABC> talk about last week?\n"
            "> keywords: [], filters: from:<@U0123456ABC> after:2025-06-12\n"
            "\n"
            "User: Find the budget spreadsheet shared in <#C024BE7LR>\n"
            '> keywords: ["\\"budget spreadsheet\\""], filters: '
            "in:<#C024BE7LR>\n"
            "> natural_language_query: Where is the budget spreadsheet shared "
            "in <#C024BE7LR>?\n"
            "</examples>\n"
            "\n"
            "Strategy: Decompose complex requests into parallel searches. Use "
            "keywords for subject terms, filters for people/channels/dates. "
            "If 0 results, broaden by removing filters or simplifying "
            "keywords.\n"
            "---"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                **_SEARCH_PROPERTIES,
                "query": {
                    "type": "string",
                    "description": (
                        "Search query (e.g., 'bug report', 'from:<@U0123456ABC> "
                        "in:dev')"
                    ),
                },
                "content_types": {
                    "type": "string",
                    "description": (
                        "Content types to include, a comma-separated list of any "
                        "combination of messages, files. Here's more info about the "
                        "content types: messages: Slack messages from public channels "
                        "accessible to the acting user\nfiles: Files of all types "
                        "accessible to the acting user\n"
                    ),
                },
                "only_my_channels": {
                    "type": "boolean",
                    "description": (
                        "Limit results to public channels the user is a member of. Set "
                        "to true when the user asks to search only their own or joined "
                        "channels. Default: false."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="search_public_and_private",
        description=(
            "Searches for messages, files in ALL Slack channels, including "
            "public channels, private channels, DMs, and group DMs. \n"
            "---\n"
            "Split your search query into 3 fields:\n"
            "\n"
            "1. `keywords` — Lexical terms that must appear in content. Each "
            'element should be a single word or "quoted phrase". All AND\'d. '
            "Use nouns identifying subject matter. People/channels/dates go "
            "in filters.\n"
            "\n"
            "2. `filters` — Slack search modifiers to constrain results:\n"
            "   in:<#C123456> | in:@username | from:<@U123456> | "
            "from:username | with:<@U123456> | creator:@user\n"
            "   has:pin | has:link | has:file | has:reaction | has::emoji: | "
            "hasmy::emoji: | is:thread | is:saved | is:dm\n"
            "   before:YYYY-MM-DD | after:YYYY-MM-DD | on:YYYY-MM-DD | "
            "during:month\n"
            "   Same modifier repeated = OR (except with/has = AND).\n"
            "\n"
            "3. `natural_language_query` —  User's question in conversational "
            "tone. Preserve original phrasing; on follow-ups incorporate "
            'prior context. Don\'t include filter-like content. Pass "" for '
            "filter-only queries with no semantic content.\n"
            "\n"
            "Require at least one of `keywords` or `filters`.\n"
            "\n"
            "<examples>\n"
            "User: What's the latest on Project Unicorn?\n"
            '> keywords: ["Project", "Unicorn"], natural_language_query: '
            "What's the latest on Project Unicorn?\n"
            "\n"
            "User: What did <@U0123456ABC> talk about last week?\n"
            "> keywords: [], filters: from:<@U0123456ABC> after:2025-06-12\n"
            "\n"
            "User: Find the budget spreadsheet shared in <#C024BE7LR>\n"
            '> keywords: ["\\"budget spreadsheet\\""], filters: '
            "in:<#C024BE7LR>\n"
            "> natural_language_query: Where is the budget spreadsheet shared "
            "in <#C024BE7LR>?\n"
            "</examples>\n"
            "\n"
            "Strategy: Decompose complex requests into parallel searches. Use "
            "keywords for subject terms, filters for people/channels/dates. "
            "If 0 results, broaden by removing filters or simplifying "
            "keywords.\n"
            "---"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                **_SEARCH_PROPERTIES,
                "query": {
                    "type": "string",
                    "description": (
                        "Search query using Slack's search syntax (e.g., 'in:#general "
                        "from:@user important')"
                    ),
                },
                "channel_types": {
                    "type": "string",
                    "description": (
                        "Comma-separated list of channel types to include in the "
                        "search. Defaults to 'public_channel,private_channel,mpim,im' "
                        "(all channel types including private channels, group DMs, and "
                        "DMs). Mix and match channel types by providing a "
                        "comma-separated list of any combination of `public_channel`, "
                        "`private_channel`, `mpim`, `im`"
                    ),
                },
                "content_types": {
                    "type": "string",
                    "description": (
                        "Content types to include, a comma-separated list of any "
                        "combination of messages, files. Here's more info about the "
                        "content types: messages: Slack messages from channels "
                        "accessible to the acting user\nfiles: Files of all types "
                        "accessible to the acting user\n"
                    ),
                },
                "only_my_channels": {
                    "type": "boolean",
                    "description": (
                        "Limit results to channels the user is a member of. Set to "
                        "true when the user asks to search only their own or joined "
                        "channels. Default: false."
                    ),
                },
            },
            "required": [],
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
