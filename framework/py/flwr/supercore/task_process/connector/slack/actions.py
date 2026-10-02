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
"""Model-facing Slack read actions grounded in Slack MCP tool definitions."""

from ..definition import ActionAccess, ActionDefinition

# Captured Slack MCP descriptions and input schemas, excluding user-specific lines.
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
                        "Number of results to return, up to a max of 20. Defaults to "
                        "20."
                    ),
                },
                "after": {
                    "type": "string",
                    "description": (
                        "Only messages after this Unix timestamp (inclusive)"
                    ),
                },
                "before": {
                    "type": "string",
                    "description": (
                        "Only messages before this Unix timestamp (inclusive)"
                    ),
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
                    "description": (
                        "Sort direction (default: 'desc'). Options: 'asc', 'desc'"
                    ),
                    "enum": ["asc", "desc"],
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail (default: 'detailed'). Options: 'detailed', "
                        "'concise'"
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
                "only_my_channels": {
                    "type": "boolean",
                    "description": (
                        "Limit results to public channels the user is a member of. Set "
                        "to true when the user asks to search only their own or joined "
                        "channels. Default: false."
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
            },
            "required": [],
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
                        "Number of results to return, up to a max of 20. Defaults to "
                        "20."
                    ),
                },
                "after": {
                    "type": "string",
                    "description": (
                        "Only messages after this Unix timestamp (inclusive)"
                    ),
                },
                "before": {
                    "type": "string",
                    "description": (
                        "Only messages before this Unix timestamp (inclusive)"
                    ),
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
                    "description": (
                        "Sort direction (default: 'desc'). Options: 'asc', 'desc'"
                    ),
                    "enum": ["asc", "desc"],
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail (default: 'detailed'). Options: 'detailed', "
                        "'concise'"
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
                "only_my_channels": {
                    "type": "boolean",
                    "description": (
                        "Limit results to channels the user is a member of. Set to "
                        "true when the user asks to search only their own or joined "
                        "channels. Default: false."
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
            },
            "required": [],
        },
    ),
    ActionDefinition(
        name="search_channels",
        description=(
            "Search for Slack channels by name or description. Returns "
            "channel names, IDs, topics, purposes, and archive status.\n"
            "\n"
            "Query tips: use terms matching channel names/descriptions (e.g., "
            '"engineering", "project alpha"). Names are typically lowercase '
            "with hyphens.\n"
            "\n"
            "Use slack_read_channel to read messages from a known channel. "
            "Use slack_search_public to search message content across "
            "channels.\n"
            "---\n"
            "Split your search query into 2 fields:\n"
            "\n"
            "1. `keywords` — Lexical terms that must appear in the channel's "
            "name or attributes. Each element should be a single word or "
            '"quoted phrase". All AND\'d.\n'
            "\n"
            "2. `natural_language_query` — User's question in conversational "
            "tone for semantic re-ranking. Preserve original phrasing; on "
            "follow-ups incorporate prior context.\n"
            "\n"
            "Require `natural_language_query` + `keywords`.\n"
            "\n"
            "\n"
            "Strategy: Use keywords for subject terms. If 0 results, broaden "
            "by simplifying keywords.\n"
            "---"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding channels",
                },
                "channel_types": {
                    "type": "string",
                    "description": (
                        "Comma-separated list of channel types to include in the "
                        "search. Defaults to public_channel. Mix and match channel "
                        "types by providing a comma-separated list of any combination "
                        "of public_channel, private_channel. Example: "
                        "public_channel,private_channel; Second Example: public_channel"
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
                        "Number of results to return, up to a max of 20. Defaults to "
                        "20."
                    ),
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail (default: 'detailed'). Options: 'detailed', "
                        "'concise'"
                    ),
                    "enum": ["detailed", "concise"],
                },
                "include_archived": {
                    "type": "boolean",
                    "description": "Include archived channels in the search results",
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
            },
            "required": [],
        },
    ),
    ActionDefinition(
        name="search_users",
        description=(
            "Search for Slack users by name, email, or profile attributes "
            "(department, role, title).\n"
            "\n"
            'Query syntax: full names ("John Smith"), partial names ("John"), '
            'emails ("john@company.com"), departments/roles ("engineering"), '
            'combinations ("John engineering"), exclusions ("engineering '
            '-intern"). Space-separated terms = AND.\n'
            "\n"
            "Use slack_read_user_profile for detailed info on a known user "
            "ID. Use slack_search_public with from: filter to find messages "
            "by a user.\n"
            "---\n"
            "Split your search query into 2 fields:\n"
            "\n"
            "1. `keywords` — Lexical terms that must appear in the user's "
            "name or attributes. Each element should be a single word or "
            '"quoted phrase". All AND\'d.\n'
            "\n"
            "2. `natural_language_query` — User's question in conversational "
            "tone for semantic re-ranking. Preserve original phrasing; on "
            "follow-ups incorporate prior context.\n"
            "\n"
            "Require `natural_language_query` + `keywords`.\n"
            "\n"
            "\n"
            "Strategy: Use keywords for subject terms. If 0 results, broaden "
            "by simplifying keywords.\n"
            "---"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query for finding users. Accepts names, email address, "
                        'and other attributes in profile\n\nExamples:\n  - "John '
                        'Smith" - exact name match\n  - john@company - find users '
                        "with john@company in email\n  - engineering -intern - users "
                        'with "engineering" but not "intern" in profile'
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
                        "Number of results to return, up to a max of 20. Defaults to "
                        "20."
                    ),
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail (default: 'detailed'). Options: 'detailed', "
                        "'concise'"
                    ),
                    "enum": ["detailed", "concise"],
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
            },
            "required": [],
        },
    ),
    ActionDefinition(
        name="read_channel",
        description=(
            "Reads messages from a Slack channel in reverse chronological "
            "order (newest first). To read DM history, use a user_id as "
            "channel_id. Read-only.\n"
            "\n"
            "Use slack_read_thread with message_ts to read thread replies. "
            "Use slack_search_channels to find a channel ID by name. Use "
            "slack_search_public to search across channels. If "
            "'channel_not_found', try slack_search_channels first.\n"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": (
                        "ID of the Channel, private group, or IM channel to fetch "
                        "history for. Can also be a user_id to read DM history."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Number of messages to return, between 1 and 100. Default "
                        "value is 100."
                    ),
                },
                "cursor": {
                    "type": "string",
                    "description": (
                        "Paginate through collections of data by setting the cursor "
                        "parameter to a next_cursor attribute returned by a previous "
                        "request"
                    ),
                },
                "latest": {
                    "type": "string",
                    "description": (
                        "End of time range of messages to include in results "
                        "(timestamp)"
                    ),
                },
                "oldest": {
                    "type": "string",
                    "description": (
                        "Start of time range of messages to include in results "
                        "(timestamp)"
                    ),
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail: 'detailed' (default, includes reactions + "
                        "thread info) or 'concise'."
                    ),
                },
            },
            "required": ["channel_id"],
        },
    ),
    ActionDefinition(
        name="read_thread",
        description=(
            "Reads messages from a specific Slack thread (parent message + "
            "all replies). Read-only.\n"
            "\n"
            "Requires channel_id and message_ts of the parent message. Use "
            "slack_search_public or slack_read_channel to find these values. "
            'Use slack_search_public with "is:thread" to find threads by '
            "content. Use slack_send_message with thread_ts to reply to a "
            "thread.\n"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": (
                        "Channel, private group, or IM channel to fetch thread replies "
                        "for"
                    ),
                },
                "message_ts": {
                    "type": "string",
                    "description": (
                        'Timestamp of the parent message (e.g. "1234567890.123456"). '
                        "Must be a string in Slack ts format with a decimal point."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Number of messages to return, between 1 and 1000. Default "
                        "value is 100."
                    ),
                },
                "cursor": {
                    "type": "string",
                    "description": (
                        "Paginate through collections of data by setting the cursor "
                        "parameter to a next_cursor attribute returned by a previous "
                        "request"
                    ),
                },
                "latest": {
                    "type": "string",
                    "description": (
                        "End of time range of messages to include in results. Slack ts "
                        'format string (e.g. "1234567890.123456").'
                    ),
                },
                "oldest": {
                    "type": "string",
                    "description": (
                        "Start of time range of messages to include in results. Slack "
                        'ts format string (e.g. "1234567890.123456").'
                    ),
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail: 'detailed' (default, includes reactions + "
                        "thread info) or 'concise'."
                    ),
                },
            },
            "required": ["channel_id", "message_ts"],
        },
    ),
    ActionDefinition(
        name="list_channel_members",
        description=(
            "Lists members of a Slack channel, group, or group DM (MPIM). "
            "Returns profile details or just user IDs.  Does not support "
            "DMs.\n"
            "\n"
            "Formats: 'detailed' (default) = full profile, 'concise' = "
            "@username + display name, 'ids_only' = user IDs only (fastest, "
            "skip profile fetch).\n"
            "Filters out deleted users and bots by default (use "
            "include_deleted/include_bots to include) for 'detailed' and "
            "'concise' formats. 'ids_only' returns all member IDs without "
            "filtering (no profile data is fetched). Returns up to 30 members "
            "per page (limit is capped at 30). Use cursor from "
            "pagination_info to fetch next page.\n"
            "\n"
            "Use slack_search_channels to find a channel ID first. Use "
            "slack_search_users to find users across the workspace. Use "
            "slack_read_user_profile for detailed info on a specific user.\n"
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "ID of the channel to list members from",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        "Number of members to return per page (default: 30, max: 30)"
                    ),
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from previous response",
                },
                "response_format": {
                    "type": "string",
                    "description": (
                        "Level of detail (default: 'detailed'). Options: 'detailed', "
                        "'concise', 'ids_only'"
                    ),
                    "enum": ["detailed", "concise", "ids_only"],
                },
                "include_deleted": {
                    "type": "boolean",
                    "description": (
                        "Include deleted/deactivated users in the member list "
                        "(default: false)"
                    ),
                },
                "include_bots": {
                    "type": "boolean",
                    "description": (
                        "Include bots and apps in the member list (default: false)"
                    ),
                },
            },
            "required": ["channel_id"],
        },
    ),
    ActionDefinition(
        name="list_user_channels",
        description=(
            "Lists channels the user is a member of. Supports public "
            "channels, private channels, DMs (im = 1-on-1 direct messages), "
            "and Group DMs (mpim = multi-party direct messages).\n"
            "\n"
            "types accepts a comma-separated list: public_channel, "
            "private_channel, im (DM), mpim (Group DM). Default: "
            '"public_channel,private_channel". To include DMs or Group DMs, '
            "add them explicitly (e.g., "
            'types="public_channel,private_channel,im,mpim" for all).\n'
            "\n"
            "Archived channels are included by default. Pass "
            "exclude_archived=true to hide them.\n"
            "\n"
            "DMs and Group DMs lack user-set names/topics. DMs show the other "
            "participant's display name; Group DMs show members.\n"
            "\n"
            "name_prefix is case-insensitive. cursor is ignored when "
            "name_prefix is set (prefix filtering scans pages internally).\n"
            "\n"
            "team_id restricts the results to a single workspace. On a "
            "multi-workspace (Grid) org, channel memberships are "
            "per-workspace, so pass team_id (an encoded workspace ID like "
            '"T012AB3C4") to list channels in a specific workspace; without '
            "it, results come from the user's default workspace only.\n"
            "\n"
            "Related: slack_search_channels (channels you're not in), "
            "slack_read_channel (read messages).\n"
            "\n"
            "Examples:\n"
            '\t- DMs only: slack_list_user_channels(types="im")\n'
            '\t- Group DMs only: slack_list_user_channels(types="mpim")\n'
            "\t- Private + DMs: "
            'slack_list_user_channels(types="private_channel,im")\n'
            "\t- All types: "
            'slack_list_user_channels(types="public_channel,private_channel,im,mpim")\n'
            '\t- Prefix filter: slack_list_user_channels(name_prefix="eng-")\n'
            '\t- IDs only: slack_list_user_channels(format="ids_only")\n'
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "types": {
                    "type": "string",
                    "description": (
                        "Comma-separated list of channel types to include. Valid "
                        "values: public_channel, private_channel, mpim, im. Default: "
                        '"public_channel,private_channel" (DMs and Group DMs are '
                        "excluded unless explicitly listed)."
                    ),
                },
                "name_prefix": {
                    "type": "string",
                    "description": (
                        "Filter channels whose name starts with this string "
                        "(case-insensitive)"
                    ),
                },
                "exclude_archived": {
                    "type": "boolean",
                    "description": "Exclude archived channels (default: false)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max channels to return (default: 50, max: 200)",
                },
                "cursor": {
                    "type": "string",
                    "description": (
                        "Pagination cursor from previous response. Ignored when "
                        "name_prefix is provided, because prefix filtering scans "
                        "multiple internal pages and cannot resume from a single "
                        "cursor."
                    ),
                },
                "format": {
                    "type": "string",
                    "description": (
                        "Output format: 'full' (default, all details), 'ids_only' "
                        "(just channel IDs), or 'names_only' (just channel names)"
                    ),
                },
                "team_id": {
                    "type": "string",
                    "description": (
                        'Encoded workspace ID (e.g. "T012AB3C4") to list channels '
                        "from. On a multi-workspace org, channel memberships are "
                        "per-workspace; without this, results come from the user's "
                        "default workspace only."
                    ),
                },
            },
            "required": [],
        },
    ),
)
