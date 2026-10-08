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
"""Execute Slack-shaped read tools through the Slack Web API."""

from __future__ import annotations

from typing import cast

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from .actions import SLACK_CONVERSATION_TYPES

_SLACK_API_BASE_URL = "https://slack.com/api"


class SlackApiError(ConnectorApiError):
    """Secret-safe Slack Web API failure."""

    provider = "Slack"


def _call_slack_api(
    method: str,
    credentials: JSONObject,
    *,
    params: dict[str, str] | None = None,
    body: JSONObject | None = None,
) -> JSONObject:
    """Call one Slack Web API method and validate its response."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise SlackApiError("invalid_credentials")
    payload = request_json_object(
        "POST" if body is not None else "GET",
        f"{_SLACK_API_BASE_URL}/{method}",
        error=SlackApiError,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        json=body,
    )
    if payload.get("ok") is not True:
        raise SlackApiError(cast(str, payload.get("error", "api_error")))
    return payload


def _search(
    arguments: JSONObject,
    context: ConnectorExecutionContext,
    *,
    channel_types: tuple[str, ...] = SLACK_CONVERSATION_TYPES,
) -> JSONObject:
    """Map search arguments to one Slack Real-time Search request."""
    query = cast(str, arguments.get("query", ""))
    terms = cast(list[str], arguments.get("keywords", []))
    filters = cast(str, arguments.get("filters", ""))
    natural_language_query = cast(str, arguments.get("natural_language_query", ""))
    query_parts = (
        [natural_language_query, query, filters]
        if natural_language_query
        else [query, *terms, filters]
    )
    payload: JSONObject = {
        "query": " ".join(part for part in query_parts if part),
        "content_types": cast(str, arguments.get("content_types", "messages")).split(
            ","
        ),
        "channel_types": list(channel_types),
        "include_bots": arguments.get("include_bots", False),
        "include_context_messages": arguments.get("include_context", True),
    }
    if channel_types != ("public_channel",) and "channel_types" in arguments:
        payload["channel_types"] = cast(str, arguments["channel_types"]).split(",")
    if terms:
        payload["term_clauses"] = terms
        if filters:
            payload["modifiers"] = filters
    for name in ("context_channel_id", "cursor", "sort", "sort_dir", "limit"):
        if name in arguments:
            payload[name] = arguments[name]
    for name in ("after", "before"):
        if name in arguments:
            payload[name] = int(cast(str, arguments[name]))
    result = _call_slack_api(
        "assistant.search.context", context.credentials, body=payload
    )
    if arguments.get("only_my_channels"):
        _filter_joined_channels(
            result, context.credentials, cast(list[str], payload["channel_types"])
        )
    if (
        "max_context_length" in arguments
        or arguments.get("response_format") == "concise"
    ):
        _format_search_result(result, arguments)
    return result


def _filter_joined_channels(
    result: JSONObject, credentials: JSONObject, channel_types: list[str]
) -> None:
    """Keep messages and files shared in conversations the user has joined."""
    joined: set[str] = set()
    params = {"types": ",".join(channel_types), "limit": "200"}
    while True:
        page = _call_slack_api("users.conversations", credentials, params=params)
        joined.update(
            cast(str, channel["id"])
            for channel in cast(list[JSONObject], page["channels"])
        )
        cursor = cast(JSONObject, page.get("response_metadata", {})).get("next_cursor")
        if not cursor:
            break
        params["cursor"] = cast(str, cursor)
    results = cast(dict[str, list[JSONObject]], result["results"])
    if "messages" in results:
        results["messages"] = [
            item for item in results["messages"] if item["channel_id"] in joined
        ]
    if "files" in results:
        files = []
        for item in results["files"]:
            info = _call_slack_api(
                "files.info", credentials, params={"file": cast(str, item["file_id"])}
            )
            file = cast(JSONObject, info["file"])
            if any(
                joined.intersection(cast(list[str], file.get(field, [])))
                for field in ("channels", "groups", "ims")
            ):
                files.append(item)
        results["files"] = files


def _format_search_result(result: JSONObject, arguments: JSONObject) -> None:
    """Apply the requested context length and concise result fields."""
    results = cast(dict[str, list[JSONObject]], result["results"])
    if "max_context_length" in arguments:
        maximum = cast(int, arguments["max_context_length"])
        for message in results.get("messages", []):
            context = cast(
                dict[str, list[JSONObject]], message.get("context_messages", {})
            )
            for direction in ("before", "after"):
                for item in context.get(direction, []):
                    item["text"] = cast(str, item["text"])[:maximum]
    if arguments.get("response_format") == "concise":
        fields = {
            "messages": (
                "channel_id",
                "message_ts",
                "author_user_id",
                "content",
                "permalink",
            ),
            "files": ("file_id", "title", "content", "permalink"),
        }
        for kind, names in fields.items():
            if kind in results:
                results[kind] = [
                    {key: item[key] for key in names if key in item}
                    for item in results[kind]
                ]


def search_public(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search public-channel messages and files."""
    return _search(
        arguments,
        context,
        channel_types=("public_channel",),
    )


def search_public_and_private(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search visible messages and files in all conversation types."""
    return _search(arguments, context)


def list_conversations(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List conversations visible to the connected Slack user."""
    types = cast(list[str], arguments.get("types", list(SLACK_CONVERSATION_TYPES)))
    params = {"types": ",".join(types)}
    for name in ("cursor", "team_id"):
        if name in arguments:
            params[name] = cast(str, arguments[name])
    if "limit" in arguments:
        params["limit"] = str(arguments["limit"])
    if "exclude_archived" in arguments:
        params["exclude_archived"] = str(arguments["exclude_archived"]).lower()
    return _call_slack_api("conversations.list", context.credentials, params=params)


def get_conversation_history(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get recent messages from a Slack conversation."""
    params = {"channel": cast(str, arguments["channel_id"])}
    if "cursor" in arguments:
        params["cursor"] = cast(str, arguments["cursor"])
    if "limit" in arguments:
        params["limit"] = str(arguments["limit"])
    return _call_slack_api("conversations.history", context.credentials, params=params)


def get_conversation_replies(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get messages in a Slack thread."""
    params = {
        "channel": cast(str, arguments["channel_id"]),
        "ts": cast(str, arguments["thread_ts"]),
    }
    if "cursor" in arguments:
        params["cursor"] = cast(str, arguments["cursor"])
    if "limit" in arguments:
        params["limit"] = str(arguments["limit"])
    return _call_slack_api("conversations.replies", context.credentials, params=params)


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_public": search_public,
    "search_public_and_private": search_public_and_private,
    "list_conversations": list_conversations,
    "get_conversation_history": get_conversation_history,
    "get_conversation_replies": get_conversation_replies,
}
