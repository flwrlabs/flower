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
"""Tests for Slack read tools backed by the Web API."""

from typing import cast
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from flwr.supercore.typing import JSONObject

from .. import registry
from ..definition import ActionAccess
from ..oauth import OAuthFlow
from .actions import ACTIONS
from .definition import PROVIDER, SLACK_CONNECTOR_REF, SLACK_USER_SCOPES
from .executors import SlackApiError

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_FILE_REQUEST = "flwr.supercore.task_process.connector.slack.executors.requests.get"
_OAUTH_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"
_IDENTITY_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.request"
_CREDENTIALS: JSONObject = {"access_token": "xoxp-secret"}


def _response(payload: object) -> Mock:
    """Return a successful HTTP response carrying a JSON payload."""
    response = Mock(status_code=200)
    response.json.return_value = payload
    return response


def test_slack_read_tool_definitions() -> None:
    """Expose only Slack actions with registered Web API executors."""
    tools = registry.get_connector_tools(SLACK_CONNECTOR_REF)
    assert [tool["name"] for tool in tools] == [
        "slack_search_public",
        "slack_search_public_and_private",
        "slack_list_conversations",
        "slack_get_conversation_history",
        "slack_get_conversation_replies",
    ]
    assert all(action.access is ActionAccess.READ for action in ACTIONS)


def test_slack_search_public_uses_web_api() -> None:
    """Public search should use the public channel filter."""
    response = _response({"ok": True, "results": {"messages": []}})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release", "content_types": "messages", "limit": 5},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result == response.json.return_value
    assert request.call_args.args == (
        "POST",
        "https://slack.com/api/assistant.search.context",
    )
    assert request.call_args.kwargs["json"] == {
        "query": "release",
        "content_types": ["messages"],
        "channel_types": ["public_channel"],
        "limit": 5,
        "include_context_messages": True,
    }


def test_slack_private_search_maps_mcp_options() -> None:
    """Translate the captured string options to Web API arrays and timestamps."""
    response = _response({"ok": True, "results": {"messages": []}})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        registry.invoke_connector(
            "slack_search_public_and_private",
            {
                "query": "launch",
                "channel_types": "private_channel,im",
                "content_types": "messages,files",
                "after": "1700000000",
                "include_context": False,
            },
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert request.call_args.kwargs["json"] == {
        "query": "launch",
        "content_types": ["messages", "files"],
        "channel_types": ["private_channel", "im"],
        "after": 1700000000,
        "include_context_messages": False,
    }


def test_slack_search_accepts_keywords_and_filters_without_query() -> None:
    """Pass lexical terms and filters to Real-time Search."""
    with patch(
        _HTTP_REQUEST, return_value=_response({"ok": True, "results": {}})
    ) as request:
        registry.invoke_connector(
            "slack_search_public",
            {
                "keywords": ["project", '"release plan"'],
                "filters": "in:<#C1>",
                "natural_language_query": "Where is the release plan?",
            },
            Mock(),
            _CREDENTIALS,
            {},
        )
    body = request.call_args.kwargs["json"]
    assert body["query"] == "Where is the release plan? in:<#C1>"
    assert body["term_clauses"] == ["project", '"release plan"']


def test_slack_search_keywords_work_in_fallback() -> None:
    """The legacy search fallback should keep the same lexical constraints."""
    responses = [
        _response({"ok": False, "error": "feature_not_enabled"}),
        _response({"ok": True, "messages": {"matches": []}, "files": {"matches": []}}),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        registry.invoke_connector(
            "slack_search_public",
            {"keywords": ["project", "release"], "filters": "in:<#C1>"},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert request.call_args.kwargs["params"]["query"] == ("project release in:<#C1>")


def test_slack_search_falls_back_to_standard_web_api() -> None:
    """A feature-gated search should still read public matches through search.all."""
    responses = [
        _response({"ok": False, "error": "feature_not_enabled"}),
        _response(
            {
                "ok": True,
                "messages": {
                    "matches": [
                        {
                            "ts": "1.0",
                            "text": "public",
                            "channel": {"id": "C1", "is_private": False},
                        },
                        {
                            "ts": "2.0",
                            "text": "private",
                            "channel": {"id": "G1"},
                        },
                    ]
                },
                "files": {
                    "matches": [
                        {"id": "F1", "title": "public", "channels": ["C1"]},
                        {"id": "F2", "title": "private", "groups": ["G1"]},
                    ]
                },
            }
        ),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        result = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_search_public",
                {"query": "release"},
                Mock(),
                _CREDENTIALS,
                {},
            ),
        )
    results = cast(JSONObject, result["results"])
    assert [
        item["content"] for item in cast(list[JSONObject], results["messages"])
    ] == ["public"]
    assert [item["file_id"] for item in cast(list[JSONObject], results["files"])] == [
        "F1"
    ]
    assert request.call_args.args == ("GET", "https://slack.com/api/search.all")


def test_slack_api_errors_include_code() -> None:
    """Slack's API errors should remain readable without exposing tokens."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response({"ok": False, "error": "missing_scope"}),
        ),
        pytest.raises(SlackApiError, match="missing_scope") as error,
    ):
        registry.invoke_connector(
            "slack_search_public", {"query": "release"}, Mock(), _CREDENTIALS, {}
        )
    assert "xoxp-secret" not in str(error.value)


def test_slack_oauth_flow_uses_web_api_user_token() -> None:
    """Request the Web API read scopes through Slack's user-token OAuth flow."""
    redirect_uri = "https://example.com/callback"
    flow = OAuthFlow(
        PROVIDER, client_id="client", client_secret="secret", redirect_uri=redirect_uri
    )
    url = flow.build_authorization_url(
        redirect_uri=redirect_uri, state="state", pkce_challenge="challenge"
    )
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.path == "/oauth/v2/authorize"
    assert query["user_scope"] == [",".join(SLACK_USER_SCOPES)]
    assert "resource" not in query
    assert "code_challenge" not in query

    response = _response(
        {
            "ok": True,
            "authed_user": {
                "id": "U1",
                "access_token": "xoxp-secret",
                "token_type": "user",
            },
        }
    )
    with (
        patch(_OAUTH_REQUEST, return_value=response) as post,
        patch(
            _IDENTITY_REQUEST,
            return_value=_response({"ok": True, "team": "Flower", "user": "alice"}),
        ),
    ):
        credentials, config = flow.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier="verifier"
        )
    assert credentials == {"access_token": "xoxp-secret", "token_type": "user"}
    assert config == {"display_name": "Slack · Flower / alice"}
    assert post.call_args.args == ("https://slack.com/api/oauth.v2.access",)


def test_slack_history_actions_forward_cursor() -> None:
    """Slack history actions should expose cursor pagination."""
    cases: tuple[tuple[str, JSONObject, dict[str, str]], ...] = (
        (
            "slack_get_conversation_history",
            {"channel_id": "C1", "cursor": "next", "limit": 15},
            {"channel": "C1", "cursor": "next", "limit": "15"},
        ),
        (
            "slack_get_conversation_replies",
            {
                "channel_id": "C1",
                "thread_ts": "1.0",
                "cursor": "next",
                "limit": 15,
            },
            {"channel": "C1", "ts": "1.0", "cursor": "next", "limit": "15"},
        ),
    )
    response = Mock(status_code=200)
    response.json.return_value = {"ok": True, "response_metadata": {}}
    for name, arguments, params in cases:
        with patch(_HTTP_REQUEST, return_value=response) as request:
            assert (
                registry.invoke_connector(
                    name, arguments, Mock(), {"access_token": "xoxp-secret"}, {}
                )
                == response.json.return_value
            )
        assert request.call_args.kwargs["params"] == params


def test_slack_list_conversations_limit() -> None:
    """Slack should apply its default when limit is omitted and accept up to 999."""
    response = Mock(status_code=200)
    response.json.return_value = {"ok": True, "channels": []}
    cases: tuple[tuple[JSONObject, str | None], ...] = (
        ({}, None),
        ({"limit": 999}, "999"),
    )
    for arguments, expected_limit in cases:
        with patch(_HTTP_REQUEST, return_value=response) as request:
            registry.invoke_connector(
                "slack_list_conversations",
                arguments,
                Mock(),
                {"access_token": "xoxp-secret"},
                {},
            )
        assert request.call_args.kwargs["params"].get("limit") == expected_limit
