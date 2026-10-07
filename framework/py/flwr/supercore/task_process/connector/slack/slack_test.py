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
        "include_bots": False,
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
                "before": "1700003599",
                "cursor": "next",
                "context_channel_id": "C1",
                "sort": "timestamp",
                "sort_dir": "asc",
                "limit": 8,
                "include_bots": True,
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
        "before": 1700003599,
        "cursor": "next",
        "context_channel_id": "C1",
        "sort": "timestamp",
        "sort_dir": "asc",
        "limit": 8,
        "include_bots": True,
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
    assert body["term_clauses"] == ['project "release plan"']
    assert body["modifiers"] == "in:<#C1>"


@pytest.mark.parametrize(
    "action_name", ("slack_search_public", "slack_search_public_and_private")
)
@pytest.mark.parametrize("natural_language_query", ("", "Where is the release plan?"))
def test_slack_search_preserves_keywords_within_clause_limit(
    action_name: str,
    natural_language_query: str,
) -> None:
    """Keep every lexical term and exact phrase without exceeding API limits."""
    keywords = ["alpha", '"release plan"', "gamma", "delta", "epsilon", "zeta"]
    expected_clause = 'alpha "release plan" gamma delta epsilon zeta'
    response = _response({"ok": True, "results": {"messages": []}})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            action_name,
            {
                "query": "has:link",
                "keywords": keywords,
                "filters": "in:<#C1>",
                "natural_language_query": natural_language_query,
            },
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result == response.json.return_value
    request.assert_called_once()
    assert request.call_args.args == (
        "POST",
        "https://slack.com/api/assistant.search.context",
    )
    body = request.call_args.kwargs["json"]
    if natural_language_query:
        assert body["query"] == "Where is the release plan? has:link in:<#C1>"
    else:
        assert body["query"] == f"has:link {expected_clause} in:<#C1>"
    assert body["term_clauses"] == [expected_clause]
    assert body["modifiers"] == "in:<#C1>"


def test_slack_search_only_my_channels_keeps_shared_files() -> None:
    """Check file shares instead of dropping files with no search channel ID."""
    message = {"channel_id": "C1", "message_ts": "1", "content": "joined"}
    file = {"file_id": "F1", "title": "joined file"}
    responses = [
        _response(
            {
                "ok": True,
                "results": {
                    "messages": [message, {"channel_id": "C2"}],
                    "files": [file, {"file_id": "F2"}],
                },
                "response_metadata": {"next_cursor": "search-next"},
            }
        ),
        _response(
            {
                "ok": True,
                "channels": [{"id": "C1"}],
                "response_metadata": {"next_cursor": "membership-next"},
            }
        ),
        _response({"ok": True, "channels": [{"id": "C3"}]}),
        _response({"ok": True, "file": {"channels": ["C1"]}}),
        _response({"ok": True, "file": {"channels": ["C2"]}}),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release", "only_my_channels": True},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result == {
        "ok": True,
        "results": {"messages": [message], "files": [file]},
        "response_metadata": {"next_cursor": "search-next"},
    }
    assert request.call_args_list[1].kwargs["params"] == {
        "types": "public_channel",
        "limit": "200",
    }
    assert request.call_args_list[2].kwargs["params"]["cursor"] == "membership-next"
    assert [call.kwargs["params"] for call in request.call_args_list[3:]] == [
        {"file": "F1"},
        {"file": "F2"},
    ]


@pytest.mark.parametrize("maximum", (0, 4, 100_001))
def test_slack_search_truncates_context_to_requested_length(maximum: int) -> None:
    """Apply the captured context limit without inventing an upper bound."""
    response = _response(
        {
            "ok": True,
            "results": {
                "messages": [
                    {
                        "content": "main result",
                        "context_messages": {
                            "before": [{"text": "before text"}],
                            "after": [{"text": "after text"}],
                        },
                    }
                ]
            },
        }
    )
    with patch(_HTTP_REQUEST, return_value=response):
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release", "max_context_length": maximum},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result == {
        "ok": True,
        "results": {
            "messages": [
                {
                    "content": "main result",
                    "context_messages": {
                        "before": [{"text": "before text"[:maximum]}],
                        "after": [{"text": "after text"[:maximum]}],
                    },
                }
            ]
        },
    }


def test_slack_search_concise_response_keeps_result_identifiers() -> None:
    """Honor concise formatting without removing the search pagination cursor."""
    response = _response(
        {
            "ok": True,
            "results": {
                "messages": [{"message_ts": "1", "content": "message", "blocks": []}],
                "files": [{"file_id": "F1", "title": "file", "file_type": "pdf"}],
            },
            "response_metadata": {"next_cursor": "next"},
        }
    )
    with patch(_HTTP_REQUEST, return_value=response):
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release", "response_format": "concise"},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result == {
        "ok": True,
        "results": {
            "messages": [{"message_ts": "1", "content": "message"}],
            "files": [{"file_id": "F1", "title": "file"}],
        },
        "response_metadata": {"next_cursor": "next"},
    }


@pytest.mark.parametrize(
    "tool_name", ("slack_search_public", "slack_search_public_and_private")
)
@pytest.mark.parametrize(
    "code",
    ("feature_not_enabled", "assistant_search_context_disabled", "missing_scope"),
)
def test_slack_search_fails_when_real_time_search_is_unavailable(
    tool_name: str, code: str
) -> None:
    """Return Slack's search error without calling a different search API."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response({"ok": False, "error": code}),
        ) as request,
        pytest.raises(SlackApiError, match=code) as error,
    ):
        registry.invoke_connector(
            tool_name, {"query": "release"}, Mock(), _CREDENTIALS, {}
        )
    request.assert_called_once()
    assert request.call_args.args == (
        "POST",
        "https://slack.com/api/assistant.search.context",
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
    assert "search:read" not in SLACK_USER_SCOPES
    assert "files:read" in SLACK_USER_SCOPES
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
