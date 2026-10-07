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
from ..oauth import OAuthFlow
from .definition import PROVIDER, SLACK_USER_SCOPES
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


def test_slack_search_defaults() -> None:
    """Use public channels and the captured default search options."""
    response = _response({"ok": True, "results": {"messages": []}})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release"},
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
        "content_types": ["messages", "files"],
        "channel_types": ["public_channel"],
        "include_bots": False,
        "include_context_messages": True,
    }


@pytest.mark.parametrize(
    ("action_name", "natural_language_query"),
    (
        ("slack_search_public", ""),
        ("slack_search_public_and_private", "Where is the release plan?"),
    ),
)
def test_slack_search_maps_captured_options(
    action_name: str, natural_language_query: str
) -> None:
    """Translate captured options and retain six AND'd keywords in one clause."""
    keywords = ["alpha", '"release plan"', "gamma", "delta", "epsilon", "zeta"]
    expected_clause = 'alpha "release plan" gamma delta epsilon zeta'
    arguments: JSONObject = {
        "keywords": keywords,
        "filters": "in:<#C1>",
        "natural_language_query": natural_language_query,
        "content_types": "messages,files",
        "after": "1700000000",
        "before": "1700003599",
        "cursor": "next",
        "context_channel_id": "C1",
        "sort": "timestamp",
        "sort_dir": "asc",
        "limit": 20,
        "include_bots": True,
        "include_context": False,
    }
    channel_types = ["public_channel"]
    if action_name == "slack_search_public_and_private":
        arguments["channel_types"] = "private_channel,im"
        channel_types = ["private_channel", "im"]
    response = _response({"ok": True, "results": {"messages": []}})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            action_name,
            arguments,
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
    assert request.call_args.kwargs["json"] == {
        "query": f"{natural_language_query or expected_clause} in:<#C1>",
        "term_clauses": [expected_clause],
        "modifiers": "in:<#C1>",
        "content_types": ["messages", "files"],
        "channel_types": channel_types,
        "after": 1700000000,
        "before": 1700003599,
        "cursor": "next",
        "context_channel_id": "C1",
        "sort": "timestamp",
        "sort_dir": "asc",
        "limit": 20,
        "include_bots": True,
        "include_context_messages": False,
    }


@pytest.mark.parametrize("limit", (0, 21))
def test_slack_search_rejects_invalid_limits(limit: int) -> None:
    """Reject out-of-range limits before making an API request."""
    with (
        patch(_HTTP_REQUEST) as request,
        pytest.raises(ValueError, match="must be between 1 and 20"),
    ):
        registry.invoke_connector(
            "slack_search_public",
            {"query": "release", "limit": limit},
            Mock(),
            _CREDENTIALS,
            {},
        )
    request.assert_not_called()


def test_slack_search_only_my_channels_keeps_shared_files() -> None:
    """Check file shares instead of dropping files with no search channel ID."""
    message = {"channel_id": "C1", "message_ts": "1", "content": "joined"}
    file = {"file_id": "F1", "title": "joined file"}
    pages = [
        {
            "ok": True,
            "results": {
                "messages": [message, {"channel_id": "C2"}],
                "files": [file, {"file_id": "F2"}],
            },
            "response_metadata": {"next_cursor": "search-next"},
        },
        {
            "ok": True,
            "channels": [{"id": "C1"}],
            "response_metadata": {"next_cursor": "membership-next"},
        },
        {"ok": True, "channels": [{"id": "C3"}]},
        {"ok": True, "file": {"channels": ["C3"]}},
        {"ok": True, "file": {"channels": ["C2"]}},
    ]
    with patch(
        _HTTP_REQUEST, side_effect=[_response(page) for page in pages]
    ) as request:
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


@pytest.mark.parametrize("maximum", (0, 4))
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


def test_slack_search_fails_when_real_time_search_is_unavailable() -> None:
    """Return Slack's search error without calling a different search API."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response({"ok": False, "error": "feature_not_enabled"}),
        ) as request,
        pytest.raises(SlackApiError, match="feature_not_enabled") as error,
    ):
        registry.invoke_connector(
            "slack_search_public", {"query": "release"}, Mock(), _CREDENTIALS, {}
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
