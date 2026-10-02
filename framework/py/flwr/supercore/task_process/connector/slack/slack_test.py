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
from unittest.mock import MagicMock, Mock, patch
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
        "slack_search_channels",
        "slack_search_users",
        "slack_read_channel",
        "slack_read_thread",
        "slack_read_canvas",
        "slack_read_user_profile",
        "slack_list_channel_members",
        "slack_read_file",
        "slack_list_user_channels",
        "slack_search_emojis",
        "slack_get_reactions",
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


def test_slack_channel_search_falls_back_to_conversations_list() -> None:
    """Channel search should work if the newer search endpoint is unavailable."""
    responses = [
        _response({"ok": False, "error": "feature_not_enabled"}),
        _response(
            {
                "ok": True,
                "channels": [
                    {"id": "C1", "name": "engineering"},
                    {"id": "C2", "name": "design"},
                ],
                "response_metadata": {"next_cursor": "next"},
            }
        ),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        result = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_search_channels",
                {"query": "engineer"},
                Mock(),
                _CREDENTIALS,
                {},
            ),
        )
    results = cast(JSONObject, result["results"])
    assert results["channels"] == [{"id": "C1", "name": "engineering"}]
    assert request.call_args.args == ("GET", "https://slack.com/api/conversations.list")


@pytest.mark.parametrize(
    ("tool", "content_types", "channel_types"),
    [
        ("slack_search_channels", ["channels"], ["public_channel"]),
        ("slack_search_users", ["users"], None),
    ],
)
def test_slack_entity_searches(
    tool: str, content_types: list[str], channel_types: list[str] | None
) -> None:
    """Search channels and users through the matching content types."""
    with patch(
        _HTTP_REQUEST, return_value=_response({"ok": True, "results": {}})
    ) as request:
        registry.invoke_connector(tool, {"query": "eng"}, Mock(), _CREDENTIALS, {})
    body = request.call_args.kwargs["json"]
    assert body["content_types"] == content_types
    assert body.get("channel_types") == channel_types


@pytest.mark.parametrize(
    ("tool", "arguments", "method", "params"),
    [
        (
            "slack_read_channel",
            {"channel_id": "C1", "cursor": "next", "limit": 15, "oldest": "1.0"},
            "conversations.history",
            {"channel": "C1", "cursor": "next", "limit": "15", "oldest": "1.0"},
        ),
        (
            "slack_read_thread",
            {"channel_id": "C1", "message_ts": "1.0", "limit": 100},
            "conversations.replies",
            {"channel": "C1", "ts": "1.0", "limit": "100"},
        ),
    ],
)
def test_slack_history_tools(
    tool: str, arguments: JSONObject, method: str, params: dict[str, str]
) -> None:
    """Read channel and thread messages with their Web API parameter names."""
    with patch(
        _HTTP_REQUEST, return_value=_response({"ok": True, "messages": []})
    ) as request:
        registry.invoke_connector(tool, arguments, Mock(), _CREDENTIALS, {})
    assert request.call_args.args == ("GET", f"https://slack.com/api/{method}")
    assert request.call_args.kwargs["params"] == params


def test_slack_list_members_fetches_profiles_and_filters_bots() -> None:
    """Detailed member results should include profiles and omit bots by default."""
    responses = [
        _response({"ok": True, "members": ["U1", "U2"], "response_metadata": {}}),
        _response({"ok": True, "user": {"id": "U1", "name": "alice"}}),
        _response({"ok": True, "user": {"id": "U2", "is_bot": True}}),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        result = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_list_channel_members",
                {"channel_id": "C1"},
                Mock(),
                _CREDENTIALS,
                {},
            ),
        )
    assert result["members"] == [{"id": "U1", "name": "alice"}]
    assert request.call_count == 3
    assert request.call_args_list[0].args[1].endswith("/conversations.members")


def test_slack_list_members_count_only() -> None:
    """Count members without fetching or paginating profiles."""
    with patch(
        _HTTP_REQUEST,
        return_value=_response({"ok": True, "channel": {"num_members": 42}}),
    ) as request:
        result = registry.invoke_connector(
            "slack_list_channel_members",
            {"channel_id": "C1", "response_format": "count_only"},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert result == {"ok": True, "channel_id": "C1", "num_members": 42}
    assert request.call_args.args[1].endswith("/conversations.info")
    assert request.call_args.kwargs["params"] == {
        "channel": "C1",
        "include_num_members": "true",
    }


def test_slack_read_canvas_uses_canvas_content_and_sections() -> None:
    """Read markdown and section IDs through canvas API methods."""
    responses = [
        _response({"ok": True, "content": "# Plan"}),
        _response({"ok": True, "sections": [{"id": "section-1"}]}),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        result = registry.invoke_connector(
            "slack_read_canvas", {"canvas_id": "F1"}, Mock(), _CREDENTIALS, {}
        )
    assert result == {
        "ok": True,
        "canvas_id": "F1",
        "content": "# Plan",
        "sections": [{"id": "section-1"}],
    }
    assert [call.args[1].split("/")[-1] for call in request.call_args_list] == [
        "canvases.getContent",
        "canvases.sections.lookup",
    ]


def test_slack_read_file_downloads_text() -> None:
    """Read file metadata and content with the connected user's token."""
    info = _response(
        {
            "ok": True,
            "file": {
                "id": "F1",
                "mimetype": "text/plain",
                "url_private_download": "https://files.slack.com/files-pri/F1",
            },
        }
    )
    download = MagicMock(status_code=200)
    download.__enter__.return_value = download
    download.iter_content.return_value = [b"hello"]
    with (
        patch(_HTTP_REQUEST, return_value=info),
        patch(_FILE_REQUEST, return_value=download) as get,
    ):
        result = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_read_file", {"file_id": "F1"}, Mock(), _CREDENTIALS, {}
            ),
        )
    assert result["content"] == "hello"
    assert result["encoding"] == "utf-8"
    assert result["file"] == {"id": "F1", "mimetype": "text/plain"}
    assert get.call_args.kwargs["headers"] == {"Authorization": "Bearer xoxp-secret"}


def test_slack_search_emojis_and_get_reactions() -> None:
    """Use the documented emoji and reactions read endpoints."""
    with patch(
        _HTTP_REQUEST,
        return_value=_response(
            {"ok": True, "emoji": {"party_parrot": "url1", "wave": "url2"}}
        ),
    ) as request:
        emoji = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_search_emojis", {"query": "party"}, Mock(), _CREDENTIALS, {}
            ),
        )
    assert emoji["emoji"] == {"party_parrot": "url1"}
    assert request.call_args.args[1].endswith("/emoji.list")

    with patch(
        _HTTP_REQUEST, return_value=_response({"ok": True, "message": {}})
    ) as request:
        registry.invoke_connector(
            "slack_get_reactions",
            {"channel_id": "C1", "message_ts": "1.0"},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert request.call_args.args[1].endswith("/reactions.get")
    assert request.call_args.kwargs["params"] == {
        "channel": "C1",
        "timestamp": "1.0",
    }


def test_slack_get_reactions_includes_user_display_names() -> None:
    """Preserve counts and resolve the displayed reaction users."""
    responses = [
        _response(
            {
                "ok": True,
                "message": {
                    "reactions": [{"name": "wave", "count": 3, "users": ["U1", "U2"]}]
                },
            }
        ),
        _response(
            {"ok": True, "user": {"id": "U1", "profile": {"display_name": "Ada"}}}
        ),
        _response({"ok": True, "user": {"id": "U2", "name": "grace"}}),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses):
        result = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_get_reactions",
                {"channel_id": "C1", "message_ts": "1.0"},
                Mock(),
                _CREDENTIALS,
                {},
            ),
        )
    message = cast(JSONObject, result["message"])
    reaction = cast(list[JSONObject], message["reactions"])[0]
    assert reaction["count"] == 3
    assert reaction["users_with_display_names"] == [
        {"user_id": "U1", "display_name": "Ada"},
        {"user_id": "U2", "display_name": "grace"},
    ]


def test_slack_list_user_channels_uses_web_api() -> None:
    """Filter and format joined channels without invoking Slack MCP."""
    response = _response(
        {
            "ok": True,
            "channels": [
                {"id": "C1", "name": "engineering"},
                {"id": "C2", "name": "sales"},
            ],
            "response_metadata": {"next_cursor": ""},
        }
    )
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = cast(
            JSONObject,
            registry.invoke_connector(
                "slack_list_user_channels",
                {"name_prefix": "eng", "format": "ids_only"},
                Mock(),
                _CREDENTIALS,
                {},
            ),
        )
    assert result["channels"] == ["C1"]
    assert request.call_args.args == (
        "GET",
        "https://slack.com/api/users.conversations",
    )
    assert request.call_args.kwargs["params"]["types"] == (
        "public_channel,private_channel"
    )


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
