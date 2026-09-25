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
"""Tests for the Notion connector."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from flwr.supercore.typing import JSONObject

from .. import registry
from ..definition import ActionAccess
from ..oauth import OAuthFlow
from .actions import ACTIONS
from .definition import NOTION_CONNECTOR_REF, PROVIDER
from .executors import NotionApiError

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_OAUTH_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"
_CREDENTIALS: JSONObject = {"access_token": "ntn-secret"}


def test_notion_definition_is_registered() -> None:
    """Notion schemas and executors should form one federation-scoped connector."""
    assert len(ACTIONS) == 6
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    assert [
        tool["name"] for tool in registry.get_connector_tools(NOTION_CONNECTOR_REF)
    ] == [
        "notion_search",
        "notion_get_page",
        "notion_get_page_property",
        "notion_list_users",
        "notion_get_user",
        "notion_get_self",
    ]


def test_notion_search_forwards_api_inputs() -> None:
    """Notion search should forward inputs using the API field names."""
    response = Mock(status_code=200)
    response.json.return_value = {"results": [], "has_more": False}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "notion_search",
            {
                "query": "release",
                "filter": {
                    "property": "object",
                    "value": "page",
                    "in_trash": False,
                },
                "sort": {
                    "timestamp": "last_edited_time",
                    "direction": "descending",
                },
                "page_size": 100,
                "start_cursor": "cursor-1",
            },
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value
    assert request.call_args.args == ("POST", "https://api.notion.com/v1/search")
    assert request.call_args.kwargs["headers"]["Notion-Version"] == "2026-03-11"
    assert request.call_args.kwargs["json"] == {
        "query": "release",
        "filter": {
            "property": "object",
            "value": "page",
            "in_trash": False,
        },
        "sort": {
            "timestamp": "last_edited_time",
            "direction": "descending",
        },
        "page_size": 100,
        "start_cursor": "cursor-1",
    }

    with patch(_HTTP_REQUEST, return_value=response) as request:
        registry.invoke_connector(
            "notion_search", {}, Mock(), credentials=_CREDENTIALS, config={}
        )
    assert request.call_args.kwargs["json"] == {}

    with patch(_HTTP_REQUEST, return_value=response) as request:
        registry.invoke_connector(
            "notion_search",
            {"filter": {"in_trash": True}},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert request.call_args.kwargs["json"] == {"filter": {"in_trash": True}}


@pytest.mark.parametrize(
    "filter_",
    [
        {},
        {"property": "object"},
        {"value": "page"},
        {"value": "page", "in_trash": True},
    ],
)
def test_notion_search_rejects_invalid_filter(filter_: JSONObject) -> None:
    """Notion search should reject incomplete and mixed filter shapes."""
    with patch(_HTTP_REQUEST) as request, pytest.raises(ValueError):
        registry.invoke_connector(
            "notion_search",
            {"filter": filter_},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    request.assert_not_called()


def test_notion_get_page_forwards_property_filter() -> None:
    """Get page should call only its endpoint and forward property filters."""
    response = Mock(status_code=200)
    response.json.return_value = {"object": "page", "id": "page-1"}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "notion_get_page",
            {"page_id": "page/1", "filter_properties": ["title", "f%5C%3Ap"]},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value
    assert request.call_args.args == (
        "GET",
        "https://api.notion.com/v1/pages/page%2F1",
    )
    assert request.call_args.kwargs["params"] == {
        "filter_properties": ["title", "f\\:p"]
    }


def test_notion_get_page_property_forwards_pagination() -> None:
    """Get page property should preserve encoded IDs and forward pagination."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "object": "list",
        "type": "property_item",
        "results": [],
        "has_more": True,
        "next_cursor": "cursor-2",
    }
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "notion_get_page_property",
            {
                "page_id": "page/1",
                "property_id": "f%5C%5C%3Ap",
                "page_size": 50,
                "start_cursor": "cursor-1",
            },
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value
    assert request.call_args.args == (
        "GET",
        "https://api.notion.com/v1/pages/page%2F1/properties/f%5C%5C%3Ap",
    )
    assert request.call_args.kwargs["params"] == {
        "page_size": "50",
        "start_cursor": "cursor-1",
    }


def test_notion_get_page_property_returns_single_item() -> None:
    """Get page property should also pass through single-item responses."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "object": "property_item",
        "id": "status",
        "type": "status",
        "status": None,
    }
    with patch(_HTTP_REQUEST, return_value=response):
        result = registry.invoke_connector(
            "notion_get_page_property",
            {"page_id": "page-1", "property_id": "status"},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value


def test_notion_list_users_forwards_pagination() -> None:
    """List users should forward Notion's pagination parameters."""
    response = Mock(status_code=200)
    response.json.return_value = {"object": "list", "results": []}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "notion_list_users",
            {"page_size": 50, "start_cursor": "cursor-1"},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value
    assert request.call_args.args == ("GET", "https://api.notion.com/v1/users")
    assert request.call_args.kwargs["params"] == {
        "page_size": "50",
        "start_cursor": "cursor-1",
    }


def test_notion_get_user_encodes_id() -> None:
    """Get user should retrieve exactly one safely encoded user ID."""
    response = Mock(status_code=200)
    response.json.return_value = {"object": "user", "id": "user-1"}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "notion_get_user",
            {"user_id": "user/1"},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value
    assert request.call_args.args == (
        "GET",
        "https://api.notion.com/v1/users/user%2F1",
    )


def test_notion_get_self_retrieves_token_user() -> None:
    """Get self should retrieve the user associated with the access token."""
    response = Mock(status_code=200)
    response.json.return_value = {"object": "user", "id": "user-1"}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "notion_get_self",
            {},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == response.json.return_value
    assert request.call_args.args == ("GET", "https://api.notion.com/v1/users/me")


def test_notion_api_errors_include_code_and_message() -> None:
    """Notion's documented error fields should remain readable to callers."""
    response = Mock(status_code=400)
    response.json.return_value = {
        "code": "validation_error",
        "message": "Invalid start_cursor value",
    }
    with (
        patch(_HTTP_REQUEST, return_value=response),
        pytest.raises(NotionApiError) as error,
    ):
        registry.invoke_connector(
            "notion_search", {"query": "release"}, Mock(), _CREDENTIALS, {}
        )
    assert error.value.code == "validation_error"
    assert str(error.value) == (
        "Notion API request failed: validation_error (400): Invalid start_cursor value."
    )
    assert "ntn-secret" not in str(error.value)


def test_notion_oauth_flow() -> None:
    """Notion OAuth should authorize and separate credentials from metadata."""
    redirect_uri = "https://example.com/callback"
    flow = OAuthFlow(
        PROVIDER,
        client_id="client",
        client_secret="secret",
        redirect_uri=redirect_uri,
    )
    url = flow.build_authorization_url(
        redirect_uri=redirect_uri, state="state", pkce_challenge=None
    )
    assert parse_qs(urlparse(url).query)["owner"] == ["user"]
    response = Mock(status_code=200)
    response.json.return_value = {
        "access_token": "token",
        "workspace_id": "workspace-1",
    }
    with patch(_OAUTH_REQUEST, return_value=response):
        credentials, config = flow.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier=None
        )
    assert credentials == {"access_token": "token"}
    assert config == {"workspace_id": "workspace-1"}

    response.json.return_value = {"error": "secret"}
    with patch(_OAUTH_REQUEST, return_value=response), pytest.raises(RuntimeError):
        flow.exchange_code(code="code", redirect_uri=redirect_uri, pkce_verifier=None)
