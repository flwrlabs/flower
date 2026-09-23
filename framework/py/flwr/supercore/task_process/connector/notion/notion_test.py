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
    assert len(ACTIONS) == 2
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    assert [
        tool["name"] for tool in registry.get_connector_tools(NOTION_CONNECTOR_REF)
    ] == ["notion_search", "notion_get_page"]


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


def test_notion_get_page_returns_page_and_block_children() -> None:
    """Get page should aggregate the page and its first-level child blocks."""
    page_response = Mock(status_code=200)
    page_response.json.return_value = {"object": "page", "id": "page-1"}
    first_blocks = Mock(status_code=200)
    first_blocks.json.return_value = {
        "object": "list",
        "results": [{"id": "block-1"}],
        "has_more": True,
        "next_cursor": "cursor-1",
    }
    last_blocks = Mock(status_code=200)
    last_blocks.json.return_value = {
        "object": "list",
        "results": [{"id": "block-2"}],
        "has_more": False,
        "next_cursor": None,
    }
    with patch(
        _HTTP_REQUEST, side_effect=[page_response, first_blocks, last_blocks]
    ) as request:
        result = registry.invoke_connector(
            "notion_get_page",
            {"page_id": "page-1"},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == {
        "page": page_response.json.return_value,
        "block_children": {
            **last_blocks.json.return_value,
            "results": [{"id": "block-1"}, {"id": "block-2"}],
        },
    }
    assert [call.args[:2] for call in request.call_args_list] == [
        ("GET", "https://api.notion.com/v1/pages/page-1"),
        ("GET", "https://api.notion.com/v1/blocks/page-1/children"),
        ("GET", "https://api.notion.com/v1/blocks/page-1/children"),
    ]
    assert request.call_args_list[-1].kwargs["params"] == {"start_cursor": "cursor-1"}


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
        "Notion API request failed: validation_error (400): "
        "Invalid start_cursor value."
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
