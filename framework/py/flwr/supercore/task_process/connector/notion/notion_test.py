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
    """Notion schemas and handlers should form one account-scoped connector."""
    assert len(ACTIONS) == 2
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    assert len(registry.get_connector_tools(NOTION_CONNECTOR_REF)) == len(ACTIONS)


@pytest.mark.parametrize(
    ("name", "arguments", "method", "path"),
    [
        ("notion_search", {"query": "release"}, "POST", "/search"),
    ],
)
def test_notion_tools_call_read_endpoints(
    name: str, arguments: JSONObject, method: str, path: str
) -> None:
    """Notion tools should call their read-only API endpoints."""
    response = Mock(status_code=200)
    response.json.return_value = {"results": [], "has_more": False}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            name, arguments, Mock(), credentials=_CREDENTIALS, config={}
        )
    assert result == response.json.return_value
    assert request.call_args.args == (method, f"https://api.notion.com/v1{path}")
    assert request.call_args.kwargs["headers"]["Notion-Version"] == "2026-03-11"


def test_notion_get_page_aggregates_page_and_children() -> None:
    """The Open Connector get_page action should perform both underlying reads."""
    responses = [
        Mock(status_code=200, **{"json.return_value": {"object": "page"}}),
        Mock(status_code=200, **{"json.return_value": {"object": "list"}}),
    ]
    with patch(_HTTP_REQUEST, side_effect=responses) as request:
        result = registry.invoke_connector(
            "notion_get_page",
            {"pageId": "page-1"},
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert result == {
        "page": {"object": "page"},
        "block_children": {"object": "list"},
    }
    assert [call.args[1] for call in request.call_args_list] == [
        "https://api.notion.com/v1/pages/page-1",
        "https://api.notion.com/v1/blocks/page-1/children",
    ]


def test_notion_search_forwards_open_connector_arguments() -> None:
    """Notion search should translate the Open Connector field names."""
    response = Mock(status_code=200)
    response.json.return_value = {"results": [], "has_more": False}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        registry.invoke_connector(
            "notion_search",
            {
                "query": "release",
                "filter": {"property": "object", "value": "page"},
                "sort": {"direction": "descending", "timestamp": "last_edited_time"},
                "pageSize": 25,
                "startCursor": "opaque",
            },
            Mock(),
            credentials=_CREDENTIALS,
            config={},
        )
    assert request.call_args.kwargs["json"] == {
        "query": "release",
        "filter": {"property": "object", "value": "page"},
        "sort": {"direction": "descending", "timestamp": "last_edited_time"},
        "page_size": 25,
        "start_cursor": "opaque",
    }


def test_notion_api_errors_are_secret_safe() -> None:
    """Notion failures should expose stable codes without credentials."""
    response = Mock(status_code=401)
    response.json.return_value = {"code": "unauthorized", "message": "ntn-secret"}
    with (
        patch(_HTTP_REQUEST, return_value=response),
        pytest.raises(NotionApiError) as error,
    ):
        registry.invoke_connector(
            "notion_search", {"query": "release"}, Mock(), _CREDENTIALS, {}
        )
    assert error.value.code == "unauthorized"
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
