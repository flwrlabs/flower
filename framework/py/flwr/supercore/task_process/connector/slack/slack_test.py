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
"""Tests for the Slack connector."""

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from .. import registry
from ..definition import ActionAccess
from .actions import ACTIONS, SLACK_TOOL_NAMES
from .definition import (
    SLACK_CONNECTOR_REF,
    SLACK_USER_SCOPES,
    SlackOAuthError,
    SlackOAuthProvider,
)
from .executors import SlackApiError

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_OAUTH_REQUEST = "flwr.supercore.task_process.connector.slack.definition.requests.post"


def test_slack_definition_is_registered() -> None:
    """Slack schemas and handlers should form one account-scoped connector."""
    assert [action.tool_name(SLACK_CONNECTOR_REF) for action in ACTIONS] == list(
        SLACK_TOOL_NAMES
    )
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    assert registry.get_connector_tools(SLACK_CONNECTOR_REF) == [
        action.tool(SLACK_CONNECTOR_REF) for action in ACTIONS
    ]
    assert all(
        registry.requires_connector_credentials(name) for name in SLACK_TOOL_NAMES
    )


def test_slack_search_calls_api_and_maps_errors() -> None:
    """Slack search should call the API and expose only stable error codes."""
    response = Mock(status_code=200)
    response.json.return_value = {"ok": True, "messages": {"matches": []}}
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "slack_search_messages",
            {"query": "release"},
            Mock(),
            {"access_token": "xoxp-secret"},
            {},
        )
    assert result == response.json.return_value
    assert request.call_args.args == ("GET", "https://slack.com/api/search.messages")

    response.json.return_value = {"ok": False, "error": "invalid_auth"}
    with (
        patch(_HTTP_REQUEST, return_value=response),
        pytest.raises(SlackApiError) as error,
    ):
        registry.invoke_connector(
            "slack_search_messages",
            {"query": "release"},
            Mock(),
            {"access_token": "xoxp-secret"},
            {},
        )
    assert error.value.code == "invalid_auth"
    assert "xoxp-secret" not in str(error.value)


def test_slack_oauth_flow() -> None:
    """Slack OAuth should request read scopes and extract user credentials."""
    redirect_uri = "https://example.com/callback"
    provider = SlackOAuthProvider(
        client_id="client", client_secret="secret", redirect_uri=redirect_uri
    )
    url = provider.build_authorization_url(
        redirect_uri=redirect_uri, state="state", pkce_challenge=None
    )
    assert parse_qs(urlparse(url).query)["user_scope"] == [",".join(SLACK_USER_SCOPES)]
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": True,
        "authed_user": {"access_token": "token", "scope": "search:read"},
    }
    with patch(_OAUTH_REQUEST, return_value=response):
        assert provider.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier=None
        )[0] == {"access_token": "token"}
    response.json.return_value = {"ok": False, "error": "secret"}
    with patch(_OAUTH_REQUEST, return_value=response), pytest.raises(SlackOAuthError):
        provider.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier=None
        )
