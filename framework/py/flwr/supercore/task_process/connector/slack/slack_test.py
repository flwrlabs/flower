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
from ..oauth import OAuthFlow
from .actions import ACTIONS
from .definition import PROVIDER, SLACK_CONNECTOR_REF, SLACK_USER_SCOPES

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_OAUTH_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"


def test_slack_actions_are_registered_and_executable() -> None:
    """Slack read actions should be registered and executable."""
    assert len(ACTIONS) == 4
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    assert len(registry.get_connector_tools(SLACK_CONNECTOR_REF)) == len(ACTIONS)
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


def test_slack_oauth_flow() -> None:
    """Slack OAuth should request read scopes and extract user credentials."""
    redirect_uri = "https://example.com/callback"
    flow = OAuthFlow(
        PROVIDER, client_id="client", client_secret="secret", redirect_uri=redirect_uri
    )
    url = flow.build_authorization_url(
        redirect_uri=redirect_uri, state="state", pkce_challenge="ignored"
    )
    assert parse_qs(urlparse(url).query)["user_scope"] == [",".join(SLACK_USER_SCOPES)]
    response = Mock(status_code=200)
    response.json.return_value = {
        "ok": True,
        "authed_user": {"access_token": "token", "scope": "search:read"},
    }
    with patch(_OAUTH_REQUEST, return_value=response):
        assert flow.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier="ignored"
        )[0] == {"access_token": "token"}

    response.json.return_value["authed_user"]["scope"] = "chat:write"
    with patch(_OAUTH_REQUEST, return_value=response), pytest.raises(RuntimeError):
        flow.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier="ignored"
        )
