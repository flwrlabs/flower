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
"""Tests for the GitHub connector."""

from base64 import b64encode
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from flwr.supercore.typing import JSONObject

from .. import registry
from ..definition import ActionAccess
from .actions import ACTIONS
from .oauth import GITHUB_CONNECTOR_REF, GitHubOAuthError, GitHubOAuthProvider

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.github.oauth.requests.post"
_ACCOUNT_REQUEST = "flwr.supercore.task_process.connector.github.oauth.requests.get"
_CREDENTIALS: JSONObject = {"access_token": "gho-secret"}


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_github_actions_are_registered_as_read_only() -> None:
    """GitHub should expose two account-scoped read actions."""
    assert len(ACTIONS) == 2
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    assert len(registry.get_connector_tools(GITHUB_CONNECTOR_REF)) == len(ACTIONS)


def test_search_code_calls_api_and_normalizes_results() -> None:
    """Code search should remain repository-scoped and return stable fields."""
    response = _response(
        {
            "total_count": 1,
            "incomplete_results": False,
            "items": [
                {
                    "name": "app.py",
                    "path": "src/app.py",
                    "sha": "abc",
                    "html_url": "https://github.com/acme/repo/blob/main/src/app.py",
                    "repository": {"full_name": "acme/repo"},
                    "text_matches": [{"fragment": "def hello():"}],
                }
            ],
        }
    )
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "github_search_code",
            {"owner": "acme", "repo": "repo", "query": "hello", "limit": 1},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert request.call_args.args == ("GET", "https://api.github.com/search/code")
    assert isinstance(result, dict)
    assert result["results"] == [
        {
            "name": "app.py",
            "path": "src/app.py",
            "sha": "abc",
            "url": "https://github.com/acme/repo/blob/main/src/app.py",
            "repository_full_name": "acme/repo",
            "fragments": ["def hello():"],
        }
    ]


def test_get_file_content_decodes_utf8() -> None:
    """File reads should decode GitHub's Base64 content."""
    response = _response(
        {
            "type": "file",
            "encoding": "base64",
            "content": b64encode(b'print("hi")\n').decode("ascii"),
            "path": "src/app.py",
        }
    )
    with patch(_HTTP_REQUEST, return_value=response):
        result = registry.invoke_connector(
            "github_get_file_content",
            {"owner": "acme", "repo": "repo", "path": "src/app.py"},
            Mock(),
            _CREDENTIALS,
            {},
        )
    assert isinstance(result, dict)
    assert result["content"] == 'print("hi")\n'


def test_github_oauth_verifies_scope_and_account() -> None:
    """OAuth should request no scope and record the authenticated account."""
    provider = GitHubOAuthProvider(
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    url = provider.build_authorization_url(
        redirect_uri="https://example.com/callback",
        state="state",
        pkce_challenge="challenge",
    )
    assert "scope" not in parse_qs(urlparse(url).query)
    token_response = _response(
        {"access_token": "token", "token_type": "bearer", "scope": ""}
    )
    account_response = _response({"id": 123, "login": "octocat"})
    with (
        patch(_TOKEN_REQUEST, return_value=token_response),
        patch(_ACCOUNT_REQUEST, return_value=account_response),
    ):
        credentials, config = provider.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    assert credentials == {"access_token": "token"}
    assert config["login"] == "octocat"

    token_response.json.return_value["scope"] = "repo"
    with (
        patch(_TOKEN_REQUEST, return_value=token_response),
        pytest.raises(GitHubOAuthError),
    ):
        provider.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
