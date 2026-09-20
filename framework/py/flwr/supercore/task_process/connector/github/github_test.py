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

from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from .. import registry
from ..oauth import OAuthFlow
from .definition import PROVIDER
from .executors import GitHubApiError

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_get_file_contents_returns_raw_response() -> None:
    """File reads should return GitHub's response unchanged."""
    response = _response(
        {
            "type": "file",
            "encoding": "base64",
            "content": "cHJpbnQoImhpIikK",
            "path": "src/app.py",
        }
    )
    with patch(_HTTP_REQUEST, return_value=response):
        result = registry.invoke_connector(
            "github_get_file_contents",
            {"owner": "acme", "repo": "repo", "path": "src/app.py"},
            Mock(),
            {"access_token": "secret"},
            {},
        )
    assert result == response.json.return_value

    with patch(_HTTP_REQUEST) as request, pytest.raises(ValueError):
        registry.invoke_connector(
            "github_get_file_contents",
            {"owner": "acme", "repo": "repo", "path": "../../user"},
            Mock(),
            {"access_token": "secret"},
            {},
        )
    request.assert_not_called()

    with patch(_HTTP_REQUEST) as request, pytest.raises(ValueError):
        registry.invoke_connector(
            "github_get_file_contents",
            {"owner": "acme", "repo": "..", "path": "foo"},
            Mock(),
            {"access_token": "secret"},
            {},
        )
    request.assert_not_called()


def test_github_search_forwards_page() -> None:
    """Code search should forward GitHub's numeric page parameter."""
    response = _response({"total_count": 12, "items": []})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = registry.invoke_connector(
            "github_search_code",
            {
                "query": "Flower repo:acme/repo",
                "sort": "indexed",
                "order": "desc",
                "per_page": 5,
                "page": 101,
            },
            Mock(),
            {"access_token": "secret"},
            {},
        )
    assert result == response.json.return_value
    assert request.call_args.kwargs["params"] == {
        "q": "Flower repo:acme/repo",
        "sort": "indexed",
        "order": "desc",
        "per_page": "5",
        "page": "101",
    }

    with pytest.raises(ValueError):
        registry.invoke_connector(
            "github_search_code",
            {"query": "Flower", "per_page": 101},
            Mock(),
            {"access_token": "secret"},
            {},
        )


def test_github_api_errors_include_message() -> None:
    """GitHub's documented error message should remain readable to callers."""
    response = _response({"message": "Validation Failed"}, status_code=422)
    with (
        patch(_HTTP_REQUEST, return_value=response),
        pytest.raises(GitHubApiError) as error,
    ):
        registry.invoke_connector(
            "github_search_code",
            {"query": "Flower repo:acme/repo"},
            Mock(),
            {"access_token": "secret"},
            {},
        )
    assert str(error.value) == (
        "GitHub API request failed: http_error (422): Validation Failed."
    )


def test_github_oauth_requests_no_scope() -> None:
    """OAuth should request and accept only scope-free credentials."""
    flow = OAuthFlow(
        PROVIDER,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    url = flow.build_authorization_url(
        redirect_uri="https://example.com/callback",
        state="state",
        pkce_challenge="challenge",
    )
    assert "scope" not in parse_qs(urlparse(url).query)
    token_response = _response(
        {"access_token": "token", "token_type": "bearer", "scope": ""}
    )
    with patch(_TOKEN_REQUEST, return_value=token_response):
        credentials, config = flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    assert credentials == {"access_token": "token", "token_type": "bearer"}
    assert not config

    token_response.json.return_value["scope"] = "repo"
    with (
        patch(_TOKEN_REQUEST, return_value=token_response),
        pytest.raises(RuntimeError),
    ):
        flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
