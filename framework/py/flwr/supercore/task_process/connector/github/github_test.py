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

from .. import registry
from ..oauth import OAuthFlow
from .definition import PROVIDER

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_get_file_contents_decodes_utf8() -> None:
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
            "github_get_file_contents",
            {"owner": "acme", "repo": "repo", "path": "src/app.py"},
            Mock(),
            {"access_token": "secret"},
            {},
        )
    assert isinstance(result, dict)
    assert result["decoded_content"] == 'print("hi")\n'
    assert result["content_base64"] == response.json.return_value["content"]


def test_search_code_forwards_open_connector_arguments() -> None:
    """Code search should forward the Open Connector search controls."""
    response = _response({"total_count": 0, "incomplete_results": False, "items": []})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        registry.invoke_connector(
            "github_search_code",
            {
                "query": "repo:flower/framework language:python",
                "sort": "indexed",
                "order": "desc",
                "perPage": 20,
                "page": 2,
            },
            Mock(),
            {"access_token": "secret"},
            {},
        )
    assert request.call_args.kwargs["params"] == {
        "q": "repo:flower/framework language:python",
        "sort": "indexed",
        "order": "desc",
        "per_page": "20",
        "page": "2",
    }


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
