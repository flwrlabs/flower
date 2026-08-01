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
"""Tests for read-only GitHub connector tools."""

from base64 import b64encode
from unittest.mock import Mock, patch

import pytest

from . import registry
from .github import (
    GITHUB_API_VERSION,
    GITHUB_CONNECTOR_REF,
    GITHUB_TOOL_NAMES,
    GitHubApiError,
    get_file_content,
    search_code,
)

_CREDENTIALS = {"access_token": "gho-secret"}


def _response(payload: object, *, status_code: int = 200) -> Mock:
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def test_github_tools_are_registered_as_read_only_credentials() -> None:
    """GitHub tools should be closed schemas backed by one OAuth connection."""
    tools = registry.get_connector_tools(GITHUB_CONNECTOR_REF)

    assert [tool["name"] for tool in tools] == list(GITHUB_TOOL_NAMES)
    assert not registry.has_builtin_connector(GITHUB_CONNECTOR_REF)
    for tool in tools:
        name = str(tool["name"])
        assert tool["type"] == "function"
        assert tool["parameters"]["additionalProperties"] is False
        assert "create" not in name and "update" not in name
        assert registry.requires_connector_credentials(name)
        assert registry.get_connector_ref(name) == GITHUB_CONNECTOR_REF


def test_search_code_calls_github_and_normalizes_matches() -> None:
    """Code search should stay repository-scoped and return stable fields."""
    response = _response(
        {
            "total_count": 1,
            "incomplete_results": False,
            "items": [
                {
                    "name": "app.py",
                    "path": "src/app.py",
                    "sha": "abc123",
                    "html_url": "https://github.com/acme/repo/blob/main/src/app.py",
                    "repository": {"full_name": "acme/repo"},
                    "text_matches": [{"fragment": "def hello():"}],
                }
            ],
        }
    )
    with patch(
        "flwr.supercore.task_process.connector.github.requests.get",
        return_value=response,
    ) as get:
        result = search_code(
            " acme ",
            "repo",
            " hello language:python ",
            limit=1,
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    get.assert_called_once_with(
        "https://api.github.com/search/code",
        headers={
            "Accept": "application/vnd.github.text-match+json",
            "Authorization": "Bearer gho-secret",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        params={
            "q": "hello language:python repo:acme/repo",
            "per_page": "1",
        },
        timeout=30.0,
    )
    assert result == {
        "results": [
            {
                "name": "app.py",
                "path": "src/app.py",
                "sha": "abc123",
                "url": "https://github.com/acme/repo/blob/main/src/app.py",
                "repository_full_name": "acme/repo",
                "fragments": ["def hello():"],
            }
        ],
        "total_count": 1,
        "incomplete_results": False,
    }


def test_get_file_content_decodes_text_and_preserves_ref() -> None:
    """File reads should URL-encode paths and decode GitHub's Base64 payload."""
    response = _response(
        {
            "type": "file",
            "encoding": "base64",
            "content": b64encode(b'print("hi")\n').decode("ascii"),
            "name": "My file.py",
            "path": "src/My file.py",
            "sha": "sha123",
            "size": 12,
            "html_url": "https://github.com/acme/repo/blob/main/src/My%20file.py",
            "download_url": (
                "https://raw.githubusercontent.com/acme/repo/main/src/My%20file.py"
            ),
        }
    )
    with patch(
        "flwr.supercore.task_process.connector.github.requests.get",
        return_value=response,
    ) as get:
        result = get_file_content(
            "acme",
            "repo",
            "/src/My file.py",
            ref=" main ",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    get.assert_called_once_with(
        "https://api.github.com/repos/acme/repo/contents/src/My%20file.py",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer gho-secret",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        params={"ref": "main"},
        timeout=30.0,
    )
    assert result == {
        "owner": "acme",
        "repo": "repo",
        "path": "src/My file.py",
        "name": "My file.py",
        "sha": "sha123",
        "size": 12,
        "url": "https://github.com/acme/repo/blob/main/src/My%20file.py",
        "download_url": (
            "https://raw.githubusercontent.com/acme/repo/main/src/My%20file.py"
        ),
        "ref": "main",
        "content": 'print("hi")\n',
    }


def test_api_failures_are_stable_and_secret_safe() -> None:
    """Provider and transport failures should expose only stable error codes."""
    response = _response(
        {"message": "Bearer gho-secret is invalid"},
        status_code=401,
    )
    with (
        patch(
            "flwr.supercore.task_process.connector.github.requests.get",
            return_value=response,
        ),
        pytest.raises(GitHubApiError) as error,
    ):
        search_code(
            "acme",
            "repo",
            "hello",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == "unauthorized"
    assert "gho-secret" not in str(error.value)


def test_malformed_file_content_is_rejected() -> None:
    """Malformed Base64 content should fail predictably."""
    with (
        patch(
            "flwr.supercore.task_process.connector.github.requests.get",
            return_value=_response(
                {
                    "type": "file",
                    "encoding": "base64",
                    "content": "***invalid***",
                }
            ),
        ),
        pytest.raises(GitHubApiError) as error,
    ):
        get_file_content(
            "acme",
            "repo",
            "src/app.py",
            credentials=_CREDENTIALS,
            config={},
            usage_recorder=Mock(),
        )

    assert error.value.code == "invalid_response"


def test_repository_escape_inputs_fail_before_request() -> None:
    """Queries and paths should not escape the explicitly selected repository."""
    with patch("flwr.supercore.task_process.connector.github.requests.get") as get:
        with pytest.raises(ValueError):
            search_code(
                "acme",
                "repo",
                "x repo:other/repo",
                credentials=_CREDENTIALS,
                config={},
                usage_recorder=Mock(),
            )
        with pytest.raises(ValueError):
            get_file_content(
                "acme",
                "repo",
                "src/../secret",
                credentials=_CREDENTIALS,
                config={},
                usage_recorder=Mock(),
            )

    get.assert_not_called()
