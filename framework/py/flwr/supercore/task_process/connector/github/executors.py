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
"""GitHub action executors."""

from urllib.parse import quote

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import optional_string, require_string
from .actions import GITHUB_PAGINATION_MINIMUM, GITHUB_PER_PAGE_MAXIMUM

_API_BASE_URL = "https://api.github.com"
_API_VERSION = "2026-03-10"
_JSON_ACCEPT = "application/vnd.github+json"
_TEXT_MATCH_ACCEPT = "application/vnd.github.text-match+json"


class GitHubApiError(ConnectorApiError):
    """Secret-safe GitHub API failure."""

    provider = "GitHub"


def search_code(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search GitHub code with GitHub search syntax."""
    query = require_string(arguments.get("query"), "GitHub", "query")
    params = {"q": query}
    for name in ("sort", "order"):
        if string_value := optional_string(arguments.get(name), "GitHub", name):
            params[name] = string_value
    for name in ("per_page", "page"):
        if name in arguments:
            integer_value = arguments[name]
            if (
                isinstance(integer_value, bool)
                or not isinstance(integer_value, int)
                or integer_value < GITHUB_PAGINATION_MINIMUM
                or (name == "per_page" and integer_value > GITHUB_PER_PAGE_MAXIMUM)
            ):
                constraint = (
                    f"between {GITHUB_PAGINATION_MINIMUM} and "
                    f"{GITHUB_PER_PAGE_MAXIMUM}"
                    if name == "per_page"
                    else f"at least {GITHUB_PAGINATION_MINIMUM}"
                )
                raise ValueError(f"GitHub {name} must be {constraint}.")
            params[name] = str(integer_value)
    return _call_api(
        "/search/code",
        context.credentials,
        params=params,
        accept=_TEXT_MATCH_ACCEPT,
    )


def get_file_contents(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Return GitHub's unchanged Contents API response, including Base64 content."""
    owner, repo = _repository_ref(arguments.get("owner"), arguments.get("repo"))
    path = _repository_path(arguments.get("path"))
    ref = optional_string(arguments.get("ref"), "GitHub", "ref")
    return _call_api(
        f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/"
        f"contents/{quote(path, safe='/')}",
        context.credentials,
        params={"ref": ref} if ref else {},
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_code": search_code,
    "get_file_contents": get_file_contents,
}


def _call_api(
    path: str,
    credentials: JSONObject,
    *,
    params: dict[str, str],
    accept: str = _JSON_ACCEPT,
) -> JSONObject:
    """Call one GitHub REST endpoint."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise GitHubApiError("invalid_credentials")
    return request_json_object(
        "GET",
        f"{_API_BASE_URL}{path}",
        error=GitHubApiError,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": _API_VERSION,
        },
        params=params,
        http_error_details=_response_error_details,
    )


def _response_error_details(response: requests.Response) -> tuple[str, str | None]:
    """Return GitHub's documented error code and message."""
    try:
        payload = response.json()
    except ValueError:
        return "http_error", None
    if not isinstance(payload, dict):
        return "http_error", None
    code = payload.get("code")
    message = payload.get("message")
    return (
        code if isinstance(code, str) and code else "http_error",
        message if isinstance(message, str) and message else None,
    )


def _repository_ref(owner: object, repo: object) -> tuple[str, str]:
    """Validate a public repository reference."""
    owner = require_string(owner, "GitHub", "owner")
    repo = require_string(repo, "GitHub", "repo")
    if owner in {".", ".."} or repo in {".", ".."}:
        raise ValueError("GitHub repository is invalid.")
    return owner, repo


def _repository_path(value: object) -> str:
    """Validate a repository-relative file path."""
    path = require_string(value, "GitHub", "path").strip("/")
    parts = [part for part in path.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("GitHub path must point to a repository file.")
    return "/".join(parts)
