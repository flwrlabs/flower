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
"""Read-only GitHub connector tools for public repositories."""

import base64
import binascii
import re
from collections.abc import Callable
from typing import cast
from urllib.parse import quote

import requests

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

GITHUB_CONNECTOR_REF = "github"
GITHUB_SEARCH_CODE_TOOL = "github_search_code"
GITHUB_GET_FILE_CONTENT_TOOL = "github_get_file_content"
GITHUB_API_VERSION = "2026-03-10"

GITHUB_TOOL_NAMES = (
    GITHUB_SEARCH_CODE_TOOL,
    GITHUB_GET_FILE_CONTENT_TOOL,
)

_GITHUB_API_BASE_URL = "https://api.github.com"
_JSON_ACCEPT = "application/vnd.github+json"
_TEXT_MATCH_ACCEPT = "application/vnd.github.text-match+json"
_REQUEST_TIMEOUT = 30.0
_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_REPOSITORY_QUALIFIER = re.compile(r"(?:^|\s)repo:", re.IGNORECASE)


class GitHubApiError(RuntimeError):
    """Secret-safe GitHub API failure."""

    def __init__(self, code: str, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        detail = code if status_code is None else f"{code} ({status_code})"
        super().__init__(f"GitHub API request failed: {detail}.")


def make_github_tools() -> list[JSONObject]:
    """Return model-facing schemas for GitHub's read-only v1 operations."""
    repository_properties: JSONObject = {
        "owner": {
            "type": "string",
            "description": "GitHub organization or user that owns the repository.",
        },
        "repo": {
            "type": "string",
            "description": "Public GitHub repository name.",
        },
    }
    return [
        {
            "type": "function",
            "name": GITHUB_SEARCH_CODE_TOOL,
            "description": "Search code in one public GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    **repository_properties,
                    "query": {
                        "type": "string",
                        "description": "Code search query without a repo qualifier.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of matches to return.",
                    },
                },
                "required": ["owner", "repo", "query"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": GITHUB_GET_FILE_CONTENT_TOOL,
            "description": "Read one UTF-8 text file from a public GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    **repository_properties,
                    "path": {
                        "type": "string",
                        "description": "Repository-relative path to the file.",
                    },
                    "ref": {
                        "type": "string",
                        "description": (
                            "Optional branch, tag, or commit. Defaults to the "
                            "repository's default branch."
                        ),
                    },
                },
                "required": ["owner", "repo", "path"],
                "additionalProperties": False,
            },
        },
    ]


# pylint: disable-next=too-many-arguments
def search_code(
    owner: str,
    repo: str,
    query: str,
    limit: int = 5,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Search code in one public GitHub repository."""
    del config, usage_recorder
    owner, repo = _repository_ref(owner, repo)
    query = _non_empty_string(query, "query")
    if _REPOSITORY_QUALIFIER.search(query) is not None:
        raise ValueError("GitHub query must not contain a repo qualifier.")
    limit = _bounded_int(limit, "limit", maximum=10)
    payload = _call_github_api(
        f"{_GITHUB_API_BASE_URL}/search/code",
        credentials,
        params={
            "q": f"{query} repo:{owner}/{repo}",
            "per_page": str(limit),
        },
        accept=_TEXT_MATCH_ACCEPT,
    )
    results = _required_object_list(payload, "items")
    total_count = payload.get("total_count")
    incomplete_results = payload.get("incomplete_results")
    if (
        isinstance(total_count, bool)
        or not isinstance(total_count, int)
        or not isinstance(incomplete_results, bool)
    ):
        raise GitHubApiError("invalid_response")
    return {
        "results": [_normalize_search_result(item) for item in results[:limit]],
        "total_count": total_count,
        "incomplete_results": incomplete_results,
    }


# pylint: disable-next=too-many-arguments
def get_file_content(
    owner: str,
    repo: str,
    path: str,
    ref: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Read one UTF-8 text file from a public GitHub repository."""
    del config, usage_recorder
    owner, repo = _repository_ref(owner, repo)
    path = _repository_path(path)
    ref = _optional_string(ref, "ref")
    params = {"ref": ref} if ref is not None else {}
    payload = _call_github_api(
        f"{_GITHUB_API_BASE_URL}/repos/{quote(owner, safe='')}/"
        f"{quote(repo, safe='')}/contents/{quote(path, safe='/')}",
        credentials,
        params=params,
    )
    if payload.get("type") != "file":
        raise GitHubApiError("not_a_file")
    if payload.get("encoding") != "base64":
        raise GitHubApiError("unsupported_content")
    encoded_content = payload.get("content")
    if not isinstance(encoded_content, str) or not encoded_content:
        raise GitHubApiError("unsupported_content")
    try:
        decoded = base64.b64decode(encoded_content.replace("\n", ""), validate=True)
    except (ValueError, binascii.Error):
        raise GitHubApiError("invalid_response") from None
    try:
        content = decoded.decode("utf-8")
    except UnicodeDecodeError:
        raise GitHubApiError("unsupported_content") from None
    return {
        "owner": owner,
        "repo": repo,
        "path": _string_field(payload, "path"),
        "name": _string_field(payload, "name"),
        "sha": _string_field(payload, "sha"),
        "size": _integer_field(payload, "size"),
        "url": _string_field(payload, "html_url"),
        "download_url": _string_field(payload, "download_url"),
        "ref": ref or "",
        "content": content,
    }


GITHUB_TOOL_HANDLERS: dict[str, Callable[..., JSONValue]] = {
    GITHUB_SEARCH_CODE_TOOL: search_code,
    GITHUB_GET_FILE_CONTENT_TOOL: get_file_content,
}


def _call_github_api(
    url: str,
    credentials: JSONObject,
    *,
    params: dict[str, str],
    accept: str = _JSON_ACCEPT,
) -> JSONObject:
    """Call one GitHub REST endpoint and validate its response envelope."""
    access_token = credentials.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise GitHubApiError("invalid_credentials")
    try:
        response = requests.get(
            url,
            headers={
                "Accept": accept,
                "Authorization": f"Bearer {access_token}",
                "X-GitHub-Api-Version": GITHUB_API_VERSION,
            },
            params=params,
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise GitHubApiError("request_failed") from None
    if response.status_code >= 400:
        raise GitHubApiError(
            _http_error_code(response.status_code),
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError:
        raise GitHubApiError("invalid_response") from None
    if not isinstance(payload, dict):
        raise GitHubApiError("invalid_response")
    return cast(JSONObject, payload)


def _http_error_code(status_code: int) -> str:
    """Map GitHub status codes to stable model-facing errors."""
    return {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "invalid_request",
        429: "rate_limited",
    }.get(status_code, "http_error")


def _normalize_search_result(item: JSONObject) -> JSONObject:
    """Return the stable subset of a GitHub code search result."""
    repository = item.get("repository")
    text_matches = item.get("text_matches")
    fragments: list[str] = []
    if isinstance(text_matches, list):
        for text_match in text_matches:
            if not isinstance(text_match, dict):
                continue
            fragment = text_match.get("fragment")
            if isinstance(fragment, str) and fragment:
                fragments.append(fragment)
    return {
        "name": _string_field(item, "name"),
        "path": _string_field(item, "path"),
        "sha": _string_field(item, "sha"),
        "url": _string_field(item, "html_url"),
        "repository_full_name": (
            _string_field(repository, "full_name")
            if isinstance(repository, dict)
            else ""
        ),
        "fragments": fragments,
    }


def _repository_ref(owner: object, repo: object) -> tuple[str, str]:
    """Validate a repository owner and name without allowing query qualifiers."""
    owner = _non_empty_string(owner, "owner")
    repo = _non_empty_string(repo, "repo")
    if _OWNER.fullmatch(owner) is None:
        raise ValueError("GitHub owner is invalid.")
    if _REPOSITORY.fullmatch(repo) is None:
        raise ValueError("GitHub repo is invalid.")
    if repo in {".", ".."}:
        raise ValueError("GitHub repo is invalid.")
    return owner, repo


def _repository_path(value: object) -> str:
    """Validate and normalize a repository-relative path."""
    path = _non_empty_string(value, "path").lstrip("/")
    if not path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("GitHub path must point to a file.")
    return path


def _non_empty_string(value: object, name: str) -> str:
    """Validate and normalize a required string argument."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GitHub {name} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    """Validate and normalize an optional string argument."""
    if value is None:
        return None
    return _non_empty_string(value, name)


def _bounded_int(value: object, name: str, *, maximum: int) -> int:
    """Validate an integer argument with inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GitHub {name} must be an integer.")
    if value < 1 or value > maximum:
        raise ValueError(f"GitHub {name} must be between 1 and {maximum}.")
    return value


def _required_object_list(payload: JSONObject, key: str) -> list[JSONObject]:
    """Read a required list of JSON objects from a GitHub response."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise GitHubApiError("invalid_response")
    return value


def _string_field(payload: JSONObject, key: str) -> str:
    """Return a string field or an empty string."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _integer_field(payload: JSONObject, key: str) -> int | None:
    """Return an integer field or None."""
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
