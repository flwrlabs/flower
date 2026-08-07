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

import base64
import binascii
import re
from urllib.parse import quote

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import (
    integer_field,
    object_list_field,
    optional_string,
    require_int_range,
    require_string,
    string_field,
)
from .oauth import GITHUB_API_VERSION

_API_BASE_URL = "https://api.github.com"
_JSON_ACCEPT = "application/vnd.github+json"
_TEXT_MATCH_ACCEPT = "application/vnd.github.text-match+json"
_OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_REPOSITORY_QUALIFIER = re.compile(r"(?:^|\s)repo:", re.IGNORECASE)


class GitHubApiError(ConnectorApiError):
    """Secret-safe GitHub API failure."""

    provider = "GitHub"


def search_code(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search code in one public GitHub repository."""
    owner, repo = _repository_ref(arguments.get("owner"), arguments.get("repo"))
    query = require_string(arguments.get("query"), "GitHub", "query")
    if _REPOSITORY_QUALIFIER.search(query) is not None:
        raise ValueError("GitHub query must not contain a repo qualifier.")
    limit = require_int_range(arguments.get("limit", 5), "GitHub", "limit", maximum=10)
    payload = _call_api(
        "/search/code",
        context.credentials,
        params={"q": f"{query} repo:{owner}/{repo}", "per_page": str(limit)},
        accept=_TEXT_MATCH_ACCEPT,
    )
    results = object_list_field(payload, "items", error=GitHubApiError)
    total_count = payload.get("total_count")
    incomplete = payload.get("incomplete_results")
    if isinstance(total_count, bool) or not isinstance(total_count, int):
        raise GitHubApiError("invalid_response")
    if not isinstance(incomplete, bool):
        raise GitHubApiError("invalid_response")
    return {
        "results": [_normalize_result(item) for item in results[:limit]],
        "total_count": total_count,
        "incomplete_results": incomplete,
    }


def get_file_content(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one UTF-8 text file from a public GitHub repository."""
    owner, repo = _repository_ref(arguments.get("owner"), arguments.get("repo"))
    path = _repository_path(arguments.get("path"))
    ref = optional_string(arguments.get("ref"), "GitHub", "ref")
    payload = _call_api(
        f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/"
        f"contents/{quote(path, safe='/')}",
        context.credentials,
        params={"ref": ref} if ref else {},
    )
    if payload.get("type") != "file" or payload.get("encoding") != "base64":
        raise GitHubApiError("unsupported_content")
    encoded = payload.get("content")
    if not isinstance(encoded, str) or not encoded:
        raise GitHubApiError("unsupported_content")
    try:
        content = base64.b64decode(encoded.replace("\n", ""), validate=True).decode(
            "utf-8"
        )
    except UnicodeDecodeError:
        raise GitHubApiError("unsupported_content") from None
    except (ValueError, binascii.Error):
        raise GitHubApiError("invalid_response") from None
    return {
        "owner": owner,
        "repo": repo,
        "path": string_field(payload, "path"),
        "name": string_field(payload, "name"),
        "sha": string_field(payload, "sha"),
        "size": integer_field(payload, "size"),
        "url": string_field(payload, "html_url"),
        "download_url": string_field(payload, "download_url"),
        "ref": ref or "",
        "content": content,
    }


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_code": search_code,
    "get_file_content": get_file_content,
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
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        params=params,
        http_error_code=lambda response: _http_error_code(response.status_code),
    )


def _normalize_result(item: JSONObject) -> JSONObject:
    """Return stable fields from one code search result."""
    repository = item.get("repository")
    matches = item.get("text_matches")
    fragments: list[str] = []
    if isinstance(matches, list):
        for match in matches:
            fragment = match.get("fragment") if isinstance(match, dict) else None
            if isinstance(fragment, str) and fragment:
                fragments.append(fragment)
    return {
        "name": string_field(item, "name"),
        "path": string_field(item, "path"),
        "sha": string_field(item, "sha"),
        "url": string_field(item, "html_url"),
        "repository_full_name": (
            string_field(repository, "full_name")
            if isinstance(repository, dict)
            else ""
        ),
        "fragments": fragments,
    }


def _repository_ref(owner: object, repo: object) -> tuple[str, str]:
    """Validate a public repository reference."""
    owner = require_string(owner, "GitHub", "owner")
    repo = require_string(repo, "GitHub", "repo")
    if _OWNER.fullmatch(owner) is None or _REPOSITORY.fullmatch(repo) is None:
        raise ValueError("GitHub repository is invalid.")
    if repo in {".", ".."}:
        raise ValueError("GitHub repository is invalid.")
    return owner, repo


def _repository_path(value: object) -> str:
    """Validate a repository-relative file path."""
    path = require_string(value, "GitHub", "path").lstrip("/")
    if not path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("GitHub path must point to a file.")
    return path


def _http_error_code(status_code: int) -> str:
    """Map GitHub status codes to stable errors."""
    return {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "invalid_request",
        429: "rate_limited",
    }.get(status_code, "http_error")
