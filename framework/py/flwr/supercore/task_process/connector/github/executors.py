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
from ..http import ConnectorApiError
from ..json_utils import optional_string, require_int_range, require_string

GITHUB_API_VERSION = "2026-03-10"
GITHUB_JSON_ACCEPT = "application/vnd.github+json"
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
    return _call_api(
        "/search/code",
        context,
        params={"q": f"{query} repo:{owner}/{repo}", "per_page": str(limit)},
        accept=_TEXT_MATCH_ACCEPT,
    )


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
        context,
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
    payload["content"] = content
    return payload


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_code": search_code,
    "get_file_content": get_file_content,
}


def _call_api(
    path: str,
    context: ConnectorExecutionContext,
    *,
    params: dict[str, str],
    accept: str = GITHUB_JSON_ACCEPT,
) -> JSONObject:
    """Call one GitHub REST endpoint."""
    if context.http is None:
        raise RuntimeError("GitHub HTTP client is not configured.")
    return context.http.request(
        "GET",
        path,
        error=GitHubApiError,
        headers={"Accept": accept},
        params=params,
    )


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
