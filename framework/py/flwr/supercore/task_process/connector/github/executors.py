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
from ..errors import ConnectorArgumentError
from ..http import ConnectorApiError, request_json_object
from ..json_utils import optional_string, require_int_range, require_string

_API_BASE_URL = "https://api.github.com"
_API_VERSION = "2026-03-10"
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
        raise ConnectorArgumentError("GitHub query must not contain a repo qualifier.")
    limit = require_int_range(arguments.get("limit", 5), "GitHub", "limit", maximum=10)
    return _call_api(
        "/search/code",
        context.credentials,
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
    payload["content"] = content
    return payload


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
            "X-GitHub-Api-Version": _API_VERSION,
        },
        params=params,
    )


def _repository_ref(owner: object, repo: object) -> tuple[str, str]:
    """Validate a public repository reference."""
    owner = require_string(owner, "GitHub", "owner")
    repo = require_string(repo, "GitHub", "repo")
    if _OWNER.fullmatch(owner) is None or _REPOSITORY.fullmatch(repo) is None:
        raise ConnectorArgumentError("GitHub repository is invalid.")
    if repo in {".", ".."}:
        raise ConnectorArgumentError("GitHub repository is invalid.")
    return owner, repo


def _repository_path(value: object) -> str:
    """Validate a repository-relative file path."""
    path = require_string(value, "GitHub", "path").lstrip("/")
    if not path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ConnectorArgumentError("GitHub path must point to a file.")
    return path
