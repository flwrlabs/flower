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
from urllib.parse import quote

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import (
    ConnectorInputError,
    optional_string,
    require_string,
)

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
    if sort := _optional_enum(arguments, "sort", ("indexed", "updated")):
        params["sort"] = sort
    if order := _optional_enum(arguments, "order", ("asc", "desc")):
        params["order"] = order
    if (per_page := _optional_integer(arguments, "perPage")) is not None:
        params["per_page"] = str(per_page)
    if (page := _optional_integer(arguments, "page")) is not None:
        params["page"] = str(page)
    payload = _call_api(
        "/search/code",
        context.credentials,
        params=params,
        accept=_TEXT_MATCH_ACCEPT,
    )
    items = payload.get("items")
    total_count = payload.get("total_count")
    return {
        "total_count": (
            total_count
            if isinstance(total_count, int) and not isinstance(total_count, bool)
            else 0
        ),
        "incomplete_results": bool(payload.get("incomplete_results")),
        "items": items if isinstance(items, list) else [],
    }


def get_file_contents(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one file from a GitHub repository."""
    owner = require_string(arguments.get("owner"), "GitHub", "owner")
    repo = require_string(arguments.get("repo"), "GitHub", "repo")
    path = _repository_path(arguments.get("path"))
    ref = optional_string(arguments.get("ref"), "GitHub", "ref")
    payload = _call_api(
        f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/"
        f"contents/{quote(path, safe='/')}",
        context.credentials,
        params={"ref": ref} if ref else {},
    )
    if payload.get("type") != "file":
        raise GitHubApiError("unsupported_content")
    encoded = payload.get("content")
    content_base64 = encoded.replace("\n", "") if isinstance(encoded, str) else ""
    payload["content_base64"] = content_base64
    payload["decoded_content"] = _decode_content(
        content_base64, payload.get("encoding")
    )
    return payload


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
    )


def _repository_path(value: object) -> str:
    """Validate a repository-relative file path."""
    path = require_string(value, "GitHub", "path").lstrip("/")
    if not path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ConnectorInputError("GitHub path must point to a file.")
    return path


def _optional_integer(arguments: JSONObject, name: str) -> int | None:
    """Return an optional integer argument."""
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConnectorInputError(f"GitHub {name} must be an integer.")
    return value


def _optional_enum(
    arguments: JSONObject, name: str, choices: tuple[str, ...]
) -> str | None:
    """Return an optional string enum argument."""
    value = optional_string(arguments.get(name), "GitHub", name)
    if value is not None and value not in choices:
        raise ConnectorInputError(
            f"GitHub {name} must be one of: {', '.join(choices)}."
        )
    return value


def _decode_content(content_base64: str, encoding: object) -> str | None:
    """Decode UTF-8 Base64 content when GitHub uses the supported encoding."""
    if not content_base64:
        return ""
    if encoding != "base64":
        return None
    try:
        return base64.b64decode(content_base64, validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError, binascii.Error):
        return None
