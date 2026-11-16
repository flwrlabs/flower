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
"""Brave-backed web search adapter."""


import os
from collections.abc import Sequence
from typing import cast

import requests

from flwr.supercore.typing import JSONObject, JSONValue

BRAVE_WEB_SEARCH_PROVIDER = "brave"
BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
BRAVE_API_KEY_ENV = "BRAVE_API_KEY"
REQUEST_TIMEOUT = 60.0


class BraveWebSearchProvider:
    """Brave Search API adapter."""

    name = BRAVE_WEB_SEARCH_PROVIDER
    env = BRAVE_API_KEY_ENV

    def __init__(self) -> None:
        api_key = os.getenv(self.env, "").strip()
        if not api_key:
            raise RuntimeError(f"Environment variable {self.env} is required.")
        self._api_key = api_key

    def search(self, query: str) -> JSONObject:
        """Execute one Brave web search request."""
        if not query.strip():
            raise ValueError("web search requires a non-empty query.")
        query = query.strip()

        try:
            response = requests.get(
                BRAVE_WEB_SEARCH_URL,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                params={"q": query},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"{self.name} web search request failed: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = cast(JSONValue, response.json())
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"{self.name} web search request failed: "
                f"{response.status_code} {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"{self.name} web search returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"{self.name} web search returned invalid JSON.")

        return {
            "results": cast(
                Sequence[JSONValue],
                _parse_results(cast(JSONObject, payload)),
            )
        }


def _parse_results(payload: JSONObject) -> list[JSONObject]:
    web = payload.get("web")
    if not isinstance(web, dict):
        return []

    raw_results = web.get("results")
    if not isinstance(raw_results, list):
        return []

    results: list[JSONObject] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        snippet = item.get("description")
        published_at = item.get("page_age")
        results.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "snippet": (
                    snippet.strip()
                    if isinstance(snippet, str) and snippet.strip()
                    else None
                ),
                "published_at": (
                    published_at.strip()
                    if isinstance(published_at, str) and published_at.strip()
                    else None
                ),
            }
        )
    return results
