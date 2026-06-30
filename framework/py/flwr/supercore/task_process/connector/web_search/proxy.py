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
"""Proxy-backed web search adapter."""


from typing import cast

import requests

from flwr.supercore.typing import JSONObject, JSONValue

PROXY_WEB_SEARCH_PROVIDER = "proxy"
REQUEST_TIMEOUT = 60.0


class ProxyWebSearchProvider:
    """Proxy web search adapter."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def search(self, query: str) -> JSONObject:
        """Execute one proxy web search request."""
        if not query.strip():
            raise ValueError("web search requires a non-empty query.")
        query = query.strip()

        try:
            response = requests.post(
                self._endpoint,
                json={"query": query},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"{PROXY_WEB_SEARCH_PROVIDER} web search request failed: {exc}"
            ) from exc
        if response.status_code >= 400:
            try:
                detail = cast(JSONValue, response.json())
            except ValueError:
                detail = response.text
            raise RuntimeError(
                f"{PROXY_WEB_SEARCH_PROVIDER} web search request failed: "
                f"{response.status_code} {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"{PROXY_WEB_SEARCH_PROVIDER} web search returned invalid JSON."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"{PROXY_WEB_SEARCH_PROVIDER} web search returned invalid JSON."
            )

        if not isinstance(payload.get("results"), list):
            raise RuntimeError(
                f"{PROXY_WEB_SEARCH_PROVIDER} web search response must contain a "
                "results list."
            )
        return cast(JSONObject, payload)
