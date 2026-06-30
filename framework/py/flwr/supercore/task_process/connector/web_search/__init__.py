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
"""Built-in web search connector."""


import os

from flwr.supercore.typing import JSONObject

from .brave import BRAVE_API_KEY_ENV, BraveWebSearchProvider
from .exa import EXA_API_KEY_ENV, ExaWebSearchProvider
from .proxy import ProxyWebSearchProvider
from .tavily import TAVILY_API_KEY_ENV, TavilyWebSearchProvider

WEB_SEARCH_CONNECTOR_NAME = "web_search"
WEB_SEARCH_ENDPOINT_ENV = "FLWR_WEB_SEARCH_ENDPOINT"
_WEB_SEARCH_API_KEY_ENV_VARS = (
    BRAVE_API_KEY_ENV,
    TAVILY_API_KEY_ENV,
    EXA_API_KEY_ENV,
)


def make_web_search_tool() -> JSONObject:
    """Return the web search function tool schema."""
    return {
        "type": "function",
        "name": WEB_SEARCH_CONNECTOR_NAME,
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def search(query: str) -> JSONObject:
    """Execute one web search request."""
    if proxy_endpoint := os.getenv(WEB_SEARCH_ENDPOINT_ENV, "").strip():
        return ProxyWebSearchProvider(proxy_endpoint).search(query)
    if os.getenv(BRAVE_API_KEY_ENV, "").strip():
        return BraveWebSearchProvider().search(query)
    if os.getenv(TAVILY_API_KEY_ENV, "").strip():
        return TavilyWebSearchProvider().search(query)
    if os.getenv(EXA_API_KEY_ENV, "").strip():
        return ExaWebSearchProvider().search(query)

    raise RuntimeError(
        "At least one web search API key environment variable is required: "
        f"{', '.join(_WEB_SEARCH_API_KEY_ENV_VARS)}."
    )


__all__ = [
    "WEB_SEARCH_CONNECTOR_NAME",
    "WEB_SEARCH_ENDPOINT_ENV",
    "make_web_search_tool",
    "search",
]
