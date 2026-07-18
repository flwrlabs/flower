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

from flwr.proto.task_pb2 import TaskUsage  # pylint: disable=E0611
from flwr.supercore.task_process.usage import WEB_SEARCH_USAGE_TYPE, TaskUsageRecorder
from flwr.supercore.typing import JSONObject

from .brave import BraveWebSearchProvider
from .exa import ExaWebSearchProvider
from .provider import WebSearchProvider
from .proxy import WEB_SEARCH_ENDPOINT_ENV, ProxyWebSearchProvider
from .tavily import TavilyWebSearchProvider

WEB_SEARCH_CONNECTOR_NAME = "web_search"
_WEB_SEARCH_PROVIDER_ENV_VARS = (
    ProxyWebSearchProvider.env,
    BraveWebSearchProvider.env,
    TavilyWebSearchProvider.env,
    ExaWebSearchProvider.env,
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


def search(query: str, *, usage_recorder: TaskUsageRecorder) -> JSONObject:
    """Execute one web search request."""
    provider: WebSearchProvider
    if proxy_endpoint := os.getenv(ProxyWebSearchProvider.env, "").strip():
        provider = ProxyWebSearchProvider(proxy_endpoint)
    elif os.getenv(BraveWebSearchProvider.env, "").strip():
        provider = BraveWebSearchProvider()
    elif os.getenv(TavilyWebSearchProvider.env, "").strip():
        provider = TavilyWebSearchProvider()
    elif os.getenv(ExaWebSearchProvider.env, "").strip():
        provider = ExaWebSearchProvider()
    else:
        raise RuntimeError(
            "At least one web search provider environment variable is required: "
            f"{', '.join(_WEB_SEARCH_PROVIDER_ENV_VARS)}."
        )

    output = provider.search(query)
    usage_recorder.record(
        TaskUsage(usage_type=WEB_SEARCH_USAGE_TYPE, provider=provider.name)
    )
    return output


__all__ = [
    "WEB_SEARCH_CONNECTOR_NAME",
    "WEB_SEARCH_ENDPOINT_ENV",
    "make_web_search_tool",
    "search",
]
