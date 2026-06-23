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
"""Connector registry."""

from collections.abc import Callable

from flwr.supercore.typing import JSONObject, JSONValue

from . import browser_use, web_fetch, web_search

ConnectorHandler = Callable[..., JSONValue]
ConnectorToolFactory = Callable[[], JSONObject]

_CONNECTOR_HANDLERS: dict[str, ConnectorHandler] = {
    web_search.WEB_SEARCH_CONNECTOR_NAME: web_search.search,
    web_fetch.WEB_FETCH_CONNECTOR_NAME: web_fetch.invoke_web_fetch_provider,
    browser_use.BROWSER_USE_CONNECTOR_NAME: browser_use.invoke_browser_use_provider,
}
_BUILTIN_CONNECTOR_TOOL_FACTORIES: dict[str, ConnectorToolFactory] = {
    web_search.WEB_SEARCH_CONNECTOR_NAME: web_search.make_web_search_tool,
    web_fetch.WEB_FETCH_CONNECTOR_NAME: web_fetch.make_web_fetch_tool,
    browser_use.BROWSER_USE_CONNECTOR_NAME: browser_use.make_browser_use_tool,
}


def invoke_connector(name: str, arguments: JSONObject) -> JSONValue:
    """Invoke one connector by name."""
    handler = _CONNECTOR_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unsupported connector '{name}'.")
    return handler(**arguments)


def get_builtin_connector_tools() -> list[JSONObject]:
    """Return function tools for built-in connectors."""
    return [make_tool() for make_tool in _BUILTIN_CONNECTOR_TOOL_FACTORIES.values()]


def get_builtin_connector_tool(name: str) -> JSONObject:
    """Return the function tool for one built-in connector."""
    make_tool = _BUILTIN_CONNECTOR_TOOL_FACTORIES.get(name)
    if make_tool is None:
        raise ValueError(f"Unsupported connector '{name}'.")
    return make_tool()


def has_builtin_connector(name: str) -> bool:
    """Return whether a built-in connector is registered."""
    return name in _CONNECTOR_HANDLERS
