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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from . import browser_use, web_fetch, web_search

ConnectorHandler = Callable[..., JSONValue]
ConnectorToolFactory = Callable[[], JSONObject]


@dataclass(frozen=True)
class ConnectorDefinition:
    """Provider-level connector metadata."""

    connector_ref: str
    display_name: str
    description: str
    oauth_enabled: bool
    default_scopes: tuple[str, ...] = ()


class ConnectorToolProvider(Protocol):
    """Tool execution interface for one connector provider."""

    @property
    def definition(self) -> ConnectorDefinition:
        """Return provider-level connector metadata."""
        ...

    def tool_definitions(self) -> list[JSONObject]:
        """Return model-compatible tool definitions for this connector."""
        ...

    def execute_tool(
        self,
        *,
        arguments: JSONObject,
        usage_recorder: TaskUsageRecorder,
    ) -> JSONValue:
        """Execute one connector tool call."""
        ...


@dataclass(frozen=True)
class _BuiltInConnectorToolProvider:
    """Adapter from current built-in connector functions to tool providers."""

    definition: ConnectorDefinition
    _make_tool: ConnectorToolFactory
    _handler: ConnectorHandler

    def tool_definitions(self) -> list[JSONObject]:
        """Return the built-in connector's function tool definition."""
        return [self._make_tool()]

    def execute_tool(
        self,
        *,
        arguments: JSONObject,
        usage_recorder: TaskUsageRecorder,
    ) -> JSONValue:
        """Execute the built-in connector handler."""
        return self._handler(**arguments, usage_recorder=usage_recorder)


_CONNECTOR_TOOL_PROVIDERS: dict[str, ConnectorToolProvider] = {
    web_search.WEB_SEARCH_CONNECTOR_NAME: _BuiltInConnectorToolProvider(
        definition=ConnectorDefinition(
            connector_ref=web_search.WEB_SEARCH_CONNECTOR_NAME,
            display_name="Web Search",
            description="Search the web for current information.",
            oauth_enabled=False,
        ),
        _make_tool=web_search.make_web_search_tool,
        _handler=web_search.search,
    ),
    web_fetch.WEB_FETCH_CONNECTOR_NAME: _BuiltInConnectorToolProvider(
        definition=ConnectorDefinition(
            connector_ref=web_fetch.WEB_FETCH_CONNECTOR_NAME,
            display_name="Web Fetch",
            description="Fetch a web page and extract readable content.",
            oauth_enabled=False,
        ),
        _make_tool=web_fetch.make_web_fetch_tool,
        _handler=web_fetch.invoke_web_fetch_provider,
    ),
    browser_use.BROWSER_USE_CONNECTOR_NAME: _BuiltInConnectorToolProvider(
        definition=ConnectorDefinition(
            connector_ref=browser_use.BROWSER_USE_CONNECTOR_NAME,
            display_name="Browser Use",
            description="Use a headless browser to complete a web task.",
            oauth_enabled=False,
        ),
        _make_tool=browser_use.make_browser_use_tool,
        _handler=browser_use.invoke_browser_use_provider,
    ),
}
_BUILTIN_CONNECTOR_REFS = tuple(_CONNECTOR_TOOL_PROVIDERS)


def invoke_connector(
    name: str,
    arguments: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONValue:
    """Invoke one connector by name."""
    provider = _get_connector_tool_provider(name)
    return provider.execute_tool(arguments=arguments, usage_recorder=usage_recorder)


def get_builtin_connector_tools() -> list[JSONObject]:
    """Return function tools for built-in connectors."""
    tools: list[JSONObject] = []
    for connector_ref in _BUILTIN_CONNECTOR_REFS:
        tools.extend(_get_connector_tool_provider(connector_ref).tool_definitions())
    return tools


def get_builtin_connector_tool(name: str) -> JSONObject:
    """Return the function tool for one built-in connector."""
    provider = _get_builtin_connector_tool_provider(name)
    tool_definitions = provider.tool_definitions()
    if len(tool_definitions) != 1:
        raise ValueError(f"Connector '{name}' must expose exactly one built-in tool.")
    return tool_definitions[0]


def has_builtin_connector(name: str) -> bool:
    """Return whether a built-in connector is registered."""
    return name in _BUILTIN_CONNECTOR_REFS


def _get_builtin_connector_tool_provider(name: str) -> ConnectorToolProvider:
    """Return the built-in tool provider for one connector ref."""
    if name not in _BUILTIN_CONNECTOR_REFS:
        raise ValueError(f"Unsupported connector '{name}'.")
    return _get_connector_tool_provider(name)


def _get_connector_tool_provider(connector_ref: str) -> ConnectorToolProvider:
    """Return the tool provider for one connector ref."""
    provider = _CONNECTOR_TOOL_PROVIDERS.get(connector_ref)
    if provider is None:
        raise ValueError(f"Unsupported connector '{connector_ref}'.")
    return provider
