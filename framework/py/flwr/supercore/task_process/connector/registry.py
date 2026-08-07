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
from copy import deepcopy

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from . import automation, browser_use, slack, web_fetch, web_search
from .definition import (
    ConnectorDefinition,
    ConnectorExecutionContext,
    ConnectorExecutor,
    ConnectorHandler,
)
from .oauth import OAuthProvider

ConnectorToolFactory = Callable[[], JSONObject]


CONNECTORS: tuple[ConnectorDefinition, ...] = (slack.CONNECTOR,)
_CONNECTORS_BY_REF = {connector.ref: connector for connector in CONNECTORS}

OAUTH_CONNECTOR_PROVIDERS: tuple[OAuthProvider, ...] = tuple(
    connector.oauth_provider
    for connector in CONNECTORS
    if connector.oauth_provider is not None
)
_CONNECTOR_HANDLERS: dict[str, ConnectorHandler] = {
    web_search.WEB_SEARCH_CONNECTOR_NAME: web_search.search,
    web_fetch.WEB_FETCH_CONNECTOR_NAME: web_fetch.invoke_web_fetch_provider,
    browser_use.BROWSER_USE_CONNECTOR_NAME: browser_use.invoke_browser_use_provider,
}
_CREDENTIAL_CONNECTOR_HANDLERS: dict[str, ConnectorExecutor] = {
    name: handler
    for connector in CONNECTORS
    for name, handler in connector.handlers.items()
}
_CREDENTIAL_CONNECTOR_REFS: dict[str, str] = {
    name: connector.ref for connector in CONNECTORS for name in connector.handlers
}
_BUILTIN_CONNECTOR_TOOL_FACTORIES: dict[str, ConnectorToolFactory] = {
    automation.START_AUTOMATION_TOOL_NAME: automation.make_start_automation_tool,
    web_search.WEB_SEARCH_CONNECTOR_NAME: web_search.make_web_search_tool,
    web_fetch.WEB_FETCH_CONNECTOR_NAME: web_fetch.make_web_fetch_tool,
    browser_use.BROWSER_USE_CONNECTOR_NAME: browser_use.make_browser_use_tool,
}


def invoke_connector(
    name: str,
    arguments: JSONObject,
    usage_recorder: TaskUsageRecorder,
    credentials: JSONObject | None = None,
    config: JSONObject | None = None,
) -> JSONValue:
    """Invoke one connector by name."""
    handler = _CONNECTOR_HANDLERS.get(name)
    if handler is not None:
        return handler(**arguments, usage_recorder=usage_recorder)

    handler = _CREDENTIAL_CONNECTOR_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unsupported connector '{name}'.")
    if credentials is None or config is None:
        raise RuntimeError("Connector credentials are required.")
    return handler(
        arguments,
        ConnectorExecutionContext(
            credentials=credentials,
            config=config,
            usage_recorder=usage_recorder,
        ),
    )


def requires_connector_credentials(name: str) -> bool:
    """Return whether a connector uses account-scoped credentials."""
    return name in _CREDENTIAL_CONNECTOR_HANDLERS


def get_connector_ref(name: str) -> str:
    """Resolve a connector tool name to its OAuth connector reference."""
    return _CREDENTIAL_CONNECTOR_REFS.get(name, name)


def get_connector_tools(connector_ref: str) -> list[JSONObject]:
    """Return model-facing tools for one built-in or OAuth connector."""
    make_builtin_tool = _BUILTIN_CONNECTOR_TOOL_FACTORIES.get(connector_ref)
    if make_builtin_tool is not None:
        return [make_builtin_tool()]
    connector = _CONNECTORS_BY_REF.get(connector_ref)
    if connector is None:
        raise ValueError(f"Unsupported connector '{connector_ref}'.")
    return list(deepcopy(connector.tools))


def get_builtin_connector_tools() -> list[JSONObject]:
    """Return function tools for built-in connectors."""
    return [make_tool() for make_tool in _BUILTIN_CONNECTOR_TOOL_FACTORIES.values()]


def get_builtin_connector_tool(name: str) -> JSONObject:
    """Return the function tool for one built-in connector."""
    make_tool = _BUILTIN_CONNECTOR_TOOL_FACTORIES.get(name)
    if make_tool is None:
        raise ValueError(f"Unsupported connector '{name}'.")
    return make_tool()


def get_oauth_connector_provider(connector_ref: str) -> OAuthProvider:
    """Return the OAuth provider registered for a connector reference."""
    for provider in OAUTH_CONNECTOR_PROVIDERS:
        if provider.connector_ref == connector_ref:
            return provider
    raise ValueError(f"Unsupported OAuth connector '{connector_ref}'.")


def has_builtin_connector(name: str) -> bool:
    """Return whether a built-in connector is registered."""
    return name in _CONNECTOR_HANDLERS
