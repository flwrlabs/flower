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
"""Connector registry indexed by connector reference and tool name."""

from copy import deepcopy

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

from . import browser_use, filesystem, web_fetch, web_search
from .automation import START_AUTOMATION_TOOL_NAME, make_start_automation_tool
from .definition import ConnectorDefinition, ConnectorExecutionContext
from .loader import load_oauth_connectors
from .oauth import OAuthFlow

CONNECTORS = (
    web_search.CONNECTOR,
    web_fetch.CONNECTOR,
    browser_use.CONNECTOR,
    filesystem.CONNECTOR,
    *load_oauth_connectors(),
)
_CONNECTORS_BY_REF = {connector.ref: connector for connector in CONNECTORS}
_CONNECTORS_BY_TOOL = {
    name: connector for connector in CONNECTORS for name in connector.executors
}

OAUTH_FLOWS: dict[str, OAuthFlow] = {
    connector.ref: connector.oauth_flow
    for connector in CONNECTORS
    if connector.oauth_flow is not None
}


def invoke_connector(
    tool_name: str,
    arguments: JSONObject,
    usage_recorder: TaskUsageRecorder,
    credentials: JSONObject | None = None,
    config: JSONObject | None = None,
) -> JSONValue:
    """Invoke one connector tool by its model-facing name."""
    connector = _resolve_connector(tool_name)
    if connector is None:
        raise ValueError(f"Unsupported connector '{tool_name}'.")
    if connector.requires_credentials and (credentials is None or config is None):
        raise RuntimeError("Connector credentials are required.")
    context = ConnectorExecutionContext(
        credentials=credentials or {},
        config=config or {},
        usage_recorder=usage_recorder,
    )
    if connector.load_connector is not None:
        discovery_tool = get_connector_discovery_tool(connector.ref)
        connector = connector.load_connector(context.credentials)
        if tool_name == discovery_tool:
            return list(connector.tools)
    if executor := connector.executors.get(tool_name):
        return executor(arguments, context)
    raise ValueError(f"Unsupported connector '{tool_name}'.")


def requires_connector_credentials(tool_name: str) -> bool:
    """Return whether a tool's connector uses federation-scoped credentials."""
    connector = _resolve_connector(tool_name)
    return connector is not None and connector.requires_credentials


def get_connector_ref(tool_name: str) -> str:
    """Resolve a connector tool name to its connector reference."""
    connector = _resolve_connector(tool_name)
    return connector.ref if connector is not None else tool_name


def get_connector_tools(connector_ref: str) -> list[JSONObject]:
    """Return model-facing tools for a connector or agent-handled automation."""
    if connector_ref == START_AUTOMATION_TOOL_NAME:
        return [make_start_automation_tool()]
    connector = _CONNECTORS_BY_REF.get(connector_ref)
    if connector is None:
        raise ValueError(f"Unsupported connector '{connector_ref}'.")
    return list(deepcopy(connector.tools))


def get_connector_discovery_tool(connector_ref: str) -> str | None:
    """Return the worker operation used to discover a connection's tools."""
    connector = _CONNECTORS_BY_REF.get(connector_ref)
    if connector is not None and connector.load_connector is not None:
        return f"{connector_ref}__discover_tools"
    return None


def _resolve_connector(tool_name: str) -> ConnectorDefinition | None:
    """Resolve static tools first, then a dynamic connector's namespace."""
    if connector := _CONNECTORS_BY_TOOL.get(tool_name):
        return connector
    for connector in _CONNECTORS_BY_REF.values():
        prefix = f"{connector.ref}_"
        if (
            connector.load_connector is not None
            and tool_name.startswith(prefix)
            and len(tool_name) > len(prefix)
        ):
            return connector
    return None


def get_oauth_flow(connector_ref: str) -> OAuthFlow:
    """Return the OAuth flow configured for a connector reference."""
    flow = OAUTH_FLOWS.get(connector_ref)
    if flow is None:
        raise ValueError(f"Unsupported OAuth connector '{connector_ref}'.")
    return flow


def has_builtin_connector(connector_ref: str) -> bool:
    """Return whether a reference identifies a connector without stored credentials."""
    connector = _CONNECTORS_BY_REF.get(connector_ref)
    return connector is not None and not connector.requires_credentials
