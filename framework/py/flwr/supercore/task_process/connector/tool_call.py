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
"""Connector function-tool helpers."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import cast

from flwr.supercore.typing import JSONObject

from .registry import get_builtin_connector_tool, has_builtin_connector


@dataclass(frozen=True)
class ConnectorToolCall:
    """A model-requested connector function call."""

    name: str
    call_id: str
    arguments: JSONObject


@dataclass(frozen=True)
class PreparedConnectorTools:
    """A model request with per-request connector tool state.

    `enabled_builtin_connectors` is the built-in connector allowlist for one
    `responses.create` call. Later calls opt in again by passing built-in
    connector names in `tools`.
    """

    request: JSONObject
    enabled_builtin_connectors: frozenset[str]


def with_builtin_connector_tools(request: JSONObject) -> PreparedConnectorTools:
    """Return request with requested built-in connector function tools enabled."""
    updated = dict(request)
    tools = request.get("tools")

    if tools is None:
        return PreparedConnectorTools(
            request=updated,
            enabled_builtin_connectors=frozenset(),
        )

    if isinstance(tools, Sequence) and not isinstance(tools, str):
        enabled_builtin_connectors: set[str] = set()
        normalized_tools: list[JSONObject] = []

        tool_list = list(tools)
        for tool in tool_list:
            if isinstance(tool, str):
                # String entries are the runtime shorthand for opting into a
                # built-in connector for this request.
                if not has_builtin_connector(tool):
                    raise ValueError(f"Unknown built-in connector tool '{tool}'.")
                if tool in enabled_builtin_connectors:
                    raise ValueError(f"Duplicate built-in connector tool '{tool}'.")

                normalized_tools.append(get_builtin_connector_tool(tool))
                enabled_builtin_connectors.add(tool)
                continue

            if isinstance(tool, dict):
                tool_name = tool.get("name")
                # JSON tool definitions belong to AgentApp/user code. Built-in
                # connector names are reserved for runtime-owned connector calls.
                if isinstance(tool_name, str) and has_builtin_connector(tool_name):
                    raise ValueError(
                        f"Built-in connector tool name '{tool_name}' is reserved. "
                        f"Use the string form '{tool_name}' to enable it."
                    )
                json_tool = cast(JSONObject, tool)
                normalized_tools.append(json_tool)
                continue

            raise ValueError(
                "AgentResponses request field 'tools' must contain JSON objects "
                "or built-in connector tool names."
            )

        updated["tools"] = normalized_tools
        if enabled_builtin_connectors and updated.get("tool_choice") in (None, "auto"):
            updated["tool_choice"] = "required"
        if (
            enabled_builtin_connectors
            and updated.get("tool_choice") != "none"
            and updated.get("stream") is True
        ):
            updated["stream"] = False

        return PreparedConnectorTools(
            request=updated,
            enabled_builtin_connectors=frozenset(enabled_builtin_connectors),
        )

    return PreparedConnectorTools(
        request=updated,
        enabled_builtin_connectors=frozenset(),
    )


def extract_builtin_connector_tool_calls(
    response: JSONObject, enabled_builtin_connectors: Collection[str]
) -> list[ConnectorToolCall]:
    """Return calls only if every function call is an enabled built-in."""
    output = response.get("output")
    # No output list means there are no tool calls for the runtime to handle.
    if not isinstance(output, Sequence) or isinstance(output, str):
        return []

    tool_calls: list[ConnectorToolCall] = []
    for item in output:
        # Responses can contain messages/reasoning alongside tool calls.
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue

        name = item.get("name")
        # A client or disabled tool call belongs to AgentApp, not the runtime.
        if not isinstance(name, str) or name not in enabled_builtin_connectors:
            return []

        call_id = item.get("call_id")
        # Connector outputs must reference the model's function_call call_id.
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(
                f"Connector function_call '{name}' requires a non-empty call_id."
            )

        arguments = item["arguments"]
        # Providers may return arguments as either a JSON object or JSON string.
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        tool_calls.append(
            ConnectorToolCall(
                name=name,
                call_id=call_id,
                arguments=cast(JSONObject, arguments),
            )
        )

    return tool_calls
