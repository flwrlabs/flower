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
    `responses.create` call. AgentApp-owned loops opt in again on later calls
    by passing built-in connector names in `tools`.
    """

    request: JSONObject
    enabled_builtin_connectors: frozenset[str]
    followup_tools: list[JSONObject] | None


def with_builtin_connector_tools(request: JSONObject) -> PreparedConnectorTools:
    """Return request with requested built-in connector function tools enabled."""
    updated = dict(request)
    tools = request.get("tools")

    if tools is None:
        return PreparedConnectorTools(
            request=updated,
            enabled_builtin_connectors=frozenset(),
            followup_tools=None,
        )

    if isinstance(tools, Sequence) and not isinstance(tools, str):
        enabled_builtin_connectors: set[str] = set()
        followup_tools: list[JSONObject] = []
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
                # connector names are reserved so the follow-up turn can remove
                # only runtime-injected connector tools.
                if isinstance(tool_name, str) and has_builtin_connector(tool_name):
                    raise ValueError(
                        f"Built-in connector tool name '{tool_name}' is reserved. "
                        f"Use the string form '{tool_name}' to enable it."
                    )
                json_tool = cast(JSONObject, tool)
                followup_tools.append(json_tool)
                normalized_tools.append(json_tool)
                continue

            raise ValueError(
                "AgentResponses request field 'tools' must contain JSON objects "
                "or built-in connector tool names."
            )

        updated["tools"] = normalized_tools
        if enabled_builtin_connectors and updated.get("stream") is True:
            # The runtime needs the complete first response to execute connector
            # calls, then restores the caller's stream setting on the follow-up.
            updated["stream"] = False

        return PreparedConnectorTools(
            request=updated,
            enabled_builtin_connectors=frozenset(enabled_builtin_connectors),
            followup_tools=followup_tools if followup_tools else None,
        )

    return PreparedConnectorTools(
        request=updated,
        enabled_builtin_connectors=frozenset(),
        followup_tools=None,
    )


def extract_builtin_connector_tool_calls(
    response: JSONObject, enabled_builtin_connectors: Collection[str]
) -> list[ConnectorToolCall]:
    """Extract enabled built-in connector function calls from one model response."""
    if not enabled_builtin_connectors:
        return []

    output = response.get("output")
    if not isinstance(output, Sequence) or isinstance(output, str):
        return []

    tool_calls = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue

        name = item.get("name")
        # Use the per-request enabled set instead of the global built-in
        # registry. A model can emit arbitrary function_call names; the runtime
        # should only consume connector calls the AgentApp explicitly enabled.
        if not isinstance(name, str) or name not in enabled_builtin_connectors:
            continue

        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(
                f"Connector function_call '{name}' requires a non-empty call_id."
            )

        # Providers may return arguments as a JSON string or an already-decoded object.
        arguments = item.get("arguments")
        if isinstance(arguments, dict):
            parsed_arguments = cast(JSONObject, arguments)
        elif isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as err:
                raise ValueError("Function-call arguments are malformed JSON.") from err

            if not isinstance(parsed, dict):
                raise ValueError(
                    "Function-call arguments must decode to a JSON object."
                )
            parsed_arguments = cast(JSONObject, parsed)
        else:
            raise ValueError("Function-call arguments must be a JSON object or string.")
        tool_calls.append(
            ConnectorToolCall(name=name, call_id=call_id, arguments=parsed_arguments)
        )

    return tool_calls
