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

from collections.abc import Sequence
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


def with_builtin_connector_tools(request: JSONObject) -> JSONObject:
    """Return request with requested built-in connector function tools enabled."""
    updated = dict(request)
    tools = request.get("tools")

    if tools is None:
        return updated

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
                normalized_tools.append(cast(JSONObject, tool))
                continue

            raise ValueError(
                "AgentResponses request field 'tools' must contain JSON objects "
                "or built-in connector tool names."
            )

        updated["tools"] = normalized_tools
        return updated

    return updated
