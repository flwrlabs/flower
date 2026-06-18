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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from flwr.supercore.typing import JSONObject

from .registry import get_builtin_connector_tools, has_builtin_connector


@dataclass(frozen=True)
class ConnectorToolCall:
    """A model-requested connector function call."""

    name: str
    call_id: str
    arguments: JSONObject


def with_builtin_connector_tools(request: JSONObject) -> JSONObject:
    """Return request with built-in connector function tools enabled."""
    updated = dict(request)
    tools = request.get("tools")
    # Built-ins are appended once before the first model call.
    builtin_tools = get_builtin_connector_tools()

    if tools is None:
        updated["tools"] = builtin_tools
        return updated

    if isinstance(tools, Sequence) and not isinstance(tools, str):
        tool_list = list(tools)
        if all(isinstance(tool, dict) for tool in tool_list):
            updated["tools"] = [*cast(list[JSONObject], tool_list), *builtin_tools]
    return updated


def extract_builtin_connector_tool_calls(
    response: JSONObject,
) -> list[ConnectorToolCall]:
    """Extract built-in connector function calls from one model response."""
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
        if not isinstance(name, str) or not has_builtin_connector(name):
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
