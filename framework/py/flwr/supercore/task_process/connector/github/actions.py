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
"""Build GitHub action definitions from the authenticated MCP catalog."""

from typing import cast

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition
from . import mcp


def load_actions(credentials: JSONObject) -> tuple[ActionDefinition, ...]:
    """Discover every GitHub tool and preserve its remote schema and name."""
    tools = cast(list[JSONObject], mcp.request(None, {}, credentials))
    return tuple(
        ActionDefinition(
            name=cast(str, tool["name"]),
            description=cast(str, tool.get("description", "")),
            access=(
                ActionAccess.READ
                if cast(JSONObject, tool.get("annotations", {})).get("readOnlyHint")
                else ActionAccess.WRITE
            ),
            input_schema=cast(JSONObject, tool["inputSchema"]),
        )
        for tool in tools
    )
