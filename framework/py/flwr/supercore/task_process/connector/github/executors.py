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
"""GitHub tool discovery and execution through MCP."""

from flwr.supercore.typing import JSONObject, JSONValue

from ..definition import ConnectorExecutionContext
from ..json_utils import require_int_range
from . import mcp


def discover_tools(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONValue:
    """Discover tools using the credential-bound connector worker."""
    del arguments
    return mcp.discover_tools(context.credentials)


def execute(
    name: str, arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONValue:
    """Forward a discovered GitHub tool without narrowing its MCP schema."""
    remote_name = name.removeprefix("github_")
    arguments = dict(arguments)
    # Preserve the pagination spelling accepted by the original Flower tool.
    if remote_name == "search_code" and "per_page" in arguments:
        if "perPage" in arguments:
            raise ValueError("Use only one of GitHub per_page and perPage.")
        arguments["perPage"] = require_int_range(
            arguments.pop("per_page"), "GitHub", "per_page", maximum=100
        )
    return mcp.call_tool(remote_name, arguments, context.credentials)
