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
"""Space AgentApp session capabilities."""

import hashlib
from collections.abc import Sequence
from logging import ERROR
from typing import cast

from flwr.agentapp import AgentGrid
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetSpaceEventsRequest,
    ListAgentConnectorsRequest,
    PullTaskInputResponse,
)
from flwr.supercore import log
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.task_process.connector.registry import (
    get_connector_ref,
    has_builtin_connector,
)
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_loads
from flwr.superlink.grid import HttpGrid

from .files import RuntimeAgentFiles
from .session import (
    AgentRuntime,
    RuntimeAgentConnectors,
    RuntimeAgentEvents,
    RuntimeAgentSession,
)
from .space import RuntimeAgentSpace


class SpaceEvents(RuntimeAgentEvents):
    """Read Space events while retaining event publication and compaction."""

    def get_trace(
        self,
        *,
        event_types: Sequence[str] | None = None,
        after_event_id: int | None = None,
        before_event_id: int | None = None,
        last_n: int | None = None,
    ) -> list[JSONObject]:
        """Get matching events from the authenticated task's Space."""
        response = self._stub.GetSpaceEvents(
            GetSpaceEventsRequest(
                # An empty list means no filtering by event type.
                event_types=event_types or [],
                after_event_id=after_event_id,
                before_event_id=before_event_id,
                last_n=0 if last_n is None else last_n,
            )
        )
        return [
            {
                "id": event.id,
                "timestamp": event.timestamp,
                "run_id": event.run_id,
                "task_id": event.task_id,
                "event": event.event,
                "data": strict_json_loads(event.data),
            }
            for event in response.events
        ]


class _SpaceAgentConnectors(RuntimeAgentConnectors):
    """Expose built-ins and the Space agent's own connections."""

    def __init__(self, agent_runtime: AgentRuntime, stub: RuntimeHttpClient) -> None:
        super().__init__(agent_runtime)
        self._stub = stub
        self._tool_calls: dict[str, tuple[str, int]] = {}

    def get_connections(self) -> list[JSONObject]:
        """Read current connection metadata from the task-authenticated Runtime."""
        return [
            {
                "connector_id": connector.connector_id,
                "connector_ref": connector.connector_ref,
                "display_name": connector.display_name,
                "description": connector.description,
            }
            for connector in self._stub.ListAgentConnectors(
                ListAgentConnectorsRequest()
            ).connectors
        ]

    def tools(self, names: Sequence[str], refresh: bool = False) -> list[JSONObject]:
        """Return requested built-ins and tools for each owned provider connection."""
        connections = (
            self.get_connections()
            if any(
                not has_builtin_connector(name) and name != START_AUTOMATION_TOOL_NAME
                for name in names
            )
            else []
        )
        tools: list[JSONObject] = []
        for name in dict.fromkeys(names):
            if name == START_AUTOMATION_TOOL_NAME:
                continue
            if has_builtin_connector(name):
                tools.extend(
                    self._agent_runtime.get_connector_tools(name, refresh=refresh)
                )
                continue
            for connection in connections:
                if connection["connector_ref"] != name:
                    continue
                connector_id = cast(int, connection["connector_id"])
                try:
                    connection_tools = self._agent_runtime.get_connector_tools(
                        name, refresh=refresh, connector_id=connector_id
                    )
                except Exception:  # pylint: disable=broad-exception-caught
                    log(
                        ERROR,
                        "Skipping failed tool discovery for %s connection %d.",
                        name,
                        connector_id,
                    )
                    continue
                for tool in connection_tools:
                    tool_name = cast(str, tool["name"])
                    alias = _connection_tool_name(tool_name, connector_id)
                    previous = self._tool_calls.get(alias)
                    if previous is not None and previous != (tool_name, connector_id):
                        raise ValueError("Conflicting connection tool names.")
                    self._tool_calls[alias] = (tool_name, connector_id)
                    tool["name"] = alias
                    tool["description"] = (
                        f"{tool.get('description', '')}\n"
                        f"Connection: {connection['display_name']}."
                    )
                    tools.append(tool)
        return tools

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Route advertised tools to their connection and reject federation tools."""
        name = cast(str, tool_call["name"]).strip().lower()
        if name == START_AUTOMATION_TOOL_NAME:
            raise NotImplementedError("Space sessions do not support automations.")
        if has_builtin_connector(get_connector_ref(name)):
            return super().call(tool_call)
        if name not in self._tool_calls:
            raise ValueError(f"Tool '{name}' is not available in this Space session.")
        tool_name, connector_id = self._tool_calls[name]
        arguments = tool_call["arguments"]
        if isinstance(arguments, str):
            arguments = strict_json_loads(arguments)
        return self._agent_runtime.call_connector_with_events(
            name=tool_name,
            call_id=cast(str, tool_call["call_id"]),
            arguments=cast(JSONObject, arguments),
            connector_id=connector_id,
            event_name=name,
        )


def _connection_tool_name(name: str, connector_id: int) -> str:
    """Keep connection-specific function names stable and at most 64 characters."""
    suffix = f"__{connector_id}"
    if len(name) + len(suffix) > 64:
        suffix = f"_{hashlib.sha256(name.encode()).hexdigest()[:12]}{suffix}"
    return f"{name[:64 - len(suffix)]}{suffix}"


class _SpaceAgentGrid(AgentGrid):
    """Disable federation Grid tools for Space sessions."""

    def tools(self) -> list[JSONObject]:
        """Expose no federation Grid tool schemas."""
        return []

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Reject Grid calls before making Runtime requests."""
        raise NotImplementedError(
            "Space sessions do not support federation Grid tools."
        )


def create_space_session(
    grid: HttpGrid, events: SpaceEvents, task_input: PullTaskInputResponse
) -> RuntimeAgentSession:
    """Initialize a session whose AgentApp retrieves its own Space messages.

    Parameters
    ----------
    grid : HttpGrid
        Task-authenticated Runtime connection.
    events : SpaceEvents
        Task event publisher shared with the worker lifecycle.
    task_input : PullTaskInputResponse
        Accepted Run and Task identifiers.

    Returns
    -------
    RuntimeAgentSession
        Session with agent-owned connections, built-ins, and federation Grid disabled.
    """
    stub = grid._runtime_client  # pylint: disable=protected-access
    runtime = AgentRuntime(
        stub=stub,
        run_id=task_input.run.run_id,
        task_id=task_input.task_id,
        events=events,
    )
    return RuntimeAgentSession(
        prompt="",
        connectors=_SpaceAgentConnectors(runtime, stub),
        events=events,
        grid=_SpaceAgentGrid(),
        files=RuntimeAgentFiles(stub),
        space=RuntimeAgentSpace(stub, events),
        identity=task_input.agent if task_input.HasField("agent") else None,
    )
