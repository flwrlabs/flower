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

from collections.abc import Sequence
from typing import cast

from flwr.agentapp import AgentGrid
from flwr.proto.runtime_pb2 import PullTaskInputResponse  # pylint: disable=E0611
from flwr.supercore.task_process.connector.registry import (
    get_connector_ref,
    has_builtin_connector,
)
from flwr.supercore.typing import JSONObject
from flwr.superlink.grid import HttpGrid

from .session import (
    AgentRuntime,
    RuntimeAgentConnectors,
    RuntimeAgentEvents,
    RuntimeAgentSession,
)
from .space import RuntimeAgentSpace


class SpaceAgentEvents(RuntimeAgentEvents):
    """Disable trace reads while retaining event publication and compaction."""

    def get_trace(
        self,
        *,
        event_types: Sequence[str] | None = None,
        after_event_id: int | None = None,
        before_event_id: int | None = None,
        last_n: int | None = None,
    ) -> list[JSONObject]:
        """Reject trace reads before making Runtime requests."""
        raise NotImplementedError("Space sessions do not support trace reads.")


class _SpaceAgentConnectors(RuntimeAgentConnectors):
    """Expose built-in connectors without federation-owned tools or automations."""

    def tools(self, names: Sequence[str], *, refresh: bool = False) -> list[JSONObject]:
        """Return schemas for the requested built-in connectors."""
        return super().tools(
            [name for name in names if has_builtin_connector(name)], refresh=refresh
        )

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute built-ins and reject unsupported tools before Runtime requests."""
        name = cast(str, tool_call["name"]).strip().lower()
        if not has_builtin_connector(get_connector_ref(name)):
            raise NotImplementedError(
                "Space sessions only support built-in connectors."
            )
        return super().call(tool_call)


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
    grid: HttpGrid, events: SpaceAgentEvents, task_input: PullTaskInputResponse
) -> RuntimeAgentSession:
    """Initialize a session whose AgentApp retrieves its own Space messages.

    Parameters
    ----------
    grid : HttpGrid
        Task-authenticated Runtime connection.
    events : SpaceAgentEvents
        Task event publisher shared with the worker lifecycle.
    task_input : PullTaskInputResponse
        Accepted Run and Task identifiers.

    Returns
    -------
    RuntimeAgentSession
        Session with built-in connectors and federation Grid tools disabled.
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
        connectors=_SpaceAgentConnectors(runtime),
        events=events,
        grid=_SpaceAgentGrid(),
        space=RuntimeAgentSpace(stub, events),
    )
