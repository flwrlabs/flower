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

from flwr.agentapp import AgentConnectors
from flwr.supercore.typing import JSONObject
from flwr.superlink.grid import HttpGrid

from .grid import RuntimeAgentGrid
from .session import RuntimeAgentEvents, RuntimeAgentSession


class SpaceAgentEvents(RuntimeAgentEvents):
    """Publish events and read traces without exposing RunSeries compaction."""

    def get_compacted_contexts(
        self, *, limit: int = 1, before_context_id: int | None = None
    ) -> list[JSONObject]:
        """Reject compacted-context reads for Space sessions."""
        raise NotImplementedError("Space sessions do not support compacted contexts.")

    def put_compacted_context(
        self, *, through_event_id: int, text: str, token_count: int
    ) -> None:
        """Reject compacted-context writes for Space sessions."""
        raise NotImplementedError("Space sessions do not support compacted contexts.")


class _SpaceAgentConnectors(AgentConnectors):
    """Disable connector and automation tools for Space sessions."""

    def tools(self, names: Sequence[str], *, refresh: bool = False) -> list[JSONObject]:
        """Expose no connector or automation tool schemas."""
        return []

    def call(self, tool_call: JSONObject) -> JSONObject:
        """Reject connector and automation calls before making Runtime requests."""
        raise NotImplementedError("Space sessions do not support connector tools.")


def create_space_session(
    grid: HttpGrid, events: SpaceAgentEvents, node_id: int
) -> RuntimeAgentSession:
    """Initialize a session whose AgentApp retrieves its own Space messages.

    Parameters
    ----------
    grid : HttpGrid
        Task-authenticated Runtime connection.
    events : SpaceAgentEvents
        Task event publisher shared with the worker lifecycle.
    node_id : int
        Node identity for the session's Grid capabilities.

    Returns
    -------
    RuntimeAgentSession
        Session with connectors, automations, and compacted contexts disabled.
    """
    return RuntimeAgentSession(
        prompt="",
        connectors=_SpaceAgentConnectors(),
        events=events,
        grid=RuntimeAgentGrid(grid, events, node_id),
    )
