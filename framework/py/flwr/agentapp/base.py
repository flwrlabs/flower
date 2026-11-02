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
"""AgentApp abstract base classes and callable type alias."""


from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from flwr.app import Context
from flwr.supercore.typing import JSONObject

# OpenResponses item types and streaming events used for run summaries.
# See https://www.openresponses.org/reference
# Schema: https://www.openresponses.org/openapi/openapi.json
RUN_SUMMARY_EVENT_TYPES: tuple[str, ...] = (
    "message",
    "response.completed",
    "response.incomplete",
    "function_call",
    "function_call_output",
)


class AgentConnectors(ABC):
    """Abstract base class for AgentApp connector execution."""

    @abstractmethod
    def tools(self, names: Sequence[str], *, refresh: bool = False) -> list[JSONObject]:
        """Return model-facing schemas for static and discovered connector tools.

        Parameters
        ----------
        names : Sequence[str]
            Connector references to include, in order.
        refresh : bool (default: False)
            Rediscover the selected dynamic connectors. Schemas are otherwise
            cached for this agent runtime; static connectors require no discovery.

        Returns
        -------
        list[JSONObject]
            Independent copies of the model-facing tool schemas.
        """

    @abstractmethod
    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute one model function_call and return a function_call_output item."""


class AgentEvents(ABC):
    """Abstract base class for AgentApp run events."""

    @abstractmethod
    def get_trace(
        self,
        *,
        event_types: Sequence[str] | None = None,
        after_event_id: int | None = None,
        before_event_id: int | None = None,
        last_n: int | None = None,
    ) -> list[JSONObject]:
        """Get matching events from all runs in the current run series.

        Parameters
        ----------
        event_types : Optional[Sequence[str]] (default: None)
            Filter by the given event types. If unset, include all event types.
            Use `RUN_SUMMARY_EVENT_TYPES` for common summary inputs.
        after_event_id : Optional[int] (default: None)
            Return events with IDs greater than this value.
        before_event_id : Optional[int] (default: None)
            Return events with IDs less than this value.
        last_n : Optional[int] (default: None)
            Return the newest N matching events in ID order. If unset or zero,
            return all matching events.

        Returns
        -------
        list[JSONObject]
            Matching event envelopes.
        """

    @abstractmethod
    def emit(self, event: JSONObject) -> None:
        """Emit one structured run event."""

    @abstractmethod
    def get_compacted_contexts(
        self, *, limit: int = 1, before_context_id: int | None = None
    ) -> list[JSONObject]:
        """Return the newest summaries."""

    @abstractmethod
    def put_compacted_context(
        self, *, through_event_id: int, text: str, token_count: int
    ) -> None:
        """Store a summary."""


class AgentGrid(ABC):
    """Model-facing access to the federation Grid."""

    @abstractmethod
    def tools(self) -> list[JSONObject]:
        """Return model-facing Grid tool schemas."""

    @abstractmethod
    def call(self, tool_call: JSONObject) -> JSONObject:
        """Execute one Grid function_call and return a function_call_output item."""


class AgentSession(ABC):
    """Abstract base class for AgentApp runtime capabilities."""

    @property
    @abstractmethod
    def prompt(self) -> str:
        """Return the initial prompt for this AgentApp run."""

    @property
    @abstractmethod
    def connectors(self) -> AgentConnectors:
        """Connector tool schema and execution API."""

    @property
    @abstractmethod
    def events(self) -> AgentEvents:
        """Frontend-visible structured run event API."""

    @property
    @abstractmethod
    def grid(self) -> AgentGrid:
        """Model-facing federation Grid API."""


AgentAppCallable = Callable[[AgentSession, Context], None]
