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
from flwr.proto.control_pb2 import Agent  # pylint: disable=E0611
from flwr.proto.file_pb2 import File, FileMetadata  # pylint: disable=E0611
from flwr.supercore.constant import MAX_SPACE_PAGE_SIZE
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

    def get_connections(self) -> list[JSONObject]:
        """List the current Space agent's connections without credentials.

        Returns
        -------
        list[JSONObject]
            Connection metadata with connector_id, connector_ref, display_name,
            and description.

        Raises
        ------
        NotImplementedError
            If the session does not support agent-owned connections.
        """
        raise NotImplementedError("Agent-owned connections require a Space session.")

    @abstractmethod
    def tools(self, names: Sequence[str], refresh: bool = False) -> list[JSONObject]:
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

        Notes
        -----
        Space sessions include only agent-owned connections and built-ins. Their
        credential-backed tool names identify a connection; pass those names
        unchanged to ``call``.
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


class AgentSpace(ABC):
    """Read and publish messages and items in the authenticated agent's Space."""

    @abstractmethod
    def get_messages(
        self,
        *,
        limit: int = 100,
        before_id: int | None = None,
        after_id: int | None = None,
        unread_only: bool = False,
    ) -> list[JSONObject]:
        """Read messages and advance this agent's read cursor.

        Parameters
        ----------
        limit : int (default: 100)
            Maximum number of messages to return, between 1 and 100.
        before_id : int | None (default: None)
            Exclusive upper message ID bound.
        after_id : int | None (default: None)
            Exclusive lower ID bound; selects the oldest matching page.
        unread_only : bool (default: False)
            Select the oldest page after this agent's read cursor.

        Returns
        -------
        list[JSONObject]
            Messages in ascending ID order, with id, space_id, author, created_at,
            and content. Author contains either flwr_aid or agent_id.

        Notes
        -----
        Reads select the newest page by default. Every read advances the cursor
        through the newest returned message, including history reads.
        """

    @abstractmethod
    def get_items(
        self,
        *,
        limit: int = MAX_SPACE_PAGE_SIZE,
        before_id: int | None = None,
        after_id: int | None = None,
        unread_only: bool = False,
    ) -> list[JSONObject]:
        """Read SpaceItems and advance this agent's item read cursor.

        Parameters
        ----------
        limit : int (default: 100)
            Maximum number of items to return, between 1 and 100.
        before_id : int | None (default: None)
            Exclusive upper item ID bound.
        after_id : int | None (default: None)
            Exclusive lower item ID bound; selects the oldest matching page.
        unread_only : bool (default: False)
            Select the oldest page after this agent's item read cursor.

        Returns
        -------
        list[JSONObject]
            Items in ascending ID order, with id, space_id, type, created_at,
            and content. Message items also have an author containing either
            flwr_aid or agent_id.

        Notes
        -----
        Reads select the newest page by default and advance the item cursor
        through the newest returned ID.
        """

    @abstractmethod
    def send_item(self, content: str) -> JSONObject:
        """Publish a message item and return its server-assigned fields.

        Parameters
        ----------
        content : str
            Nonempty message text to publish.

        Returns
        -------
        JSONObject
            The created item, including its type and server-assigned fields.

        Notes
        -----
        Only message items can be sent through the current Runtime endpoint.
        """

    @abstractmethod
    def tools(self) -> list[JSONObject]:
        """Return the model-facing send_space_message tool schema."""

    @abstractmethod
    def call(self, tool_call: JSONObject) -> JSONObject:
        """Publish a message and return a function_call_output item."""


class AgentFiles(ABC):
    """Persistent files owned by the authenticated agent."""

    @abstractmethod
    def get(self, key: str) -> File | None:
        """Return the current file content and metadata.

        Parameters
        ----------
        key : str
            File key within the agent's scope.

        Returns
        -------
        File | None
            Content and metadata, or None if absent or deleted.
        """

    @abstractmethod
    def put(
        self, key: str, data: bytes, *, expected_revision: int | None = None
    ) -> int:
        """Create or update a file, preserving its previous revisions.

        Parameters
        ----------
        key : str
            Non-empty, non-whitespace file key, preserved unchanged.
        data : bytes
            File content to store.
        expected_revision : int | None (default: None)
            None means create-only and fails if a live file exists. Otherwise,
            the current live revision must match for the update to succeed.

        Returns
        -------
        int
            The new revision ID.
        """

    @abstractmethod
    def list(
        self,
        *,
        prefix: str = "",
        start_after: str = "",
        limit: int | None = None,
    ) -> tuple[list[FileMetadata], str]:
        """List live file metadata in lexicographic key order.

        Parameters
        ----------
        prefix : str (default: "")
            Include only keys starting with this prefix.
        start_after : str (default: "")
            Exclusive key cursor; empty starts from the beginning.
        limit : int | None (default: None)
            Maximum number of entries. None returns all matches; zero returns
            an empty page.

        Returns
        -------
        tuple[list[FileMetadata], str]
            Metadata and the next cursor, which is empty when no more remain.
        """

    @abstractmethod
    def delete(self, key: str, *, expected_revision: int) -> None:
        """Delete a file from the current view, retaining its history.

        Parameters
        ----------
        key : str
            File key within the agent's scope.
        expected_revision : int
            Must match the current live revision; missing or deleted files
            also cause a revision conflict.
        """


class AgentSession(ABC):
    """Abstract base class for AgentApp runtime capabilities."""

    @property
    def identity(self) -> Agent | None:
        """Return this agent's stored identity, or None when unavailable."""
        return None

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

    @property
    @abstractmethod
    def files(self) -> AgentFiles:
        """Persistent file API."""

    @property
    def space(self) -> AgentSpace | None:
        """Space messaging API, or None when unavailable for this session."""
        return None


AgentAppCallable = Callable[[AgentSession, Context], None]
