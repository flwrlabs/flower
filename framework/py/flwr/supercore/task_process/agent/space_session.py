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

from flwr.agentapp import AgentFiles, AgentGrid
from flwr.proto.file_pb2 import File, FileMetadata  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    DeleteFileRequest,
    GetFileRequest,
    GetSpaceEventsRequest,
    ListFilesRequest,
    PullTaskInputResponse,
    PutFileRequest,
)
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_process.connector.registry import (
    get_connector_ref,
    has_builtin_connector,
)
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_loads
from flwr.superlink.grid import HttpGrid

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


class SpaceAgentFiles(AgentFiles):
    """Access the agent's persistent files through the Runtime API."""

    def __init__(self, stub: RuntimeHttpClient) -> None:
        self._stub = stub

    def get(self, key: str) -> File | None:
        """Return the current file, or None if absent or deleted."""
        response = self._stub.GetFile(GetFileRequest(key=key))
        return response.file if response.HasField("file") else None

    def put(
        self, key: str, data: bytes, *, expected_revision: int | None = None
    ) -> int:
        """Create or update a file using the supplied revision precondition."""
        response = self._stub.PutFile(
            PutFileRequest(key=key, data=data, expected_revision=expected_revision)
        )
        return response.revision

    def list(
        self,
        *,
        prefix: str = "",
        start_after: str = "",
        limit: int | None = None,
    ) -> tuple[list[FileMetadata], str]:
        """Return one page of metadata and its continuation cursor."""
        response = self._stub.ListFiles(
            ListFilesRequest(prefix=prefix, start_after=start_after, limit=limit)
        )
        return list(response.files), response.next_start_after

    def delete(self, key: str, *, expected_revision: int) -> None:
        """Delete a file only if the current revision matches."""
        self._stub.DeleteFile(
            DeleteFileRequest(key=key, expected_revision=expected_revision)
        )


class SpaceAgentSession(RuntimeAgentSession):
    """AgentSession with Space messaging and persistent files."""

    def __init__(
        self,
        stub: RuntimeHttpClient,
        events: SpaceEvents,
        task_input: PullTaskInputResponse,
    ) -> None:
        runtime = AgentRuntime(
            stub=stub,
            run_id=task_input.run.run_id,
            task_id=task_input.task_id,
            events=events,
        )
        super().__init__(
            prompt="",
            connectors=_SpaceAgentConnectors(runtime),
            events=events,
            grid=_SpaceAgentGrid(),
            space=RuntimeAgentSpace(stub, events),
            identity=task_input.agent if task_input.HasField("agent") else None,
        )
        self._files = SpaceAgentFiles(stub)

    @property
    def files(self) -> AgentFiles:
        """Return the agent's persistent file API."""
        return self._files


def create_space_session(
    grid: HttpGrid, events: SpaceEvents, task_input: PullTaskInputResponse
) -> SpaceAgentSession:
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
    SpaceAgentSession
        Session with built-in connectors and federation Grid tools disabled.
    """
    stub = grid._runtime_client  # pylint: disable=protected-access
    return SpaceAgentSession(stub, events, task_input)
