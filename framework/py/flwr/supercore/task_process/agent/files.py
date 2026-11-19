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
"""Persistent file access for AgentApps."""

from flwr.agentapp import AgentFiles
from flwr.proto.file_pb2 import File, FileMetadata  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    DeleteFileRequest,
    GetFileRequest,
    ListFilesRequest,
    PutFileRequest,
)
from flwr.supercore.runtime import RuntimeHttpClient


class RuntimeAgentFiles(AgentFiles):
    """Access the agent's persistent files through the Runtime API."""

    def __init__(self, client: RuntimeHttpClient) -> None:
        self._client = client

    def get(self, key: str) -> File | None:
        """Return the current file, or None if absent or deleted."""
        response = self._client.GetFile(GetFileRequest(key=key))
        return response.file if response.HasField("file") else None

    def put(
        self, key: str, data: bytes, *, expected_revision: int | None = None
    ) -> int:
        """Create or update a file using the supplied revision precondition."""
        response = self._client.PutFile(
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
        response = self._client.ListFiles(
            ListFilesRequest(prefix=prefix, start_after=start_after, limit=limit)
        )
        return list(response.files), response.next_start_after

    def delete(self, key: str, *, expected_revision: int) -> None:
        """Delete a file only if the current revision matches."""
        self._client.DeleteFile(
            DeleteFileRequest(key=key, expected_revision=expected_revision)
        )
