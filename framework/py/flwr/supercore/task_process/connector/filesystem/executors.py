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
"""File system action executors."""

import os

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError
from ..json_utils import require_string


class FilesystemApiError(ConnectorApiError):
    """Secret-safe file system access failure."""

    provider = "Filesystem"


_MAX_DIRECTORY_ENTRIES = 1000
_MAX_FILE_BYTES = 1024 * 1024


def list_directory(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List entries in an allowed directory."""
    path = require_string(arguments.get("path"), "Filesystem", "path")
    allowed = _allowed_dirs(context)
    resolved = _safe_resolve(path, allowed)
    if not os.path.isdir(resolved):
        raise FilesystemApiError("not_a_directory")
    try:
        raw = sorted(os.listdir(resolved))
    except OSError:
        raise FilesystemApiError("access_denied") from None
    if len(raw) > _MAX_DIRECTORY_ENTRIES:
        raise FilesystemApiError("too_many_entries")
    items: list[JSONObject] = []
    for name in raw:
        entry_path = os.path.join(resolved, name)
        items.append(
            {
                "name": name,
                "type": "directory" if os.path.isdir(entry_path) else "file",
            }
        )
    return {"entries": items}


def read_file(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Read one UTF-8 text file inside an allowed directory."""
    path = require_string(arguments.get("path"), "Filesystem", "path")
    allowed = _allowed_dirs(context)
    resolved = _safe_resolve(path, allowed)
    if not os.path.isfile(resolved):
        raise FilesystemApiError("not_a_file")
    st = os.stat(resolved)
    if st.st_size > _MAX_FILE_BYTES:
        raise FilesystemApiError("file_too_large")
    try:
        with open(resolved, encoding="utf-8") as handle:
            content = handle.read(_MAX_FILE_BYTES)
    except (OSError, UnicodeDecodeError):
        raise FilesystemApiError("access_denied") from None
    return {"content": content, "path": resolved}


EXECUTORS: dict[str, ConnectorExecutor] = {
    "list_directory": list_directory,
    "read_file": read_file,
}


def _safe_resolve(path: str, allowed: list[str]) -> str:
    """Resolve and sandbox an absolute path."""
    if not os.path.isabs(path):
        raise FilesystemApiError("access_denied")
    real = os.path.realpath(path)
    for root in allowed:
        resolved_root = os.path.realpath(root)
        if not resolved_root.endswith(os.sep):
            resolved_root += os.sep
        if real.startswith(resolved_root) or real == resolved_root.rstrip(os.sep):
            return real
    raise FilesystemApiError("access_denied")


def _allowed_dirs(context: ConnectorExecutionContext) -> list[str]:
    """Extract the list of allowed directories from connector config."""
    value = context.config.get("allowed_dirs")
    if not isinstance(value, list) or not all(
        isinstance(d, str) and d.strip() for d in value
    ):
        raise FilesystemApiError("invalid_config")
    return [d.strip() for d in value]
