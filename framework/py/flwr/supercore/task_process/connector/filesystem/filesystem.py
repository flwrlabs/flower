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
"""Built-in filesystem connector implementation."""

from __future__ import annotations

import os
import stat

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject

from ..http import ConnectorApiError

FILESYSTEM_CONNECTOR_NAME = "filesystem"
FILESYSTEM_ALLOWED_DIRS_ENV = "FLWR_FILESYSTEM_ALLOWED_DIRS"

_MAX_DIRECTORY_ENTRIES = 1000
_MAX_FILE_BYTES = 1024 * 1024
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_PLATFORM_SUPPORTED = os.name != "nt" and _O_NOFOLLOW != 0


class FilesystemApiError(ConnectorApiError):
    """Secret-safe file system access failure."""

    provider = "Filesystem"


def make_filesystem_tool() -> JSONObject:
    """Return the filesystem function tool schema."""
    return {
        "type": "function",
        "name": FILESYSTEM_CONNECTOR_NAME,
        "description": (
            "Read files and directories from the local file system within "
            "configured allowed directories."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list_directory", "read_file"],
                    "description": (
                        "The filesystem action to perform. 'list_directory' lists "
                        "entries of a directory sorted by name. 'read_file' reads "
                        "one UTF-8 text file and returns its content."
                    ),
                },
                "path": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Absolute path inside one of the allowed directories."
                    ),
                },
            },
            "required": ["action", "path"],
            "additionalProperties": False,
        },
    }


def invoke_filesystem_provider(
    action: str, path: str, *, usage_recorder: TaskUsageRecorder
) -> JSONObject:
    """Execute one filesystem action."""
    del usage_recorder
    if not _PLATFORM_SUPPORTED:
        raise FilesystemApiError("unsupported_platform")
    allowed = _allowed_dirs()
    if action == "list_directory":
        return _list_directory(path, allowed)
    if action == "read_file":
        return _read_file(path, allowed)
    raise FilesystemApiError("invalid_action")


def _list_directory(path: str, allowed: list[str]) -> JSONObject:
    """List entries in an allowed directory."""
    fd, _ = _open_sandboxed(path, allowed, os.O_RDONLY | _O_DIRECTORY)
    try:
        items: list[JSONObject] = []
        with os.scandir(fd) as entries:
            for index, entry in enumerate(entries):
                if index >= _MAX_DIRECTORY_ENTRIES:
                    raise FilesystemApiError("too_many_entries")
                mode = entry.stat(follow_symlinks=False).st_mode
                items.append(
                    {
                        "name": entry.name,
                        "type": (
                            "directory"
                            if stat.S_ISDIR(mode)
                            else "file" if stat.S_ISREG(mode) else "other"
                        ),
                    }
                )
        items.sort(key=lambda item: str(item["name"]))
        return {"entries": items}
    except OSError:
        raise FilesystemApiError("access_denied") from None
    finally:
        os.close(fd)


def _read_file(path: str, allowed: list[str]) -> JSONObject:
    """Read one UTF-8 text file inside an allowed directory."""
    fd, resolved = _open_sandboxed(path, allowed, os.O_RDONLY | _O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise FilesystemApiError("not_a_file")
        handle = os.fdopen(fd, "rb")
        fd = -1
        with handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
    except OSError:
        raise FilesystemApiError("access_denied") from None
    finally:
        if fd >= 0:
            os.close(fd)
    if len(raw) > _MAX_FILE_BYTES:
        raise FilesystemApiError("file_too_large")
    try:
        return {"content": raw.decode("utf-8"), "path": resolved}
    except UnicodeDecodeError:
        raise FilesystemApiError("access_denied") from None


def _open_sandboxed(path: str, allowed: list[str], flags: int) -> tuple[int, str]:
    """Resolve and open a path beneath an allowed directory."""
    if not os.path.isabs(path):
        raise FilesystemApiError("access_denied")
    resolved = os.path.realpath(path)
    if not any(_is_beneath(resolved, root) for root in allowed):
        raise FilesystemApiError("access_denied")
    try:
        # O_NOFOLLOW closes the common race where the resolved target is replaced
        # with a symlink before it is opened. Parent-directory rename races require
        # hostile filesystem control and are outside this connector's threat model.
        return os.open(resolved, flags | _O_NOFOLLOW), resolved
    except OSError:
        raise FilesystemApiError("access_denied") from None


def _is_beneath(path: str, root: str) -> bool:
    """Return whether path is root or one of its descendants."""
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        return False


def _allowed_dirs() -> list[str]:
    """Parse allowed directories from the environment variable."""
    paths = os.getenv(FILESYSTEM_ALLOWED_DIRS_ENV, "").split(os.pathsep)
    if not paths or not all(path and os.path.isabs(path) for path in paths):
        raise FilesystemApiError("invalid_config")
    roots = [os.path.realpath(path) for path in paths]
    if not all(os.path.isdir(root) for root in roots):
        raise FilesystemApiError("invalid_config")
    return roots
