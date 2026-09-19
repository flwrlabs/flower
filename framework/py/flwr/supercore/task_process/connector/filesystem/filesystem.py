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
    if os.name == "nt":
        raise FilesystemApiError("unsupported_platform")
    allowed = _allowed_dirs()
    if action == "list_directory":
        return _list_directory(path, allowed)
    if action == "read_file":
        return _read_file(path, allowed)
    raise FilesystemApiError("invalid_action")


def _list_directory(path: str, allowed: list[str]) -> JSONObject:
    """List entries in an allowed directory.

    The directory is opened via a per-component O_NOFOLLOW walk
    (_open_sandboxed) and the descriptor is kept open through the entry
    metadata loop so stat lookups are fd-relative and cannot escape the
    sandbox. Entries are collected through a bounded scan so directories
    with more than _MAX_DIRECTORY_ENTRIES entries fail fast instead of
    being materialized in memory.
    """
    resolved = _safe_resolve(path, allowed)
    try:
        fd = _open_sandboxed(resolved, os.O_RDONLY | _O_DIRECTORY | _O_NONBLOCK)
    except OSError:
        raise FilesystemApiError("access_denied") from None
    try:
        entries = _bounded_listdir(fd)
        items: list[JSONObject] = []
        for name in sorted(entries):
            try:
                entry_st = os.stat(name, follow_symlinks=False, dir_fd=fd)
            except OSError:
                continue
            items.append(
                {
                    "name": name,
                    "type": (
                        "directory"
                        if stat.S_ISDIR(entry_st.st_mode)
                        else "file" if stat.S_ISREG(entry_st.st_mode) else "other"
                    ),
                }
            )
        return {"entries": items}
    finally:
        os.close(fd)


def _bounded_listdir(fd: int) -> list[str]:
    """List entry names of an open directory fd, capped at the entry limit."""
    names: list[str] = []
    with os.scandir(fd) as iterator:
        for entry in iterator:
            names.append(entry.name)
            if len(names) > _MAX_DIRECTORY_ENTRIES:
                raise FilesystemApiError("too_many_entries")
    return names


def _read_file(path: str, allowed: list[str]) -> JSONObject:
    """Read one UTF-8 text file inside an allowed directory.

    The file is opened via a per-component O_NOFOLLOW walk so no
    intermediate directory can be swapped for a symlink between
    _safe_resolve and the open syscall; O_NONBLOCK avoids hanging on
    named pipes. fstat on the returned fd validates the file is a
    regular file (S_ISREG).
    """
    resolved = _safe_resolve(path, allowed)
    fd = None
    try:
        fd = _open_sandboxed(resolved, os.O_RDONLY | _O_NONBLOCK)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise FilesystemApiError("not_a_file")
        if st.st_size > _MAX_FILE_BYTES:
            raise FilesystemApiError("file_too_large")
        raw = _read_all(fd)
    except OSError:
        raise FilesystemApiError("access_denied") from None
    finally:
        if fd is not None:
            os.close(fd)
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise FilesystemApiError("access_denied") from None
    return {"content": content, "path": resolved}


def _read_all(fd: int) -> bytes:
    """Read from fd until EOF, capped at _MAX_FILE_BYTES.

    A single os.read() may return fewer bytes than requested before EOF,
    and a file may grow after the fstat size check, so keep reading in
    bounded chunks and reject once the cap is exceeded.
    """
    chunks: list[bytes] = []
    remaining = _MAX_FILE_BYTES
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise FilesystemApiError("file_too_large")
    return b"".join(chunks)


def _open_sandboxed(resolved: str, flags: int) -> int:
    """Open resolved by walking every component with O_NOFOLLOW.

    Each component is opened relative to the previous directory fd, so a
    concurrent swap of any intermediate directory for a symlink is
    rejected by the kernel instead of followed.
    """
    parts = [c for c in resolved.split(os.sep) if c]
    if not parts:
        return os.open(os.sep, flags | _O_NOFOLLOW | _O_DIRECTORY)
    fd = os.open(os.sep, os.O_RDONLY)
    try:
        for part in parts[:-1]:
            new_fd = os.open(part, os.O_RDONLY | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        new_fd = os.open(parts[-1], flags | _O_NOFOLLOW, dir_fd=fd)
        os.close(fd)
        return new_fd
    except BaseException:
        os.close(fd)
        raise


def _safe_resolve(path: str, allowed: list[str]) -> str:
    """Resolve and sandbox an absolute path.

    realpath canonicalization catches symlinks and '..' components before
    the path is used. _open_sandboxed then re-opens the result component
    by component with O_NOFOLLOW, closing the race window for every path
    level.
    """
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


def filesystem_is_configured() -> bool:
    """Return whether filesystem access is available and configured."""
    if os.name == "nt":
        return False
    raw = os.getenv(FILESYSTEM_ALLOWED_DIRS_ENV, "")
    return any(d.strip() for d in raw.split(os.pathsep))


def _allowed_dirs() -> list[str]:
    """Parse allowed directories from the environment variable."""
    raw = os.getenv(FILESYSTEM_ALLOWED_DIRS_ENV, "")
    dirs = [d.strip() for d in raw.split(os.pathsep) if d.strip()]
    if not dirs:
        raise FilesystemApiError("invalid_config")
    return dirs
