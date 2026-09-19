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
_O_SEARCH = getattr(os, "O_SEARCH", 0) or getattr(os, "O_PATH", 0) or os.O_RDONLY
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
    """List entries in an allowed directory.

    The configured root is pinned by descriptor before the requested path is
    opened component by component with O_NOFOLLOW. The target descriptor stays
    open through metadata lookup so concurrent path replacement cannot escape
    the sandbox.
    """
    fd, _ = _open_sandboxed(path, allowed, os.O_RDONLY | _O_DIRECTORY | _O_NONBLOCK)
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

    The configured root is pinned by descriptor before the requested path is
    opened component by component with O_NOFOLLOW. O_NONBLOCK avoids hanging
    on named pipes, and fstat validates the target is a regular file.
    """
    fd = None
    try:
        fd, resolved = _open_sandboxed(path, allowed, os.O_RDONLY | _O_NONBLOCK)
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


def _open_sandboxed(path: str, allowed: list[str], flags: int) -> tuple[int, str]:
    """Open an absolute path relative to a pinned allowed-directory fd."""
    if not os.path.isabs(path):
        raise FilesystemApiError("access_denied")
    normalized = os.path.realpath(path)
    for root in allowed:
        normalized_root = os.path.normpath(root)
        try:
            if os.path.commonpath((normalized, normalized_root)) != normalized_root:
                continue
        except ValueError:
            continue
        root_fd = None
        try:
            root_fd = _open_root(normalized_root)
            return (
                _open_relative(
                    root_fd,
                    os.path.relpath(normalized, normalized_root),
                    flags,
                ),
                normalized,
            )
        except OSError:
            continue
        finally:
            if root_fd is not None:
                os.close(root_fd)
    raise FilesystemApiError("access_denied")


def _open_root(root: str) -> int:
    """Open every component of a configured root without following symlinks."""
    parts = [part for part in root.split(os.sep) if part]
    fd = os.open(os.sep, _O_SEARCH | _O_DIRECTORY)
    try:
        for part in parts:
            new_fd = os.open(part, _O_SEARCH | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_relative(root_fd: int, relative: str, flags: int) -> int:
    """Open a relative path without following any of its components."""
    parts = [] if relative == "." else relative.split(os.sep)
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            new_fd = os.open(part, _O_SEARCH | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = new_fd
        name = parts[-1] if parts else "."
        new_fd = os.open(name, flags | _O_NOFOLLOW, dir_fd=fd)
        try:
            if not _is_beneath(fd, root_fd):
                raise FilesystemApiError("access_denied")
            opened_stat = os.fstat(new_fd)
            entry_stat = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                entry_stat.st_dev,
                entry_stat.st_ino,
            ):
                raise FilesystemApiError("access_denied")
        except BaseException:
            os.close(new_fd)
            raise
        os.close(fd)
        return new_fd
    except BaseException:
        os.close(fd)
        raise


def _is_beneath(directory_fd: int, root_fd: int) -> bool:
    """Return whether an open directory still descends from the pinned root."""
    root_stat = os.fstat(root_fd)
    current_fd = os.dup(directory_fd)
    try:
        while True:
            current_stat = os.fstat(current_fd)
            if (current_stat.st_dev, current_stat.st_ino) == (
                root_stat.st_dev,
                root_stat.st_ino,
            ):
                return True
            parent_fd = os.open(
                "..", _O_SEARCH | _O_NOFOLLOW | _O_DIRECTORY, dir_fd=current_fd
            )
            parent_stat = os.fstat(parent_fd)
            if (parent_stat.st_dev, parent_stat.st_ino) == (
                current_stat.st_dev,
                current_stat.st_ino,
            ):
                os.close(parent_fd)
                return False
            os.close(current_fd)
            current_fd = parent_fd
    finally:
        os.close(current_fd)


def _allowed_dirs() -> list[str]:
    """Parse allowed directories from the environment variable."""
    raw = os.getenv(FILESYSTEM_ALLOWED_DIRS_ENV, "")
    dirs = raw.split(os.pathsep)
    if not dirs or not all(path.strip() and os.path.isabs(path) for path in dirs):
        raise FilesystemApiError("invalid_config")
    normalized = [os.path.normpath(path) for path in dirs]
    if any(os.path.realpath(path) != path for path in normalized):
        raise FilesystemApiError("invalid_config")
    return normalized
