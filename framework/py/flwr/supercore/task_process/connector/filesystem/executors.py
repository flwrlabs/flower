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
import stat

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError
from ..json_utils import require_string


class FilesystemApiError(ConnectorApiError):
    """Secret-safe file system access failure."""

    provider = "Filesystem"


_MAX_DIRECTORY_ENTRIES = 1000
_MAX_FILE_BYTES = 1024 * 1024

# O_NOFOLLOW, O_DIRECTORY, and O_NONBLOCK are POSIX-only; degrade to 0 on
# Windows so the module imports there. On Windows the realpath sandbox in
# _safe_resolve remains the primary control.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NONBLOCK = getattr(os, "O_NONBLOCK", 0)


def list_directory(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List entries in an allowed directory.

    On POSIX the directory is opened via a per-component O_NOFOLLOW walk
    (_open_sandboxed) and the descriptor is kept open through the entry
    metadata loop so stat lookups are fd-relative and cannot escape the
    sandbox. On Windows, where O_NOFOLLOW is unavailable, listdir and
    stat fall back to path-based forms and _safe_resolve remains the
    sandbox control.
    """
    path = require_string(arguments.get("path"), "Filesystem", "path")
    allowed = _allowed_dirs(context)
    resolved = _safe_resolve(path, allowed)
    fd = None
    try:
        if _O_NOFOLLOW:
            fd = _open_sandboxed(resolved, os.O_RDONLY | _O_DIRECTORY | _O_NONBLOCK)
            entries = os.listdir(fd)
        else:
            entries = os.listdir(resolved)
    except OSError:
        raise FilesystemApiError("access_denied") from None
    try:
        if len(entries) > _MAX_DIRECTORY_ENTRIES:
            raise FilesystemApiError("too_many_entries")
        items: list[JSONObject] = []
        for name in sorted(entries):
            try:
                entry_st = (
                    os.stat(name, follow_symlinks=False, dir_fd=fd)
                    if fd is not None
                    else os.stat(os.path.join(resolved, name), follow_symlinks=False)
                )
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
        if fd is not None:
            os.close(fd)


def read_file(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Read one UTF-8 text file inside an allowed directory.

    On POSIX the file is opened via a per-component O_NOFOLLOW walk so
    no intermediate directory can be swapped for a symlink between
    _safe_resolve and the open syscall; O_NONBLOCK avoids hanging on
    named pipes. fstat on the returned fd validates the file is a
    regular file (S_ISREG). On Windows, where O_NOFOLLOW is unavailable,
    open falls back to plain os.open and _safe_resolve remains the
    sandbox control.
    """
    path = require_string(arguments.get("path"), "Filesystem", "path")
    allowed = _allowed_dirs(context)
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


EXECUTORS: dict[str, ConnectorExecutor] = {
    "list_directory": list_directory,
    "read_file": read_file,
}


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
    rejected by the kernel instead of followed. Without O_NOFOLLOW
    (Windows), falls back to a plain open; _safe_resolve remains the
    sandbox control there.
    """
    if not _O_NOFOLLOW:
        return os.open(resolved, flags)
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
    the path is used. On POSIX, _open_sandboxed then re-opens the result
    component by component with O_NOFOLLOW, closing the race window for
    every path level. On Windows, where O_NOFOLLOW is unavailable, this
    realpath check remains the sole sandbox control.
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


def _allowed_dirs(context: ConnectorExecutionContext) -> list[str]:
    """Extract the list of allowed directories from connector config."""
    value = context.config.get("allowed_dirs")
    if not isinstance(value, list) or not all(
        isinstance(d, str) and d.strip() for d in value
    ):
        raise FilesystemApiError("invalid_config")
    return value
