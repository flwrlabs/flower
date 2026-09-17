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


def list_directory(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List entries in an allowed directory.

    Opens the directory atomically with O_NOFOLLOW | O_DIRECTORY so the
    kernel rejects symlinks and non-directories in a single syscall,
    closing one TOCTOU window after _safe_resolve. Entry types are
    determined via os.stat(..., follow_symlinks=False) to avoid leaking
    information about targets outside the allowed tree.
    """
    path = require_string(arguments.get("path"), "Filesystem", "path")
    allowed = _allowed_dirs(context)
    resolved = _safe_resolve(path, allowed)
    try:
        fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY)
        raw = sorted(os.listdir(fd))
    except OSError:
        raise FilesystemApiError("access_denied") from None
    finally:
        if "fd" in locals():
            os.close(fd)
    if len(raw) > _MAX_DIRECTORY_ENTRIES:
        raise FilesystemApiError("too_many_entries")
    items: list[JSONObject] = []
    for name in raw:
        entry_path = os.path.join(resolved, name)
        try:
            entry_st = os.stat(entry_path, follow_symlinks=False)
        except OSError:
            continue
        items.append(
            {
                "name": name,
                "type": "directory" if stat.S_ISDIR(entry_st.st_mode) else "file",
            }
        )
    return {"entries": items}


def read_file(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Read one UTF-8 text file inside an allowed directory.

    Opens the file atomically with O_NOFOLLOW so the kernel rejects
    symlinks in a single syscall. fstat on the returned fd validates
    the file is a regular file (S_ISREG) and checks the size against
    _MAX_FILE_BYTES before reading. Reading and UTF-8 decoding are
    done from the fd rather than re-resolving the path string.
    """
    path = require_string(arguments.get("path"), "Filesystem", "path")
    allowed = _allowed_dirs(context)
    resolved = _safe_resolve(path, allowed)
    try:
        fd = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise FilesystemApiError("not_a_file")
        if st.st_size > _MAX_FILE_BYTES:
            raise FilesystemApiError("file_too_large")
        raw = os.read(fd, _MAX_FILE_BYTES)
    except OSError:
        raise FilesystemApiError("access_denied") from None
    finally:
        if "fd" in locals():
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


def _safe_resolve(path: str, allowed: list[str]) -> str:
    """Resolve and sandbox an absolute path.

    realpath canonicalization catches symlinks and '..' components before
    the path is used. The per-executor O_NOFOLLOW open then hardens against
    a concurrent attacker who swaps the final path component between this
    check and the open syscall. Intermediate directory components are not
    re-verified after realpath; a full openat(O_NOFOLLOW) walk per level
    would close that residual window if the threat model demands it.
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
    return [d.strip() for d in value]
