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
"""Tests for the filesystem connector."""

import os
import tempfile
from typing import cast
from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from . import filesystem as filesystem_module
from .filesystem import (
    FILESYSTEM_ALLOWED_DIRS_ENV,
    FilesystemApiError,
    invoke_filesystem_provider,
    make_filesystem_tool,
)


def _allow(monkeypatch: pytest.MonkeyPatch, *dirs: str) -> None:
    monkeypatch.setenv(FILESYSTEM_ALLOWED_DIRS_ENV, os.pathsep.join(dirs))


def _call(action: str, path: str) -> JSONObject:
    return invoke_filesystem_provider(action, path, usage_recorder=Mock())


def test_list_directory_in_temp_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listing should return sorted file and directory entries."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "subdir"))
        with open(os.path.join(root, "a.txt"), "w", encoding="utf-8") as handle:
            handle.write("hello")
        with open(os.path.join(root, "b.txt"), "w", encoding="utf-8") as handle:
            handle.write("world")
        _allow(monkeypatch, root)
        result = _call("list_directory", root)
        assert result == {
            "entries": [
                {"name": "a.txt", "type": "file"},
                {"name": "b.txt", "type": "file"},
                {"name": "subdir", "type": "directory"},
            ]
        }


def test_read_file_reads_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """File reading should return UTF-8 content and resolved path."""
    with tempfile.TemporaryDirectory() as root:
        filepath = os.path.join(root, "note.txt")
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("hello, world")
        _allow(monkeypatch, root)
        result = _call("read_file", filepath)
        assert result["content"] == "hello, world"
        assert result["path"] == os.path.normpath(filepath)


@pytest.mark.skipif(
    not (getattr(os, "O_SEARCH", 0) or getattr(os, "O_PATH", 0)) or os.geteuid() == 0,
    reason="requires O_SEARCH/O_PATH and non-root permission checks",
)
def test_read_file_traverses_search_only_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Files beneath execute-only ancestors should stay readable."""
    with tempfile.TemporaryDirectory() as root:
        middle = os.path.join(root, "search_only")
        os.makedirs(middle)
        filepath = os.path.join(middle, "note.txt")
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("hidden but readable")
        os.chmod(middle, 0o111)
        try:
            _allow(monkeypatch, root)
            result = _call("read_file", filepath)
        finally:
            os.chmod(middle, 0o700)
        assert result["content"] == "hidden but readable"


def test_read_file_symlink_outside_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Symlinks that resolve outside allowed dirs should be denied."""
    with tempfile.TemporaryDirectory() as good:
        with tempfile.TemporaryDirectory() as outside:
            sym = os.path.join(good, "link")
            os.symlink(os.path.join(outside, "secret"), sym)
            _allow(monkeypatch, good)
            with pytest.raises(FilesystemApiError, match="access_denied"):
                _call("read_file", sym)


def test_replaced_allowed_root_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing an allowed root with a symlink must not move the sandbox."""
    with tempfile.TemporaryDirectory() as parent:
        with tempfile.TemporaryDirectory() as outside:
            root = os.path.join(parent, "allowed")
            os.mkdir(root)
            secret = os.path.join(outside, "secret.txt")
            with open(secret, "w", encoding="utf-8") as handle:
                handle.write("secret")
            _allow(monkeypatch, root)
            os.rmdir(root)
            os.symlink(outside, root)

            with pytest.raises(FilesystemApiError, match="access_denied"):
                _call("read_file", os.path.join(root, "secret.txt"))


def test_allowed_root_preserves_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace in an allowed directory name must remain significant."""
    with tempfile.TemporaryDirectory() as parent:
        root = os.path.join(parent, "allowed ")
        os.mkdir(root)
        path = os.path.join(root, "note.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("hello")
        _allow(monkeypatch, root)

        assert _call("read_file", path)["content"] == "hello"


def test_read_file_path_traversal_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path that resolves outside allowed dirs should be denied."""
    with tempfile.TemporaryDirectory() as root:
        with tempfile.TemporaryDirectory() as outside:
            outside_file = os.path.join(outside, "secret.txt")
            with open(outside_file, "w", encoding="utf-8") as handle:
                handle.write("secret")
            path = os.path.join(root, "..", os.path.basename(outside), "secret.txt")
            _allow(monkeypatch, root)
            with pytest.raises(FilesystemApiError, match="access_denied"):
                _call("read_file", path)


def test_read_file_rejects_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling read_file on a directory should raise."""
    with tempfile.TemporaryDirectory() as root:
        _allow(monkeypatch, root)
        with pytest.raises(FilesystemApiError, match="not_a_file"):
            _call("read_file", root)


def test_list_directory_rejects_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling list_directory on a file should raise."""
    with tempfile.TemporaryDirectory() as root:
        filepath = os.path.join(root, "f.txt")
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("x")
        _allow(monkeypatch, root)
        with pytest.raises(FilesystemApiError, match="access_denied"):
            _call("list_directory", filepath)


def test_list_directory_reports_symlink_as_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symlinks should be reported as type 'other'."""
    with tempfile.TemporaryDirectory() as root:
        target = os.path.join(root, "target.txt")
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("x")
        os.symlink(target, os.path.join(root, "link"))
        _allow(monkeypatch, root)
        result = _call("list_directory", root)
        assert result == {
            "entries": [
                {"name": "link", "type": "other"},
                {"name": "target.txt", "type": "file"},
            ]
        }


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX-only")
def test_list_directory_reports_fifo_as_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIFOs should be reported as type 'other'."""
    with tempfile.TemporaryDirectory() as root:
        os.mkfifo(os.path.join(root, "fifo"))
        _allow(monkeypatch, root)
        result = _call("list_directory", root)
        assert result == {"entries": [{"name": "fifo", "type": "other"}]}


def test_list_directory_rejects_too_many_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directories above the entry cap should raise before materializing."""
    with tempfile.TemporaryDirectory() as root:
        for index in range(2):
            os.makedirs(os.path.join(root, f"entry-{index}"))
        _allow(monkeypatch, root)
        monkeypatch.setattr(filesystem_module, "_MAX_DIRECTORY_ENTRIES", 1)
        with pytest.raises(FilesystemApiError, match="too_many_entries"):
            _call("list_directory", root)


def test_windows_rejected_as_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unsupported platforms should be rejected."""
    monkeypatch.setattr(filesystem_module, "_PLATFORM_SUPPORTED", False)
    monkeypatch.setenv(FILESYSTEM_ALLOWED_DIRS_ENV, "/tmp/example")
    with pytest.raises(FilesystemApiError, match="unsupported_platform"):
        _call("list_directory", "/tmp/example")


def test_invalid_action_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown action should raise."""
    with tempfile.TemporaryDirectory() as root:
        _allow(monkeypatch, root)
        with pytest.raises(FilesystemApiError, match="invalid_action"):
            _call("delete_file", root)


def test_relative_path_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative path should be rejected before realpath resolution."""
    with tempfile.TemporaryDirectory() as root:
        _allow(monkeypatch, root)
        with pytest.raises(FilesystemApiError, match="access_denied"):
            _call("read_file", "some/relative/path")


def test_allowed_dirs_rejects_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset allowed_dirs should trigger invalid_config."""
    monkeypatch.delenv(FILESYSTEM_ALLOWED_DIRS_ENV, raising=False)
    with pytest.raises(FilesystemApiError, match="invalid_config"):
        _call("list_directory", "/tmp/x")


def test_read_file_enforces_max_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Files larger than 1 MB should be rejected."""
    with tempfile.TemporaryDirectory() as root:
        big = os.path.join(root, "big.bin")
        with open(big, "wb") as handle:
            handle.write(b"\x00" * (1024 * 1024 + 1))
        _allow(monkeypatch, root)
        with pytest.raises(FilesystemApiError, match="file_too_large"):
            _call("read_file", big)


def test_make_filesystem_tool_schema() -> None:
    """Tool schema should expose the filesystem connector contract."""
    tool = make_filesystem_tool()
    assert tool["name"] == "filesystem"
    assert tool["type"] == "function"
    params = cast(JSONObject, tool["parameters"])
    assert params["additionalProperties"] is False
    assert set(cast(list[str], params["required"])) == {"action", "path"}
    properties = cast(JSONObject, params["properties"])
    assert cast(JSONObject, properties["action"])["enum"] == [
        "list_directory",
        "read_file",
    ]


def test_allowed_dirs_rejects_relative_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allowed directory configuration must contain absolute paths."""
    _allow(monkeypatch, "relative")
    with pytest.raises(FilesystemApiError, match="invalid_config"):
        _call("list_directory", "/tmp/x")
