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
from pathlib import Path

import pytest

from flwr.supercore.typing import JSONObject

from .filesystem import (
    FILESYSTEM_ALLOWED_DIRS_ENV,
    FILESYSTEM_LIST_DIRECTORY_TOOL_NAME,
    FILESYSTEM_READ_FILE_TOOL_NAME,
    FilesystemApiError,
    invoke_filesystem,
    make_filesystem_tools,
)


def _allow(monkeypatch: pytest.MonkeyPatch, *dirs: Path) -> None:
    monkeypatch.setenv(
        FILESYSTEM_ALLOWED_DIRS_ENV,
        os.pathsep.join(str(path.resolve()) for path in dirs),
    )


def _list(path: Path | str) -> JSONObject:
    return invoke_filesystem(
        FILESYSTEM_LIST_DIRECTORY_TOOL_NAME,
        {"path": str(path)},
    )


def _read(path: Path | str) -> JSONObject:
    return invoke_filesystem(
        FILESYSTEM_READ_FILE_TOOL_NAME,
        {"path": str(path)},
    )


def test_tool_schema_lists_allowed_directories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model should know which absolute paths it can request."""
    _allow(monkeypatch, tmp_path)

    tools = make_filesystem_tools()
    assert [tool["name"] for tool in tools] == [
        FILESYSTEM_LIST_DIRECTORY_TOOL_NAME,
        FILESYSTEM_READ_FILE_TOOL_NAME,
    ]
    assert str(tmp_path.resolve()) in repr(tools)


def test_tool_schema_is_empty_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unconfigured optional connector should advertise no tools."""
    monkeypatch.delenv(FILESYSTEM_ALLOWED_DIRS_ENV, raising=False)

    assert not make_filesystem_tools()


def test_reads_file_and_lists_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Core filesystem operations should return file content and sorted entries."""
    (tmp_path / "subdir").mkdir()
    file = tmp_path / "note.txt"
    file.write_text("hello", encoding="utf-8")
    _allow(monkeypatch, tmp_path)

    assert _read(file) == {
        "content": "hello",
        "path": str(file.resolve()),
    }
    assert _list(tmp_path) == {
        "entries": [
            {"name": "note.txt", "type": "file"},
            {"name": "subdir", "type": "directory"},
        ]
    }


def test_denies_symlink_outside_allowed_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A symlink must not provide access outside the allowed directory."""
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = allowed / "link.txt"
    link.symlink_to(secret)
    _allow(monkeypatch, allowed)

    with pytest.raises(FilesystemApiError, match="access_denied"):
        _read(link)


def test_denies_path_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Parent components must not escape the allowed directory."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    _allow(monkeypatch, allowed)

    with pytest.raises(FilesystemApiError, match="access_denied"):
        _read(allowed / ".." / secret.name)


def test_rejects_wrong_target_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read-file accepts files and list-directory accepts directories."""
    file = tmp_path / "note.txt"
    file.write_text("hello", encoding="utf-8")
    _allow(monkeypatch, tmp_path)

    with pytest.raises(FilesystemApiError, match="not_a_file"):
        _read(tmp_path)
    with pytest.raises(FilesystemApiError, match="access_denied"):
        _list(file)


def test_requires_absolute_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sandbox must have an absolute configured root."""
    monkeypatch.setenv(FILESYSTEM_ALLOWED_DIRS_ENV, "relative")

    with pytest.raises(FilesystemApiError, match="invalid_config"):
        _list(tmp_path)


def test_enforces_file_size_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files larger than one MiB should be rejected."""
    file = tmp_path / "large.txt"
    file.write_bytes(b"x" * (1024 * 1024 + 1))
    _allow(monkeypatch, tmp_path)

    with pytest.raises(FilesystemApiError, match="file_too_large"):
        _read(file)
