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
from unittest.mock import Mock

import pytest

from flwr.proto.task_pb2 import TaskUsage  # pylint: disable=E0611
from flwr.supercore.typing import JSONObject

from ..registry import invoke_connector
from .filesystem import (
    FILESYSTEM_ALLOWED_DIRS_ENV,
    FILESYSTEM_LIST_DIRECTORY_TOOL_NAME,
    FILESYSTEM_READ_FILE_TOOL_NAME,
    invoke_filesystem,
    list_directory,
    make_filesystem_tools,
    read_file,
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


def test_handlers_record_usage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Filesystem handlers should record their tool usage."""
    file = tmp_path / "note.txt"
    file.write_text("hello", encoding="utf-8")
    _allow(monkeypatch, tmp_path)
    usage_recorder = Mock()

    list_directory(str(tmp_path), usage_recorder=usage_recorder)
    usage_recorder.record.assert_called_once_with(
        TaskUsage(usage_type="filesystem_list_directory")
    )

    usage_recorder.reset_mock()
    read_file(str(file), usage_recorder=usage_recorder)
    usage_recorder.record.assert_called_once_with(
        TaskUsage(usage_type="filesystem_read_file")
    )


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (FILESYSTEM_LIST_DIRECTORY_TOOL_NAME, {}),
        (FILESYSTEM_READ_FILE_TOOL_NAME, {}),
        (FILESYSTEM_LIST_DIRECTORY_TOOL_NAME, {"path": "/allowed", "extra": True}),
        (FILESYSTEM_READ_FILE_TOOL_NAME, {"path": "/allowed", "extra": True}),
    ],
)
def test_handler_invalid_arguments_return_structured_error(
    name: str, arguments: JSONObject
) -> None:
    """Malformed arguments should remain a model-facing validation error."""
    assert invoke_connector(name, arguments, Mock()) == {
        "error": {"code": "invalid_request"}
    }


def test_read_missing_file_returns_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing file should be reported to the model without raising."""
    _allow(monkeypatch, tmp_path)

    assert _read(tmp_path / "missing.txt") == {
        "error": {"code": "not_found", "message": "Path not found."}
    }


def test_rejects_non_object_arguments() -> None:
    """Tool arguments must be a JSON object."""
    assert invoke_filesystem(FILESYSTEM_READ_FILE_TOOL_NAME, None) == {
        "error": {"code": "invalid_request"}
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

    assert _read(link) == {"error": {"code": "access_denied"}}


def test_denies_path_traversal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Parent components must not escape the allowed directory."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    _allow(monkeypatch, allowed)

    assert _read(allowed / ".." / secret.name) == {"error": {"code": "access_denied"}}


def test_rejects_wrong_target_types(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read-file accepts files and list-directory accepts directories."""
    file = tmp_path / "note.txt"
    file.write_text("hello", encoding="utf-8")
    _allow(monkeypatch, tmp_path)

    assert _read(tmp_path) == {"error": {"code": "not_a_file"}}
    assert _list(file) == {"error": {"code": "access_denied"}}


def test_requires_absolute_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sandbox must have an absolute configured root."""
    monkeypatch.setenv(FILESYSTEM_ALLOWED_DIRS_ENV, "relative")

    assert _list(tmp_path) == {"error": {"code": "invalid_config"}}


def test_enforces_file_size_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Files larger than one MiB should be rejected."""
    file = tmp_path / "large.txt"
    file.write_bytes(b"x" * (1024 * 1024 + 1))
    _allow(monkeypatch, tmp_path)

    assert _read(file) == {"error": {"code": "file_too_large"}}
