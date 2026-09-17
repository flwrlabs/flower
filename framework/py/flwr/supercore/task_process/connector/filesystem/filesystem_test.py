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
from unittest.mock import Mock

import pytest

from .executors import FilesystemApiError, list_directory, read_file


def _context(**kwargs: object) -> Mock:
    config: dict[str, object] = {"allowed_dirs": []} | kwargs
    return Mock(config=config)


def test_list_directory_in_temp_dir() -> None:
    """Listing should return sorted file and directory entries."""
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "subdir"))
        with open(os.path.join(root, "a.txt"), "w", encoding="utf-8") as handle:
            handle.write("hello")
        with open(os.path.join(root, "b.txt"), "w", encoding="utf-8") as handle:
            handle.write("world")
        result = list_directory({"path": root}, _context(allowed_dirs=[root]))
        assert result == {
            "entries": [
                {"name": "a.txt", "type": "file"},
                {"name": "b.txt", "type": "file"},
                {"name": "subdir", "type": "directory"},
            ]
        }


def test_read_file_reads_content() -> None:
    """File reading should return UTF-8 content and resolved path."""
    with tempfile.TemporaryDirectory() as root:
        filepath = os.path.join(root, "note.txt")
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("hello, world")
        result = read_file({"path": filepath}, _context(allowed_dirs=[root]))
        assert result["content"] == "hello, world"
        assert result["path"] == os.path.realpath(filepath)


def test_read_file_symlink_outside_denied() -> None:
    """Symlinks that resolve outside allowed dirs should be denied."""
    with tempfile.TemporaryDirectory() as good:
        with tempfile.TemporaryDirectory() as outside:
            sym = os.path.join(good, "link")
            os.symlink(os.path.join(outside, "secret"), sym)
            with pytest.raises(FilesystemApiError, match="access_denied"):
                read_file({"path": sym}, _context(allowed_dirs=[good]))


def test_read_file_path_traversal_denied() -> None:
    """A path that resolves outside allowed dirs should be denied."""
    with tempfile.TemporaryDirectory() as root:
        with tempfile.TemporaryDirectory() as outside:
            outside_file = os.path.join(outside, "secret.txt")
            with open(outside_file, "w", encoding="utf-8") as handle:
                handle.write("secret")
            path = os.path.join(root, "..", os.path.basename(outside), "secret.txt")
            with pytest.raises(FilesystemApiError, match="access_denied"):
                read_file({"path": path}, _context(allowed_dirs=[root]))


def test_read_file_rejects_directory() -> None:
    """Calling read_file on a directory should raise."""
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(FilesystemApiError, match="not_a_file"):
            read_file({"path": root}, _context(allowed_dirs=[root]))


def test_list_directory_rejects_file() -> None:
    """Calling list_directory on a file should raise."""
    with tempfile.TemporaryDirectory() as root:
        filepath = os.path.join(root, "f.txt")
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("x")
        with pytest.raises(FilesystemApiError, match="access_denied"):
            list_directory({"path": filepath}, _context(allowed_dirs=[root]))


def test_list_directory_requires_path() -> None:
    """Missing path argument should raise a ValueError."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        list_directory({}, _context(allowed_dirs=["/tmp"]))


def test_read_file_requires_path() -> None:
    """Missing path argument should raise a ValueError."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        read_file({}, _context(allowed_dirs=["/tmp"]))


def test_relative_path_rejected() -> None:
    """A relative path should be rejected before realpath resolution."""
    with tempfile.TemporaryDirectory() as root:
        with pytest.raises(FilesystemApiError, match="access_denied"):
            read_file({"path": "some/relative/path"}, _context(allowed_dirs=[root]))


def test_allowed_dirs_rejects_empty_string() -> None:
    """Empty-string allowed_dirs entries should trigger invalid_config."""
    with pytest.raises(FilesystemApiError, match="invalid_config"):
        list_directory({"path": "/tmp/x"}, _context(allowed_dirs=[""]))


def test_read_file_enforces_max_size() -> None:
    """Files larger than 1 MB should be rejected."""
    with tempfile.TemporaryDirectory() as root:
        big = os.path.join(root, "big.bin")
        with open(big, "wb") as handle:
            handle.write(b"\x00" * (1024 * 1024 + 1))
        with pytest.raises(FilesystemApiError, match="file_too_large"):
            read_file({"path": big}, _context(allowed_dirs=[root]))
