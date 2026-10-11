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
"""Tests for Flower command line interface `install` command."""


import hashlib
import io
import zipfile
from pathlib import Path

import click
import pytest

from .archive_utils import safe_extract_zip
from .install import _verify_hashes, install_from_fab


def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    """Create ZIP bytes from (path, content) entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries:
            zf.writestr(name, content)
    return buf.getvalue()


def test_safe_extract_zip_extracts_regular_files(tmp_path: Path) -> None:
    """Safe extraction should succeed for regular archive entries."""
    zip_bytes = _zip_bytes([("dir/file.txt", b"hello")])

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        safe_extract_zip(zf, tmp_path)

    assert (tmp_path / "dir" / "file.txt").read_bytes() == b"hello"


def test_safe_extract_zip_rejects_parent_traversal(tmp_path: Path) -> None:
    """Safe extraction should reject path traversal via '..' entries."""
    zip_bytes = _zip_bytes([("../evil.txt", b"x")])

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        with pytest.raises(click.ClickException, match="Unsafe path in FAB archive"):
            safe_extract_zip(zf, tmp_path)

    assert not (tmp_path.parent / "evil.txt").exists()


def test_safe_extract_zip_rejects_absolute_paths(tmp_path: Path) -> None:
    """Safe extraction should reject absolute archive paths."""
    zip_bytes = _zip_bytes([("/tmp/evil.txt", b"x")])

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        with pytest.raises(click.ClickException, match="Unsafe path in FAB archive"):
            safe_extract_zip(zf, tmp_path)


def test_install_from_fab_rejects_zip_slip(tmp_path: Path) -> None:
    """install_from_fab should fail fast on zip-slip entries."""
    fab_bytes = _zip_bytes(
        [
            ("../evil.txt", b"x"),
            (".info/CONTENT", b""),
        ]
    )

    with pytest.raises(click.ClickException, match="Unsafe path in FAB archive"):
        _ = install_from_fab(fab_bytes, install_dir=tmp_path, skip_prompt=True)


def _manifest_line(path: str, content: bytes) -> str:
    """Build a CONTENT manifest line: "path,sha256,size_bits"."""
    return f"{path},{hashlib.sha256(content).hexdigest()},{len(content) * 8}"


def test_verify_hashes_accepts_commas_in_file_names(tmp_path: Path) -> None:
    """Hash verification should handle file names containing commas."""
    name = "data,v2.json"
    content = b'{"round": 1}'
    (tmp_path / name).write_bytes(content)

    assert _verify_hashes(_manifest_line(name, content), tmp_path)


def test_verify_hashes_rejects_modified_content(tmp_path: Path) -> None:
    """A hash mismatch should be detected even when the name has a comma."""
    name = "data,v2.json"
    (tmp_path / name).write_bytes(b"tampered")

    assert not _verify_hashes(_manifest_line(name, b'{"round": 1}'), tmp_path)


def test_verify_hashes_rejects_missing_file(tmp_path: Path) -> None:
    """A manifest entry without a matching file should not verify."""
    assert not _verify_hashes(_manifest_line("absent,v2.json", b"x"), tmp_path)
