# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Test for Flower command line interface `app publish` command."""


from contextlib import ExitStack
from pathlib import Path

import click
import pytest

from flwr.supercore.constant import (
    MAX_DIR_DEPTH,
    MAX_FILE_BYTES,
    MAX_FILE_COUNT,
    MAX_TOTAL_BYTES,
)

from .publish import (
    _build_multipart_files_param,
    _collect_file_paths,
    _detect_mime,
    _validate_app_name,
    _validate_description,
    _validate_files,
)

TEXT_EXT = ".py"
ALT_TEXT_EXT = ".md"


def write(tmp: Path, file_name: str, data: bytes) -> Path:
    """Write data to given path."""
    path = tmp / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_detect_mime_has_string(tmp_path: Path) -> None:
    """Test the MIME detection."""
    f = write(tmp_path, f"a{TEXT_EXT}", b"print('x')")
    mime = _detect_mime(f)
    assert isinstance(mime, str) and len(mime) > 0


def test_collect_files_depth_limit(tmp_path: Path) -> None:
    """Test collect files depth limit."""
    # Create a path deeper than allowed
    parts = [f"d{i}" for i in range(MAX_DIR_DEPTH + 1)]
    deep = Path(*parts) / f"too_deep{TEXT_EXT}"
    write(tmp_path, deep.as_posix(), b"x")

    with pytest.raises(click.ClickException):
        _collect_file_paths(tmp_path)


def test_collect_files_count_limit(tmp_path: Path) -> None:
    """Test collect files count limit."""
    # Create (max_count + 1) tiny files
    for i in range(MAX_FILE_COUNT + 1):
        write(tmp_path, f"f{i}{TEXT_EXT}", b"x")

    with pytest.raises(click.ClickException):
        _validate_files(_collect_file_paths(tmp_path))


def test_collect_files_total_bytes_limit(tmp_path: Path) -> None:
    """Test collect files total bytes limit."""
    # One file exceeding the total limit by 1 byte
    data = b"x" * (MAX_TOTAL_BYTES + 1)
    write(tmp_path, f"big{TEXT_EXT}", data)

    with pytest.raises(click.ClickException):
        _validate_files(_collect_file_paths(tmp_path))


def test_collect_files_per_file_size_limit(tmp_path: Path) -> None:
    """Test collect files per file size limit."""
    data = b"x" * (MAX_FILE_BYTES + 1)
    write(tmp_path, f"too_big{ALT_TEXT_EXT}", data)

    with pytest.raises(click.ClickException):
        _validate_files(_collect_file_paths(tmp_path))


def test_collect_files_non_utf8_raises_for_text(tmp_path: Path) -> None:
    """Test collect files UTF8 format."""
    # Invalid UTF-8 payload in a text extension
    write(tmp_path, f"bad{TEXT_EXT}", b"\xff\xfe\xfa")

    with pytest.raises(click.ClickException):
        _validate_files(_collect_file_paths(tmp_path))


def test_collect_files_includes_additional_allowed_publish_files(
    tmp_path: Path,
) -> None:
    """Test publish includes newly allowed metadata/config file types."""
    write(tmp_path, ".gitignore", b"# keep\n")
    write(tmp_path, ".editorconfig", b"root = true\n")
    write(tmp_path, "cfg/config.yaml", b"a: 1\n")
    write(tmp_path, "cfg/config.yml", b"b: 2\n")
    write(tmp_path, "cfg/config.json", b'{"c": 3}\n')
    write(tmp_path, "cfg/config.jsonl", b'{"d": 4}\n')
    write(tmp_path, "cfg/.editorconfig", b"indent_style = space\n")
    write(tmp_path, "cfg/extra.editorconfig", b"not included\n")
    write(tmp_path, "skip.txt", b"skip\n")

    paths = _collect_file_paths(tmp_path)
    rel_paths = {path.relative_to(tmp_path).as_posix() for path in paths}

    assert ".gitignore" in rel_paths
    assert ".editorconfig" in rel_paths
    assert "cfg/config.yaml" in rel_paths
    assert "cfg/config.yml" in rel_paths
    assert "cfg/config.json" in rel_paths
    assert "cfg/config.jsonl" in rel_paths
    assert "cfg/.editorconfig" in rel_paths
    assert "cfg/extra.editorconfig" not in rel_paths
    assert "skip.txt" not in rel_paths


def test_collect_files_includes_root_license_without_pyproject_license_file(
    tmp_path: Path,
) -> None:
    """Test root LICENSE is accepted even without [project].license.file."""
    license_file = write(tmp_path, "LICENSE", b"Apache-2.0")
    write(tmp_path, f"ok{TEXT_EXT}", b"print('ok')")

    paths = _collect_file_paths(tmp_path)

    assert license_file in paths


def test_collect_files_rejects_invalid_license_path_in_pyproject(
    tmp_path: Path,
) -> None:
    """Test [project].license.file only allows LICENSE or LICENSE.md."""
    write(
        tmp_path,
        "pyproject.toml",
        (
            b'[project]\nname = "app"\nversion = "1.0.0"\n'
            b'license = { file = "legal/LICENSE.txt" }\n'
        ),
    )
    write(tmp_path, f"ok{TEXT_EXT}", b"print('ok')")

    with pytest.raises(click.ClickException, match="only `LICENSE` or `LICENSE.md`"):
        _collect_file_paths(tmp_path)


def test_collect_files_rejects_missing_license_file_in_pyproject(
    tmp_path: Path,
) -> None:
    """Test declared [project].license.file must exist."""
    write(
        tmp_path,
        "pyproject.toml",
        b'[project]\nname = "app"\nversion = "1.0.0"\nlicense = { file = "LICENSE" }\n',
    )
    write(tmp_path, f"ok{TEXT_EXT}", b"print('ok')")

    with pytest.raises(click.ClickException, match="was declared but does not exist"):
        _collect_file_paths(tmp_path)


def test_collect_files_rejects_license_file_excluded_by_gitignore(
    tmp_path: Path,
) -> None:
    """Test declared [project].license.file cannot be excluded by .gitignore."""
    write(
        tmp_path,
        "pyproject.toml",
        b'[project]\nname = "app"\nversion = "1.0.0"\nlicense = { file = "LICENSE" }\n',
    )
    write(tmp_path, ".gitignore", b"LICENSE\n")
    write(tmp_path, "LICENSE", b"Apache-2.0")
    write(tmp_path, f"ok{TEXT_EXT}", b"print('ok')")

    with pytest.raises(click.ClickException, match="excluded by `.gitignore`"):
        _collect_file_paths(tmp_path)


def test_collect_files_rejects_license_file_with_text_in_pyproject(
    tmp_path: Path,
) -> None:
    """Test [project].license.file and [project].license.text are exclusive."""
    write(
        tmp_path,
        "pyproject.toml",
        (
            b'[project]\nname = "app"\nversion = "1.0.0"\n'
            b'license = { file = "LICENSE", text = "Apache-2.0" }\n'
        ),
    )
    write(tmp_path, "LICENSE", b"Apache-2.0")
    write(tmp_path, f"ok{TEXT_EXT}", b"print('ok')")

    with pytest.raises(
        click.ClickException, match="`file` and `text` cannot be set together"
    ):
        _collect_file_paths(tmp_path)


def test_collect_files_accepts_license_text_in_pyproject(tmp_path: Path) -> None:
    """Test [project].license.text is accepted."""
    pyproject = write(
        tmp_path,
        "pyproject.toml",
        (
            b'[project]\nname = "app"\nversion = "1.0.0"\n'
            b'license = { text = "Apache-2.0" }\n'
        ),
    )
    source = write(tmp_path, f"ok{TEXT_EXT}", b"print('ok')")

    paths = _collect_file_paths(tmp_path)

    assert pyproject in paths
    assert source in paths


def test_build_multipart_files_param(tmp_path: Path) -> None:
    """Test multipart files building."""
    f1 = write(tmp_path, f"a{TEXT_EXT}", b"hello")

    with ExitStack() as stack:
        parts = _build_multipart_files_param(tmp_path, [f1], stack)
        assert len(parts) == 1
        key, (fname, fobj, mime) = parts[0]
        assert key == "files"
        assert fname == f"a{TEXT_EXT}"
        assert hasattr(fobj, "read")
        assert isinstance(mime, str) and mime

    # ExitStack closes the opened file object
    with pytest.raises(ValueError):
        fobj.read(1)  # closed file


def test_validate_description_empty_raises() -> None:
    """Test empty description raises ClickException."""
    with pytest.raises(click.ClickException, match="can't be empty"):
        _validate_description("")


def test_validate_description_valid() -> None:
    """Test valid description passes validation."""
    _validate_description("A simple Flower federated learning app.")


@pytest.mark.parametrize("value", ["app-numpy33", "App-NumPy33"])
def test_validate_app_name_accepts_valid_values(value: str) -> None:
    """Test valid app names pass validation."""
    _validate_app_name(value, "Flower App name")


@pytest.mark.parametrize("value", ["app_numpy33", "33app", "app.numpy33"])
def test_validate_app_name_rejects_invalid_values(value: str) -> None:
    """Test invalid app names are rejected."""
    with pytest.raises(click.ClickException, match="valid app name must start"):
        _validate_app_name(value, "Flower App name")


def test_validate_description_long_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test long description continues when user confirms."""
    monkeypatch.setattr("typer.confirm", lambda *args, **kwargs: True)

    long_desc = "x" * 201
    _validate_description(long_desc)


def test_validate_description_long_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test long description cancels when user declines."""
    monkeypatch.setattr("typer.confirm", lambda *args, **kwargs: False)

    long_desc = "x" * 201
    with pytest.raises(click.ClickException, match="cancelled"):
        _validate_description(long_desc)


@pytest.mark.parametrize("value", [None, 123, 3.14, [], {}])
def test_validate_description_non_string_raises(value: object) -> None:
    """Test non-string descriptions raise ClickException."""
    with pytest.raises(click.ClickException):
        _validate_description(value)
