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
"""Test for Flower command line interface utils."""


import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import click
import grpc
import pytest
from parameterized import parameterized

from flwr.cli.constant import (
    LOCAL_CONTROL_API_ADDRESS,
    LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
)
from flwr.cli.typing import SuperLinkConnection, SuperLinkSimulationOptions
from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FLWR_DIR,
    REFRESH_TOKEN_KEY,
)
from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH
from flwr.supercore.constant import MAX_DIR_DEPTH, MAX_NAME_LENGTH
from flwr.supercore.interceptors import RuntimeVersionClientInterceptor

from .utils import (
    _format_grpc_error,
    build_pathspec,
    cli_output_handler,
    collect_files,
    depth_of,
    filter_paths_for_publish,
    flwr_cli_grpc_exc_handler,
    get_executed_command,
    get_sha256_hash,
    init_channel_from_connection,
    load_gitignore_patterns,
    validate_credentials_content,
    validate_federation_name,
)


class _GrpcErrorWithDetails:
    """Test helper object carrying a gRPC-like details string."""

    def __init__(self, details: str) -> None:
        self._details = details

    def details(self) -> str:
        """Return the stored gRPC details string."""
        return self._details


def _grpc_error_with_details(details: str) -> grpc.RpcError:
    """Return a grpc.RpcError-compatible test helper with a details method."""
    return cast(grpc.RpcError, _GrpcErrorWithDetails(details))


class TestGetSHA256Hash(unittest.TestCase):
    """Unit tests for `get_sha256_hash` function."""

    def test_hash_with_integer(self) -> None:
        """Test the SHA-256 hash calculation when input is an integer."""
        # Prepare
        test_int = 13413
        expected_hash = hashlib.sha256(str(test_int).encode()).hexdigest()

        # Execute
        result = get_sha256_hash(test_int)

        # Assert
        self.assertEqual(result, expected_hash)

    def test_hash_with_file(self) -> None:
        """Test the SHA-256 hash calculation when input is a file path."""
        # Prepare - Create a temporary file with known content
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content for SHA-256 hashing.")
            temp_file_path = Path(temp_file.name)

        try:
            # Execute
            sha256 = hashlib.sha256()
            with open(temp_file_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha256.update(data)
            expected_hash = sha256.hexdigest()

            result = get_sha256_hash(temp_file_path)

            # Assert
            self.assertEqual(result, expected_hash)
        finally:
            # Clean up the temporary file
            os.remove(temp_file_path)

    def test_empty_file(self) -> None:
        """Test the SHA-256 hash calculation for an empty file."""
        # Prepare
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)

        try:
            # Execute
            expected_hash = hashlib.sha256(b"").hexdigest()
            result = get_sha256_hash(temp_file_path)

            # Assert
            self.assertEqual(result, expected_hash)
        finally:
            os.remove(temp_file_path)

    def test_large_file(self) -> None:
        """Test the SHA-256 hash calculation for a large file."""
        # Prepare - Generate large data (e.g., 10 MB)
        large_data = b"a" * (10 * 1024 * 1024)  # 10 MB of 'a's
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(large_data)
            temp_file_path = Path(temp_file.name)

        try:
            expected_hash = hashlib.sha256(large_data).hexdigest()
            # Execute
            result = get_sha256_hash(temp_file_path)

            # Assert
            self.assertEqual(result, expected_hash)
        finally:
            os.remove(temp_file_path)

    def test_nonexistent_file(self) -> None:
        """Test the SHA-256 hash calculation when the file does not exist."""
        # Prepare
        nonexistent_path = Path("/path/to/nonexistent/file.txt")

        # Execute & assert
        with self.assertRaises(FileNotFoundError):
            get_sha256_hash(nonexistent_path)


def test_validate_credentials_content_success(tmp_path: Path) -> None:
    """Test the credentials content loading."""
    creds = {
        AUTHN_TYPE_JSON_KEY: "userpass",
        ACCESS_TOKEN_KEY: "abc",
        REFRESH_TOKEN_KEY: "def",
    }
    path = tmp_path / "creds.json"
    path.write_text(json.dumps(creds), encoding="utf-8")
    token = validate_credentials_content(path)
    assert token == "abc"


def test_load_gitignore_patterns(tmp_path: Path) -> None:
    """Test gitignore pattern loading."""
    path = tmp_path / ".gitignore"
    path.write_text("*.log\nsecret/\n# comment\n\n", encoding="utf-8")
    patterns_from_path = load_gitignore_patterns(path)
    patterns_from_bytes = load_gitignore_patterns(path.read_bytes())

    assert patterns_from_path == ["*.log", "secret/"]
    assert patterns_from_bytes == ["*.log", "secret/"]


def test_load_gitignore_patterns_with_pathspec() -> None:
    """Test gitignore patterns with pathspec matching."""
    patterns = load_gitignore_patterns(b"*.tmp\n")
    spec = build_pathspec(patterns + [f"{FLWR_DIR}/"])

    # Should match .tmp files
    assert spec.match_file("a.tmp") is True

    # Should match FLWR_DIR
    assert spec.match_file(f"{FLWR_DIR}/creds.json") is True

    # Should not match normal files
    assert spec.match_file("good.py") is False


def test_get_executed_command_single() -> None:
    """Test get_executed_command with a two-word command (e.g., flwr ls)."""
    root_group = click.Group("flwr")
    ls_cmd = click.Command("ls")

    with click.Context(root_group, info_name="flwr") as root_ctx:
        with click.Context(ls_cmd, parent=root_ctx, info_name="ls"):
            assert get_executed_command() == "flwr ls"


def test_get_executed_command_nested() -> None:
    """Test get_executed_command with nested commands (e.g., flwr federation list)."""
    # Create parent group "flwr" with child group "federation" and command "list"
    root_group = click.Group("flwr")
    federation_group = click.Group("federation")
    list_cmd = click.Command("list")

    with click.Context(root_group, info_name="flwr") as root_ctx:
        with click.Context(
            federation_group, parent=root_ctx, info_name="federation"
        ) as fed_ctx:
            with click.Context(list_cmd, parent=fed_ctx, info_name="list"):
                assert get_executed_command() == "flwr federation list"


def test_init_channel_from_connection_uses_resolved_connection() -> None:
    """Ensure resolved connection values are used for channel creation."""
    unresolved = SuperLinkConnection(
        name="local",
        address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )
    resolved = SuperLinkConnection(
        name="local",
        address=LOCAL_CONTROL_API_ADDRESS,
        insecure=True,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )
    auth_plugin = Mock()
    auth_plugin.load_tokens = Mock()

    with patch(
        "flwr.cli.utils.ensure_local_superlink", return_value=resolved
    ) as mock_ensure:
        with patch("flwr.cli.utils.load_certificate_in_connection", return_value=None):
            with patch("flwr.cli.utils.create_channel") as mock_create:
                channel = Mock()
                mock_create.return_value = channel

                ret = init_channel_from_connection(unresolved, auth_plugin)

    assert ret is channel
    mock_ensure.assert_called_once_with(unresolved)
    auth_plugin.load_tokens.assert_called_once()

    kwargs = mock_create.call_args.kwargs
    assert kwargs["server_address"] == LOCAL_CONTROL_API_ADDRESS
    assert kwargs["insecure"] is True
    assert kwargs["root_certificates"] is None
    assert kwargs["max_message_length"] == GRPC_MAX_MESSAGE_LENGTH
    assert len(kwargs["interceptors"]) == 2
    assert isinstance(kwargs["interceptors"][0], RuntimeVersionClientInterceptor)
    # pylint: disable-next=protected-access
    assert kwargs["interceptors"][0]._metadata.component_name == "flwr CLI"
    channel.subscribe.assert_called_once()


def test_custom_grpc_err_handler() -> None:
    """Test flwr_cli_grpc_exc_handler with a custom error handler."""

    # Prepare
    class CustomError(Exception):
        """Custom error for testing."""

    mock_handler = Mock(side_effect=CustomError)
    grpc_error = grpc.RpcError()

    # Execute & assert
    with pytest.raises(CustomError):
        with flwr_cli_grpc_exc_handler(mock_handler):
            raise grpc_error

    mock_handler.assert_called_once_with(grpc_error)


def test_format_grpc_error_uses_json_message_field() -> None:
    """Structured Flower errors combine public message and details."""
    err = _grpc_error_with_details(
        '{"public_message": "request failed", '
        '"public_details": "missing entitlement", "code": 400}'
    )

    assert _format_grpc_error(err) == "request failed\nmissing entitlement"


def test_format_grpc_error_falls_back_to_plain_string() -> None:
    """Non-JSON errors fall back to their normal string form."""
    err = _grpc_error_with_details("plain failure")

    assert _format_grpc_error(err) == "plain failure"


def test_cli_output_handler_raises_click_exception_for_json_error() -> None:
    """cli_output_handler preserves the original ClickException text."""
    with pytest.raises(click.ClickException, match="request failed") as exc_info:
        with cli_output_handler():
            raise click.ClickException('{"message": "request failed", "code": 400}')

    assert exc_info.value.message == '{"message": "request failed", "code": 400}'


@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        (Path("a.py"), 0),
        (Path("d1/file.txt"), 1),
        (Path("d1/d2/d3/f.txt"), 3),
        (Path("d1/d2/d3/d4/d5/x"), 5),
    ],
)
def test_depth_of(rel: Path, expected: int) -> None:
    """Test the directory depth detection."""
    assert depth_of(rel) == expected


# === collect_files tests ===


def test_collect_files_empty_dir(tmp_path: Path) -> None:
    """Empty directory returns an empty dict."""
    assert not collect_files(tmp_path)


def test_collect_files_basic(tmp_path: Path) -> None:
    """Files are collected with correct POSIX relative paths and absolute values."""
    # Prepare
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("y", encoding="utf-8")

    # Execute
    result = collect_files(tmp_path)

    # Assert
    assert set(result.keys()) == {"a.py", "b.txt"}
    assert result["a.py"] == tmp_path / "a.py"
    assert result["b.txt"] == tmp_path / "b.txt"


def test_collect_files_ignores_symlinked_files(tmp_path: Path) -> None:
    """Symlinked files are excluded from the collected files."""
    # Prepare
    real = tmp_path / "real.py"
    real.write_text("real", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(real)

    # Execute
    result = collect_files(tmp_path)

    # Assert
    assert "real.py" in result
    assert "link.py" not in result


def test_collect_files_ignores_symlinked_dirs(tmp_path: Path) -> None:
    """Symlinked directories are not traversed."""
    # Prepare
    # Create a real directory with a file outside the root
    external = tmp_path / "external"
    external.mkdir()
    (external / "secret.py").write_text("s", encoding="utf-8")

    # Root with a symlinked directory pointing to external
    root = tmp_path / "root"
    root.mkdir()
    (root / "app.py").write_text("a", encoding="utf-8")
    (root / "linked_dir").symlink_to(external)

    # Execute
    result = collect_files(root)

    # Assert
    assert "app.py" in result
    assert "linked_dir/secret.py" not in result


# === filter_paths_for_publish tests ===


def _to_path_files(files: dict[str, bytes], tmp_path: Path) -> dict[str, Path]:
    """Write bytes to disk and return a mapping of relative paths to Path objects."""
    result: dict[str, Path] = {}
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        result[name] = p
    return result


@pytest.mark.parametrize(
    ("files", "expected_keys"),
    [
        # Included extensions pass through
        ({"app.py": b""}, {"app.py"}),
        ({"pyproject.toml": b""}, {"pyproject.toml"}),
        ({"README.md": b""}, {"README.md"}),
        ({"config.yaml": b""}, {"config.yaml"}),
        ({"config.yml": b""}, {"config.yml"}),
        ({"data.json": b""}, {"data.json"}),
        ({"data.jsonl": b""}, {"data.jsonl"}),
        # Non-matching extensions are excluded
        ({"image.png": b"", "app.py": b""}, {"app.py"}),
        # Nested files work
        ({"src/main.py": b""}, {"src/main.py"}),
    ],
)
@pytest.mark.parametrize("use_paths", [False, True], ids=["bytes", "path"])
def test_filter_paths_for_publish_include(
    files: dict[str, bytes],
    expected_keys: set[str],
    use_paths: bool,
    tmp_path: Path,
) -> None:
    """Files with included extensions are kept; others are dropped."""
    # Prepare
    input_files = cast(
        dict[str, Path | bytes], _to_path_files(files, tmp_path) if use_paths else files
    )
    # Execute & assert
    assert set(filter_paths_for_publish(input_files).keys()) == expected_keys


@pytest.mark.parametrize(
    "excluded_path",
    [
        "__pycache__/mod.py",
        ".flwr/creds.json",
    ],
)
@pytest.mark.parametrize("use_paths", [False, True], ids=["bytes", "path"])
def test_filter_paths_for_publish_excludes(
    excluded_path: str, use_paths: bool, tmp_path: Path
) -> None:
    """__pycache__ and .flwr paths are excluded."""
    # Prepare
    raw: dict[str, bytes] = {excluded_path: b"", "app.py": b""}
    files = cast(
        dict[str, Path | bytes], _to_path_files(raw, tmp_path) if use_paths else raw
    )
    # Execute & assert
    assert set(filter_paths_for_publish(files).keys()) == {"app.py"}


@pytest.mark.parametrize("use_paths", [False, True], ids=["bytes", "path"])
def test_filter_paths_for_publish_respects_gitignore(
    use_paths: bool, tmp_path: Path
) -> None:
    """Patterns in .gitignore cause matching files to be excluded."""
    # Prepare
    raw: dict[str, bytes] = {
        ".gitignore": b"secret.py\n",
        "app.py": b"",
        "secret.py": b"",
    }
    files = cast(
        dict[str, Path | bytes], _to_path_files(raw, tmp_path) if use_paths else raw
    )
    # Execute
    result = filter_paths_for_publish(files)
    # Assert
    assert "secret.py" not in result
    assert "app.py" in result


@pytest.mark.parametrize("use_paths", [False, True], ids=["bytes", "path"])
def test_filter_paths_for_publish_max_depth_exceeded(
    use_paths: bool, tmp_path: Path
) -> None:
    """ValueError is raised when a file exceeds MAX_DIR_DEPTH."""
    # Prepare
    deep = "/".join(["d"] * (MAX_DIR_DEPTH + 1)) + "/f.py"
    raw: dict[str, bytes] = {deep: b""}
    files = cast(
        dict[str, Path | bytes], _to_path_files(raw, tmp_path) if use_paths else raw
    )
    # Execute & assert
    with pytest.raises(ValueError, match="exceeds the maximum directory depth"):
        filter_paths_for_publish(files)


def test_filter_paths_for_publish_empty() -> None:
    """Empty input returns empty output."""
    assert not filter_paths_for_publish({})


@parameterized.expand(  # type: ignore
    [
        ("federation123", True, ""),  # alphanumeric
        ("test-federation", True, ""),  # hyphenated
        ("f" * MAX_NAME_LENGTH, True, ""),  # exactly_max_length
        (
            "f" * (MAX_NAME_LENGTH + 1),
            False,
            f"Must be no longer than {MAX_NAME_LENGTH} characters.",
        ),  # too_long
        ("", False, "Cannot be empty."),  # empty
        ("-federation", False, "Must start with a letter."),  # starts_with_symbol
        (
            "test federation",
            False,
            "Can only contain letters, digits, and hyphens.",
        ),  # contains_space
        ("Testfederation", True, ""),  # uppercase allowed
        (
            "test_federation",
            False,
            "Can only contain letters, digits, and hyphens.",
        ),  # invalid_symbol
        (
            "Test Federation!",
            False,
            "Can only contain letters, digits, and hyphens.",
        ),  # multiple_violations
    ]
)
def test_validate_federation_name(
    name: str, expected_valid: bool, expected_message: str
) -> None:
    """Test federation name validator function."""
    valid, message = validate_federation_name(name)

    assert valid is expected_valid
    assert message == expected_message
