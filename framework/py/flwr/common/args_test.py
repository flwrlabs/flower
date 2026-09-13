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
"""Tests for runtime dependency installation CLI arguments."""

import argparse
import io
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from flwr.common.args import (
    add_args_flwr_app_common,
    add_args_runtime_dependency_install,
    try_obtain_flwr_app_token,
)
from flwr.common.constant import FLWR_TASK_TOKEN_LENGTH, RUNTIME_DEPENDENCY_INSTALL

_VALID_TASK_TOKEN = "a" * (FLWR_TASK_TOKEN_LENGTH * 2)


def test_runtime_dependency_install_args_defaults() -> None:
    """Verify runtime dependency installation args default values."""
    parser = argparse.ArgumentParser()
    add_args_runtime_dependency_install(parser)

    args = parser.parse_args([])

    assert args.runtime_dependency_install is RUNTIME_DEPENDENCY_INSTALL


def test_runtime_dependency_install_args_flags() -> None:
    """Verify runtime dependency installation args parse correctly."""
    parser = argparse.ArgumentParser()
    add_args_runtime_dependency_install(parser)

    args = parser.parse_args(["--allow-runtime-dependency-installation"])

    assert args.runtime_dependency_install is True


def test_flwr_app_common_args_require_token() -> None:
    """App process CLIs should require a token."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_flwr_app_common_args_parse_token() -> None:
    """App process CLIs should parse token and common flags."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    args = parser.parse_args(
        [
            "--token",
            "test-token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.token == "test-token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_app_common_args_parse_token_file() -> None:
    """App process CLIs should parse token-file and common flags."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    args = parser.parse_args(
        [
            "--token-file",
            "/path/to/token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.token is None
    assert args.token_file == "/path/to/token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_app_common_args_reject_token_and_token_file() -> None:
    """App process CLIs should reject duplicate token sources."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--token", "test-token", "--token-file", "/path/to/token"])


def test_flwr_app_common_args_accepts_only_one_token_source() -> None:
    """The private stdin mode should be opt-in and mutually exclusive."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser, include_token_stdin=True)

    assert parser.parse_args(["--token-stdin"]).token_stdin is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--token", "test-token", "--token-stdin"])


def test_flwr_app_common_args_reject_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--token", "test-token", "--run-once"])


def test_try_obtain_flwr_app_token_returns_token() -> None:
    """Token resolution should return direct token arguments."""
    args = argparse.Namespace(token="test-token")

    assert try_obtain_flwr_app_token(args) == "test-token"


def test_try_obtain_flwr_app_token_reads_token_file(tmp_path: Path) -> None:
    """Token resolution should read token-file contents."""
    token_file = tmp_path / "token"
    token_file.write_text("test-token\n", encoding="utf-8")
    args = argparse.Namespace(token=None, token_file=str(token_file))

    assert try_obtain_flwr_app_token(args) == "test-token"


def test_try_obtain_flwr_app_token_rejects_missing_token_file(
    tmp_path: Path,
) -> None:
    """Token resolution should reject missing token files."""
    args = argparse.Namespace(token=None, token_file=str(tmp_path / "missing-token"))

    with pytest.raises(SystemExit):
        try_obtain_flwr_app_token(args)


def test_try_obtain_flwr_app_token_rejects_empty_token_file(tmp_path: Path) -> None:
    """Token resolution should reject empty token files."""
    token_file = tmp_path / "token"
    token_file.write_text(" \n", encoding="utf-8")
    args = argparse.Namespace(token=None, token_file=str(token_file))

    with pytest.raises(SystemExit):
        try_obtain_flwr_app_token(args)


def test_try_obtain_flwr_app_token_accumulates_split_stdin_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid task token can arrive across multiple standard-input reads."""
    token_stdin = Mock()
    token_stdin.buffer.read1.side_effect = [
        _VALID_TASK_TOKEN[:128].encode("ascii"),
        f"{_VALID_TASK_TOKEN[128:]}\n".encode("ascii"),
    ]
    monkeypatch.setattr(sys, "stdin", token_stdin)

    token = try_obtain_flwr_app_token(argparse.Namespace(token_stdin=True))

    assert token == _VALID_TASK_TOKEN
    assert token_stdin.buffer.read1.call_count == 2
    token_stdin.close.assert_called_once()


@pytest.mark.parametrize(
    "token_input",
    [
        "",
        "not-a-task-token",
        f"{_VALID_TASK_TOKEN}\n{_VALID_TASK_TOKEN}\n",
    ],
)
def test_try_obtain_flwr_app_token_rejects_invalid_stdin_without_disclosure(
    token_input: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid private handoffs should fail closed without echoing their input."""
    token_stdin = io.TextIOWrapper(io.BytesIO(token_input.encode("ascii")))
    monkeypatch.setattr(sys, "stdin", token_stdin)

    with pytest.raises(SystemExit) as exc_info:
        try_obtain_flwr_app_token(argparse.Namespace(token_stdin=True))

    assert token_stdin.closed
    assert not token_input or token_input not in str(exc_info.value)
