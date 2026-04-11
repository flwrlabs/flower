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
"""Tests for the CLI."""


import importlib
import inspect
from types import SimpleNamespace
from typing import Annotated, Any, get_args, get_origin
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from flwr.supercore.version import package_version

from . import app as app_module
from .app import app

run_module = importlib.import_module("flwr.cli.run.run")
run_command = run_module.run

runner = CliRunner()


def _invoke_flwr(args: list[str]) -> Any:
    with patch("flwr.cli.app.warn_if_flwr_update_available"):
        return runner.invoke(app, args)


def test_version_args() -> None:
    """Test the --version flag."""
    result = _invoke_flwr(["--version"])
    assert result.exit_code == 0
    assert f"Flower version: {package_version}\n" in result.output

    # Test the -V flag
    result = _invoke_flwr(["-V"])
    assert result.exit_code == 0
    assert f"Flower version: {package_version}\n" in result.output


def test_help_command() -> None:
    """Test the -h flag."""
    result = _invoke_flwr(["-h"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Options " in result.output
    assert "Commands " in result.output


def test_new_command() -> None:
    """Add appropriate assertions for the new command."""
    result = _invoke_flwr(["new", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "new" in result.output


def test_run_command() -> None:
    """Add appropriate assertions for the run command."""
    result = _invoke_flwr(["run", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "run" in result.output


def test_run_command_accepts_remote_app_spec() -> None:
    """Remote app specs should stay as raw strings instead of path-normalized values."""
    app_annotation = inspect.signature(run_command).parameters["app"].annotation
    assert get_origin(app_annotation) is Annotated
    assert get_args(app_annotation)[0] is str

    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch.object(run_module, "read_superlink_connection") as mock_read_connection,
        patch.object(run_module, "_run_with_control_api") as mock_run_with_control_api,
    ):
        mock_read_connection.return_value = SimpleNamespace(federation=None)

        result = runner.invoke(app, ["run", "@flwrlabs/quickstart-numpy"])

    assert result.exit_code == 0
    assert mock_run_with_control_api.call_args is not None
    assert mock_run_with_control_api.call_args.args[-1] == "@flwrlabs/quickstart-numpy"


def test_build_command() -> None:
    """Add appropriate assertions for the build command."""
    result = _invoke_flwr(["build", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "build" in result.output


def test_install_command() -> None:
    """Add appropriate assertions for the install command."""
    result = _invoke_flwr(["install", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "install" in result.output


def test_log_command() -> None:
    """Add appropriate assertions for the log command."""
    result = _invoke_flwr(["log", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "log" in result.output


def test_ls_command() -> None:
    """Add appropriate assertions for the ls command."""
    result = _invoke_flwr(["ls", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "ls" in result.output


def test_stop_command() -> None:
    """Add appropriate assertions for the stop command."""
    result = _invoke_flwr(["stop", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "stop" in result.output


def test_login_command() -> None:
    """Add appropriate assertions for the login command."""
    result = _invoke_flwr(["login", "--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "login" in result.output


def test_flwr_callback_checks_for_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """The top-level flwr callback should perform the startup update check."""

    class _SentinelError(Exception):
        pass

    captured: dict[str, str] = {}

    def _raise_sentinel(process_name: str | None = None) -> None:
        if process_name is not None:
            captured["process_name"] = process_name
        raise _SentinelError()

    monkeypatch.setattr(app_module, "warn_if_flwr_update_available", _raise_sentinel)

    with pytest.raises(_SentinelError):
        app_module.main(version=False)

    assert captured == {"process_name": "flwr"}


@pytest.mark.parametrize(
    "argv", [["flwr", "ls", "--format", "json"], ["flwr", "ls", "--format=json"]]
)
def test_flwr_callback_skips_update_check_for_json_format(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    """The top-level flwr callback should skip update checks for JSON output."""
    monkeypatch.setattr("flwr.cli.app.sys.argv", argv)
    called = False

    def _warn(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(app_module, "warn_if_flwr_update_available", _warn)

    app_module.main(version=False)

    assert called is False


@pytest.mark.parametrize(
    ("argv", "should_call"),
    [
        (["flwr", "ls", "--format", "default", "--format", "json"], False),
        (["flwr", "ls", "--format", "json", "--format", "default"], True),
    ],
)
def test_flwr_callback_uses_last_format_value(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], should_call: bool
) -> None:
    """The last provided --format value should decide update-check suppression."""
    monkeypatch.setattr("flwr.cli.app.sys.argv", argv)
    called = False

    def _warn(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(app_module, "warn_if_flwr_update_available", _warn)

    app_module.main(version=False)

    assert called is should_call


def test_invalid_command() -> None:
    """Test CLI behavior with invalid commands and arguments."""
    # Test unknown command
    result = _invoke_flwr(["nonexistent-command"])
    assert result.exit_code != 0
    assert "No such command" in result.output
