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
"""Tests for ServerApp process CLI parsing and wiring."""


import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flwr.common.constant import SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS

from .app import _parse_args_run_flwr_serverapp

serverapp_module = importlib.import_module("flwr.server.serverapp.app")


def test_parse_flwr_serverapp_requires_token() -> None:
    """The ServerApp process CLI should require a token."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_serverapp().parse_args([])


def test_parse_flwr_serverapp_rejects_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_serverapp().parse_args(
            ["--token", "test-token", "--run-once"]
        )


def test_parse_flwr_serverapp_parses_tokenized_invocation() -> None:
    """The ServerApp process CLI should still parse the supported flags."""
    args = _parse_args_run_flwr_serverapp().parse_args(
        [
            "--token",
            "test-token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.serverappio_api_address == SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
    assert args.token == "test-token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_serverapp_parses_args_before_mirroring_output() -> None:
    """Argument parsing should happen before stdout/stderr redirection."""

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Raise a parser error before any side effects happen."""
            raise SystemExit(2)

    mirror_output_to_queue = Mock()

    with (
        patch.object(serverapp_module, "_parse_args_run_flwr_serverapp", _Parser),
        patch.object(
            serverapp_module,
            "mirror_output_to_queue",
            mirror_output_to_queue,
        ),
        pytest.raises(SystemExit),
    ):
        serverapp_module.flwr_serverapp()

    mirror_output_to_queue.assert_not_called()


def test_flwr_serverapp_forwards_cli_args() -> None:
    """The ServerApp CLI should forward parsed args to the runtime."""
    args = SimpleNamespace(
        insecure=True,
        serverappio_api_address="127.0.0.1:9091",
        token="test-token",
        root_certificates=None,
        parent_pid=321,
        runtime_dependency_install=True,
    )

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return a fixed namespace for CLI forwarding tests."""
            return args

    mirror_output_to_queue = Mock()
    restore_output = Mock()
    run_serverapp = Mock()

    with (
        patch.object(serverapp_module, "_parse_args_run_flwr_serverapp", _Parser),
        patch.object(
            serverapp_module,
            "mirror_output_to_queue",
            mirror_output_to_queue,
        ),
        patch.object(serverapp_module, "restore_output", restore_output),
        patch.object(serverapp_module, "run_serverapp", run_serverapp),
    ):
        serverapp_module.flwr_serverapp()

    mirror_output_to_queue.assert_called_once()
    restore_output.assert_called_once_with()
    run_serverapp.assert_called_once()
    kwargs = run_serverapp.call_args.kwargs
    assert kwargs["serverappio_api_address"] == "127.0.0.1:9091"
    assert kwargs["token"] == "test-token"
    assert kwargs["certificates"] is None
    assert kwargs["parent_pid"] == 321
    assert kwargs["runtime_dependency_install"] is True
