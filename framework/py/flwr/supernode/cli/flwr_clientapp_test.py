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
"""Tests for ClientApp process CLI parsing and wiring."""


import importlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flwr.common.constant import CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS

from .flwr_clientapp import _parse_args_run_flwr_clientapp

flwr_clientapp_module = importlib.import_module("flwr.supernode.cli.flwr_clientapp")


def test_parse_flwr_clientapp_requires_token() -> None:
    """The ClientApp process CLI should require a token."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_clientapp().parse_args([])


def test_parse_flwr_clientapp_rejects_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_clientapp().parse_args(
            ["--token", "test-token", "--run-once"]
        )


def test_parse_flwr_clientapp_parses_tokenized_invocation() -> None:
    """The ClientApp process CLI should still parse the supported flags."""
    args = _parse_args_run_flwr_clientapp().parse_args(
        [
            "--token",
            "test-token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.clientappio_api_address == CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS
    assert args.token == "test-token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_clientapp_forwards_cli_args() -> None:
    """The ClientApp CLI should forward parsed args to the runtime."""
    args = SimpleNamespace(
        insecure=True,
        clientappio_api_address="127.0.0.1:9094",
        token="test-token",
        parent_pid=321,
        runtime_dependency_install=True,
    )

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return a fixed namespace for CLI forwarding tests."""
            return args

    run_clientapp = Mock()

    with (
        patch.object(flwr_clientapp_module, "_parse_args_run_flwr_clientapp", _Parser),
        patch.object(flwr_clientapp_module, "run_clientapp", run_clientapp),
    ):
        flwr_clientapp_module.flwr_clientapp()

    run_clientapp.assert_called_once()
    kwargs = run_clientapp.call_args.kwargs
    assert kwargs["clientappio_api_address"] == "127.0.0.1:9094"
    assert kwargs["token"] == "test-token"
    assert kwargs["certificates"] is None
    assert kwargs["parent_pid"] == 321
    assert kwargs["runtime_dependency_install"] is True
