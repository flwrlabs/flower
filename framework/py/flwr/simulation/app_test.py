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
"""Tests for simulation runtime wiring."""


import importlib
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flwr.common.constant import SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS

from .app import _parse_args_run_flwr_simulation, run_simulation_process

simulation_app_module = importlib.import_module("flwr.simulation.app")


class TestRunSimulationProcess(unittest.TestCase):
    """Tests for `run_simulation_process`."""

    @patch("flwr.simulation.app.flwr_exit")
    @patch("flwr.simulation.app.register_signal_handlers")
    @patch("flwr.simulation.app.SimulationIoConnection")
    def test_run_simulation_process_passes_token_to_connection(
        self,
        mock_connection_cls: Mock,
        _mock_register_signal_handlers: Mock,
        mock_flwr_exit: Mock,
    ) -> None:
        """`run_simulation_process` should pass token into SimulationIoConnection."""
        mock_conn = Mock()
        mock_conn.configure_mock(
            **{"_stub.PullAppInputs.side_effect": RuntimeError("boom")}
        )
        mock_connection_cls.return_value = mock_conn

        run_simulation_process(
            serverappio_api_address="127.0.0.1:9091",
            log_queue=Queue(),
            insecure=True,
            token="test-token",
        )

        mock_connection_cls.assert_called_once_with(
            serverappio_api_address="127.0.0.1:9091",
            insecure=True,
            root_certificates=None,
            token="test-token",
        )
        mock_flwr_exit.assert_called_once()


def test_parse_flwr_simulation_requires_token() -> None:
    """The simulation process CLI should require a token."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_simulation().parse_args([])


def test_parse_flwr_simulation_rejects_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_simulation().parse_args(
            ["--token", "test-token", "--run-once"]
        )


def test_parse_flwr_simulation_parses_tokenized_invocation() -> None:
    """The simulation process CLI should still parse the supported flags."""
    args = _parse_args_run_flwr_simulation().parse_args(
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


def test_flwr_simulation_parses_args_before_mirroring_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Argument parsing should happen before stdout/stderr redirection."""

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Raise a parser error before any side effects happen."""
            raise SystemExit(2)

    calls: list[str] = []

    monkeypatch.setattr(
        simulation_app_module, "_parse_args_run_flwr_simulation", _Parser
    )
    monkeypatch.setattr(
        simulation_app_module,
        "mirror_output_to_queue",
        lambda *_args, **_kwargs: calls.append("mirror"),
    )

    with pytest.raises(SystemExit):
        simulation_app_module.flwr_simulation()

    assert not calls


def test_flwr_simulation_forwards_cli_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The simulation CLI should forward parsed args to the runtime."""
    args = SimpleNamespace(
        insecure=True,
        serverappio_api_address="127.0.0.1:9091",
        token="test-token",
        root_certificates=None,
        parent_pid=321,
        runtime_dependency_install=True,
    )
    calls: list[str] = []
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return a fixed namespace for CLI forwarding tests."""
            return args

    def _mirror_output_to_queue(*_args: object, **_kwargs: object) -> None:
        calls.append("mirror")

    def _restore_output() -> None:
        calls.append("restore")

    def _run_simulation_process(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        simulation_app_module, "_parse_args_run_flwr_simulation", _Parser
    )
    monkeypatch.setattr(
        simulation_app_module, "mirror_output_to_queue", _mirror_output_to_queue
    )
    monkeypatch.setattr(simulation_app_module, "restore_output", _restore_output)
    monkeypatch.setattr(
        simulation_app_module, "run_simulation_process", _run_simulation_process
    )

    simulation_app_module.flwr_simulation()

    assert calls == ["mirror", "restore"]
    assert captured["serverappio_api_address"] == "127.0.0.1:9091"
    assert captured["token"] == "test-token"
    assert captured["certificates"] is None
    assert captured["parent_pid"] == 321
    assert captured["runtime_dependency_install"] is True
