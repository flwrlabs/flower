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
"""Tests for AgentApp process CLI parsing and wiring."""


import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flwr.supercore.cli.flwr_agentapp import _parse_args_run_flwr_agentapp
from flwr.supercore.constant import SUPERLINK_DEFAULT_CLIENT_ADDRESS

flwr_agentapp_module = importlib.import_module("flwr.supercore.cli.flwr_agentapp")


def test_parse_flwr_agentapp_requires_token() -> None:
    """The AgentApp process CLI should require a token."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_agentapp().parse_args([])


def test_parse_flwr_agentapp_rejects_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    with pytest.raises(SystemExit):
        _parse_args_run_flwr_agentapp().parse_args(
            ["--token", "test-token", "--run-once"]
        )


def test_parse_flwr_agentapp_parses_tokenized_invocation() -> None:
    """The AgentApp process CLI should still parse the supported flags."""
    args = _parse_args_run_flwr_agentapp().parse_args(
        [
            "--token",
            "test-token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.runtime_api_address == SUPERLINK_DEFAULT_CLIENT_ADDRESS
    assert args.token == "test-token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_agentapp_parses_args_before_mirroring_output() -> None:
    """Argument parsing should happen before stdout/stderr redirection."""

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Raise a parser error before any side effects happen."""
            raise SystemExit(2)

    mirror_output_to_queue = Mock()

    with (
        patch.object(flwr_agentapp_module, "_parse_args_run_flwr_agentapp", _Parser),
        patch.object(
            flwr_agentapp_module,
            "mirror_output_to_queue",
            mirror_output_to_queue,
        ),
        pytest.raises(SystemExit),
    ):
        flwr_agentapp_module.flwr_agentapp()

    mirror_output_to_queue.assert_not_called()


def test_flwr_agentapp_forwards_cli_args() -> None:
    """The AgentApp CLI should forward parsed args to the runtime."""
    args = SimpleNamespace(
        insecure=True,
        runtime_api_address="127.0.0.1:9091",
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
    run_agentapp = Mock()

    with (
        patch.object(flwr_agentapp_module, "_parse_args_run_flwr_agentapp", _Parser),
        patch.object(
            flwr_agentapp_module,
            "mirror_output_to_queue",
            mirror_output_to_queue,
        ),
        patch.object(flwr_agentapp_module, "restore_output", restore_output),
        patch.object(flwr_agentapp_module, "run_agentapp", run_agentapp),
    ):
        flwr_agentapp_module.flwr_agentapp()

    mirror_output_to_queue.assert_called_once()
    restore_output.assert_called_once_with()
    run_agentapp.assert_called_once()
    kwargs = run_agentapp.call_args.kwargs
    assert kwargs["runtime_api_address"] == "127.0.0.1:9091"
    assert kwargs["log_queue"] is mirror_output_to_queue.call_args.args[0]
    assert kwargs["token"] == "test-token"
    assert kwargs["insecure"] is True
    assert kwargs["certificates"] is None
    assert kwargs["parent_pid"] == 321
    assert kwargs["runtime_dependency_install"] is True


def test_flwr_agentapp_exposes_explicit_root_certificates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Expose the Runtime root certificate file to SDK clients."""
    monkeypatch.setenv("SSL_CERT_FILE", "inherited-ca.pem")
    certificate_path = tmp_path / "runtime-ca.pem"
    certificate_path.write_bytes(b"root-certificates")
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        insecure=False,
        runtime_api_address="runtime.example:9092",
        token="test-token",
        root_certificates=certificate_path.name,
        parent_pid=None,
        runtime_dependency_install=False,
    )

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return a fixed namespace with explicit root certificates."""
            return args

    with (
        patch.object(flwr_agentapp_module, "_parse_args_run_flwr_agentapp", _Parser),
        patch.object(flwr_agentapp_module, "mirror_output_to_queue"),
        patch.object(flwr_agentapp_module, "restore_output"),
        patch.object(flwr_agentapp_module, "run_agentapp") as run_agentapp,
    ):
        flwr_agentapp_module.flwr_agentapp()

    assert os.environ["SSL_CERT_FILE"] == str(certificate_path.resolve())
    assert run_agentapp.call_args.kwargs["certificates"] == b"root-certificates"
