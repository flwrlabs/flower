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
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from flwr.common.constant import FLWR_TASK_TOKEN_LENGTH
from flwr.supercore.cli.flwr_agentapp import (
    _parse_args_run_flwr_agentapp,
    _try_obtain_agentapp_token,
)
from flwr.supercore.constant import SUPERLINK_DEFAULT_CLIENT_ADDRESS

flwr_agentapp_module = importlib.import_module("flwr.supercore.cli.flwr_agentapp")
_VALID_TASK_TOKEN = "a" * (FLWR_TASK_TOKEN_LENGTH * 2)


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


def test_parse_flwr_agentapp_accepts_only_one_token_source() -> None:
    """The private stdin mode should be opt-in and mutually exclusive."""
    parser = _parse_args_run_flwr_agentapp()

    assert parser.parse_args(["--token-stdin"]).token_stdin is True

    with pytest.raises(SystemExit):
        parser.parse_args(["--token", "test-token", "--token-stdin"])


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
    assert kwargs["certificates_path"] is None
    assert kwargs["parent_pid"] == 321
    assert kwargs["runtime_dependency_install"] is True


def test_flwr_agentapp_reads_stdin_token_and_acknowledges_start(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An open sender should receive acknowledgement after one token write."""
    read_fd, write_fd = os.pipe()
    token_stdin = os.fdopen(read_fd, encoding="ascii")
    token_writer = os.fdopen(write_fd, "w", encoding="ascii")
    token_writer.write(f"{_VALID_TASK_TOKEN}\n")
    token_writer.flush()
    monkeypatch.setattr(sys, "stdin", token_stdin)
    monkeypatch.setattr(sys, "argv", ["flwr-agentapp", "--token-stdin"])

    try:
        with (
            patch.object(flwr_agentapp_module, "mirror_output_to_queue"),
            patch.object(flwr_agentapp_module, "restore_output"),
            patch.object(flwr_agentapp_module, "run_agentapp") as run_agentapp,
        ):
            flwr_agentapp_module.flwr_agentapp()
    finally:
        token_writer.close()

    assert token_stdin.closed
    assert capsys.readouterr().out == "FLWR_AGENTAPP_TOKEN_ACCEPTED\n"
    run_agentapp.assert_called_once()
    assert run_agentapp.call_args.kwargs["token"] == _VALID_TASK_TOKEN


def test_flwr_agentapp_accumulates_split_stdin_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid token can arrive across multiple stdin reads."""
    token_stdin = Mock()
    token_stdin.buffer.read1.side_effect = [
        _VALID_TASK_TOKEN[:128].encode("ascii"),
        f"{_VALID_TASK_TOKEN[128:]}\n".encode("ascii"),
    ]
    monkeypatch.setattr(sys, "stdin", token_stdin)
    args = _parse_args_run_flwr_agentapp().parse_args(["--token-stdin"])

    assert _try_obtain_agentapp_token(args) == _VALID_TASK_TOKEN
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
def test_flwr_agentapp_rejects_invalid_stdin_without_disclosure(
    token_input: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid private handoffs should fail closed without echoing their input."""
    token_stdin = io.TextIOWrapper(io.BytesIO(token_input.encode("ascii")))
    monkeypatch.setattr(sys, "stdin", token_stdin)
    args = _parse_args_run_flwr_agentapp().parse_args(["--token-stdin"])

    with pytest.raises(SystemExit) as exc_info:
        _try_obtain_agentapp_token(args)

    assert token_stdin.closed
    assert not token_input or token_input not in str(exc_info.value)


def test_flwr_agentapp_forwards_explicit_root_certificates_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Forward the resolved Runtime root certificate path to the runtime."""
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

    assert os.environ["SSL_CERT_FILE"] == "inherited-ca.pem"
    assert run_agentapp.call_args.kwargs["certificates_path"] == str(
        certificate_path.resolve()
    )
