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
"""Tests for Flower SuperNode CLI argument parsing."""


import importlib
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from flwr.common.constant import ISOLATION_MODE_PROCESS, ISOLATION_MODE_SUBPROCESS
from flwr.common.exit import ExitCode
from flwr.supercore.version import package_version

from .flower_supernode import _parse_args_run_supernode

flower_supernode_module = importlib.import_module("flwr.supernode.cli.flower_supernode")


def _make_args(**overrides: object) -> SimpleNamespace:
    """Build a default SuperNode CLI args namespace with optional overrides."""
    args = {
        "trusted_entities": None,
        "superlink": "127.0.0.1:9092",
        "transport": "grpc-rere",
        "max_retries": None,
        "max_wait_time": None,
        "node_config": None,
        "isolation": ISOLATION_MODE_SUBPROCESS,
        "clientappio_api_address": "127.0.0.1:9094",
        "health_server_address": None,
        "superexec_auth_secret_file": None,
        "insecure": True,
        "auth_supernode_private_key": None,
        "runtime_dependency_install": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


class _FixedArgsParser:
    """Simple parser stub that always returns a fixed args namespace."""

    def __init__(self, args: SimpleNamespace) -> None:
        self._args = args

    def parse_args(self) -> SimpleNamespace:
        """Return fixed args namespace for test paths."""
        return self._args


@dataclass
class _CliHarness:
    """Shared harness for CLI tests with patched startup hooks."""

    args: SimpleNamespace
    start_kwargs: dict[str, object] = field(default_factory=dict)

    def capture_start_client_internal(self, **kwargs: object) -> None:
        """Capture `start_client_internal` kwargs for assertions."""
        self.start_kwargs.update(kwargs)


def _return_none(*_args: object, **_kwargs: object) -> None:
    """Return None for monkeypatched dependency stubs."""


def _parse_config_args_stub(*_args: object, **_kwargs: object) -> dict[str, str]:
    """Return empty parsed config for monkeypatched CLI test paths."""
    return {}


def _patch_common_supernode_startup(
    monkeypatch: pytest.MonkeyPatch, harness: _CliHarness
) -> None:
    """Patch common SuperNode startup dependencies for CLI tests."""

    def _parse_args_run_supernode() -> _FixedArgsParser:
        return _FixedArgsParser(harness.args)

    monkeypatch.setattr(
        flower_supernode_module, "warn_if_flwr_update_available", _return_none
    )
    monkeypatch.setattr(
        flower_supernode_module, "_parse_args_run_supernode", _parse_args_run_supernode
    )
    monkeypatch.setattr(
        flower_supernode_module, "_try_obtain_trusted_entities", _return_none
    )
    monkeypatch.setattr(
        flower_supernode_module, "try_obtain_root_certificates", _return_none
    )
    monkeypatch.setattr(
        flower_supernode_module, "_try_setup_client_authentication", _return_none
    )
    monkeypatch.setattr(
        flower_supernode_module, "parse_config_args", _parse_config_args_stub
    )
    monkeypatch.setattr(flower_supernode_module, "event", _return_none)
    monkeypatch.setattr(
        flower_supernode_module,
        "start_client_internal",
        harness.capture_start_client_internal,
    )


@pytest.fixture(name="cli_harness")
def fixture_cli_harness(monkeypatch: pytest.MonkeyPatch) -> _CliHarness:
    """Create and patch a reusable SuperNode CLI test harness."""
    harness = _CliHarness(args=_make_args())
    _patch_common_supernode_startup(monkeypatch, harness)
    return harness


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_parse_supernode_version_flag(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The version flags should print the package version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_args_run_supernode().parse_args([flag])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"Flower version: {package_version}\n"


def test_flower_supernode_checks_for_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperNode should run the startup update check before parsing arguments."""

    class _SentinelError(Exception):
        pass

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return SimpleNamespace()

    def _parse_args() -> _Parser:
        return _Parser()

    captured: list[str] = []

    def _raise_sentinel(process_name: str | None = None) -> None:
        captured.append("update")
        if process_name is not None:
            captured.append(process_name)
        raise _SentinelError()

    def _unexpected_parse_args() -> _Parser:
        captured.append("parse")
        return _parse_args()

    monkeypatch.setattr(
        flower_supernode_module, "_parse_args_run_supernode", _unexpected_parse_args
    )
    monkeypatch.setattr(
        flower_supernode_module, "warn_if_flwr_update_available", _raise_sentinel
    )

    with pytest.raises(_SentinelError):
        flower_supernode_module.flower_supernode()

    assert captured == ["update", "flower-supernode"]


def test_flower_supernode_subprocess_does_not_load_superexec_secret(
    cli_harness: _CliHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess mode should not load a SuperExec secret file."""
    cli_harness.args.isolation = ISOLATION_MODE_SUBPROCESS
    cli_harness.args.superexec_auth_secret_file = "ignored-secret-file"

    def _unexpected_load_superexec_auth_secret(**_: object) -> bytes:
        """Fail fast if subprocess path unexpectedly loads a secret."""
        raise AssertionError("should not be called")

    monkeypatch.setattr(
        flower_supernode_module,
        "load_superexec_auth_secret",
        _unexpected_load_superexec_auth_secret,
    )

    flower_supernode_module.flower_supernode()

    assert cli_harness.start_kwargs["superexec_auth_secret"] is None
    assert cli_harness.start_kwargs["isolation"] == ISOLATION_MODE_SUBPROCESS


def test_flower_supernode_process_exits_on_invalid_superexec_secret_file(
    cli_harness: _CliHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process mode should exit with dedicated code if secret load fails."""
    cli_harness.args.isolation = ISOLATION_MODE_PROCESS
    cli_harness.args.superexec_auth_secret_file = "bad-secret-file"

    class _ExitCalled(Exception):
        def __init__(self, code: int, message: str) -> None:
            super().__init__(message)
            self.code = code

    def _flwr_exit(code: int, message: str) -> None:
        """Raise captured exception instead of exiting test process."""
        raise _ExitCalled(code, message)

    def _raise_invalid_secret(**_: object) -> bytes:
        """Simulate invalid secret-file parsing failure."""
        raise ValueError("invalid")

    monkeypatch.setattr(
        flower_supernode_module,
        "load_superexec_auth_secret",
        _raise_invalid_secret,
    )
    monkeypatch.setattr(flower_supernode_module, "flwr_exit", _flwr_exit)

    with pytest.raises(_ExitCalled) as err:
        flower_supernode_module.flower_supernode()

    assert err.value.code == ExitCode.SUPEREXEC_AUTH_SECRET_LOAD_FAILED
