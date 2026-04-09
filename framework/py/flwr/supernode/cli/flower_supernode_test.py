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
from types import SimpleNamespace

import pytest

from flwr.common.constant import ISOLATION_MODE_PROCESS, ISOLATION_MODE_SUBPROCESS
from flwr.common.exit import ExitCode
from flwr.supercore.version import package_version

from .flower_supernode import _parse_args_run_supernode

flower_supernode_module = importlib.import_module("flwr.supernode.cli.flower_supernode")


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess mode should not load a SuperExec secret file."""
    args = SimpleNamespace(
        trusted_entities=None,
        superlink="127.0.0.1:9092",
        transport="grpc-rere",
        max_retries=None,
        max_wait_time=None,
        node_config=None,
        isolation=ISOLATION_MODE_SUBPROCESS,
        clientappio_api_address="127.0.0.1:9094",
        health_server_address=None,
        superexec_auth_secret_file="ignored-secret-file",
        insecure=True,
        auth_supernode_private_key=None,
    )
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return fixed args namespace for subprocess-mode test."""
            return args

    def _start_client_internal(**kwargs: object) -> None:
        """Capture startup kwargs for assertions."""
        captured.update(kwargs)

    def _warn_if_flwr_update_available(**_: object) -> None:
        """Disable update check side effects in unit test."""
        return

    def _parse_args_run_supernode() -> _Parser:
        """Return parser stub with fixed args."""
        return _Parser()

    monkeypatch.setattr(
        flower_supernode_module,
        "warn_if_flwr_update_available",
        _warn_if_flwr_update_available,
    )
    monkeypatch.setattr(
        flower_supernode_module, "_parse_args_run_supernode", _parse_args_run_supernode
    )
    monkeypatch.setattr(
        flower_supernode_module, "_try_obtain_trusted_entities", lambda *_: None
    )
    monkeypatch.setattr(
        flower_supernode_module, "try_obtain_root_certificates", lambda *_: None
    )
    monkeypatch.setattr(
        flower_supernode_module, "_try_setup_client_authentication", lambda *_: None
    )
    monkeypatch.setattr(flower_supernode_module, "parse_config_args", lambda *_: {})
    monkeypatch.setattr(flower_supernode_module, "event", lambda *_: None)
    monkeypatch.setattr(
        flower_supernode_module,
        "load_superexec_auth_secret",
        lambda **_: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(
        flower_supernode_module, "start_client_internal", _start_client_internal
    )

    flower_supernode_module.flower_supernode()

    assert captured["superexec_auth_secret"] is None
    assert captured["isolation"] == ISOLATION_MODE_SUBPROCESS


def test_flower_supernode_process_exits_on_invalid_superexec_secret_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process mode should exit with dedicated code if secret load fails."""
    args = SimpleNamespace(
        trusted_entities=None,
        superlink="127.0.0.1:9092",
        transport="grpc-rere",
        max_retries=None,
        max_wait_time=None,
        node_config=None,
        isolation=ISOLATION_MODE_PROCESS,
        clientappio_api_address="127.0.0.1:9094",
        health_server_address=None,
        superexec_auth_secret_file="bad-secret-file",
        insecure=True,
        auth_supernode_private_key=None,
    )

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return fixed args namespace for process-mode test."""
            return args

    class _ExitCalled(Exception):
        def __init__(self, code: int, message: str) -> None:
            super().__init__(message)
            self.code = code

    def _flwr_exit(code: int, message: str) -> None:
        """Raise captured exception instead of exiting test process."""
        raise _ExitCalled(code, message)

    def _warn_if_flwr_update_available(**_: object) -> None:
        """Disable update check side effects in unit test."""
        return

    def _parse_args_run_supernode() -> _Parser:
        """Return parser stub with fixed args."""
        return _Parser()

    monkeypatch.setattr(
        flower_supernode_module,
        "warn_if_flwr_update_available",
        _warn_if_flwr_update_available,
    )
    monkeypatch.setattr(
        flower_supernode_module, "_parse_args_run_supernode", _parse_args_run_supernode
    )
    monkeypatch.setattr(
        flower_supernode_module, "_try_obtain_trusted_entities", lambda *_: None
    )
    monkeypatch.setattr(
        flower_supernode_module, "try_obtain_root_certificates", lambda *_: None
    )
    monkeypatch.setattr(
        flower_supernode_module, "_try_setup_client_authentication", lambda *_: None
    )
    monkeypatch.setattr(flower_supernode_module, "event", lambda *_: None)
    monkeypatch.setattr(
        flower_supernode_module,
        "load_superexec_auth_secret",
        lambda **_: (_ for _ in ()).throw(ValueError("invalid")),
    )
    monkeypatch.setattr(flower_supernode_module, "flwr_exit", _flwr_exit)

    with pytest.raises(_ExitCalled) as err:
        flower_supernode_module.flower_supernode()

    assert err.value.code == ExitCode.SUPEREXEC_AUTH_SECRET_LOAD_FAILED
