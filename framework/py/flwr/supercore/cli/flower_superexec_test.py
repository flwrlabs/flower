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
"""Tests for Flower SuperExec CLI argument parsing."""


import importlib
from types import SimpleNamespace

import pytest

from flwr.common.constant import ExecPluginType
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.version import package_version

from .flower_superexec import _parse_args

flower_superexec_module = importlib.import_module("flwr.supercore.cli.flower_superexec")


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_parse_superexec_version_flag(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The version flags should print the package version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_args().parse_args([flag])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"Flower version: {package_version}\n"


def test_flower_superexec_checks_for_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperExec should run the startup update check before parsing arguments."""

    class _SentinelError(Exception):
        pass

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return SimpleNamespace(insecure=True)

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

    monkeypatch.setattr(flower_superexec_module, "_parse_args", _unexpected_parse_args)
    monkeypatch.setattr(
        flower_superexec_module, "warn_if_flwr_update_available", _raise_sentinel
    )

    with pytest.raises(_SentinelError):
        flower_superexec_module.flower_superexec()

    assert captured == ["update", "flower-superexec"]


def test_flower_superexec_clientapp_allows_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ClientApp plugin should not require a SuperExec auth secret."""
    args = SimpleNamespace(
        insecure=True,
        plugin_type=ExecPluginType.CLIENT_APP,
        plugin_config=None,
        root_certificates=None,
        superexec_auth_secret_file=None,
        appio_api_address="127.0.0.1:9091",
        parent_pid=None,
        health_server_address=None,
        runtime_dependency_install=False,
    )
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return args

    def _run_superexec(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        flower_superexec_module,
        "warn_if_flwr_update_available",
        lambda **_: None,
    )
    monkeypatch.setattr(flower_superexec_module, "_parse_args", _Parser)
    monkeypatch.setattr(
        flower_superexec_module,
        "_get_plugin_and_stub_class",
        lambda _plugin_type: (object, ClientAppIoStub),
    )
    monkeypatch.setattr(flower_superexec_module, "run_superexec", _run_superexec)

    flower_superexec_module.flower_superexec()

    assert captured["superexec_auth_secret"] is None


def test_flower_superexec_serverapp_allows_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ServerApp plugin should allow missing secret in subprocess-mode flows."""
    args = SimpleNamespace(
        insecure=True,
        plugin_type=ExecPluginType.SERVER_APP,
        plugin_config=None,
        root_certificates=None,
        superexec_auth_secret_file=None,
        appio_api_address="127.0.0.1:9091",
        parent_pid=None,
        health_server_address=None,
        runtime_dependency_install=False,
    )

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return args

    captured: dict[str, object] = {}

    def _run_superexec(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        flower_superexec_module,
        "warn_if_flwr_update_available",
        lambda **_: None,
    )
    monkeypatch.setattr(flower_superexec_module, "_parse_args", _Parser)
    monkeypatch.setattr(
        flower_superexec_module,
        "_get_plugin_and_stub_class",
        lambda _plugin_type: (object, ServerAppIoStub),
    )
    monkeypatch.setattr(flower_superexec_module, "run_superexec", _run_superexec)

    flower_superexec_module.flower_superexec()

    assert captured["superexec_auth_secret"] is None
