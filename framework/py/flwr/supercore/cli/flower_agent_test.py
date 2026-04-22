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
"""Tests for Flower Agent CLI argument parsing and wiring."""


import importlib
from types import SimpleNamespace

import pytest

from flwr.common import EventType
from flwr.supercore.version import package_version

from .flower_agent import _parse_args

flower_agent_module = importlib.import_module("flwr.supercore.cli.flower_agent")


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_parse_agent_version_flag(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The version flags should print the package version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_args().parse_args([flag])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"Flower version: {package_version}\n"


def test_flower_agent_checks_for_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flower Agent should run the startup update check before parsing args."""

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

    monkeypatch.setattr(flower_agent_module, "_parse_args", _unexpected_parse_args)
    monkeypatch.setattr(
        flower_agent_module, "disable_process_dumping", lambda **_: None
    )
    monkeypatch.setattr(
        flower_agent_module, "warn_if_flwr_update_available", _raise_sentinel
    )

    with pytest.raises(_SentinelError):
        flower_agent_module.flower_agent()

    assert captured == ["update", "flower-agent"]


def test_flower_agent_forwards_cli_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flower Agent CLI should forward parsed arguments to the runtime stub."""
    args = SimpleNamespace(
        insecure=True,
        appio_api_address="127.0.0.1:9091",
        parent_pid=4321,
        health_server_address="127.0.0.1:9099",
        superexec_auth_secret_file="/tmp/agent-secret",
        runtime_dependency_install=True,
    )
    captured: dict[str, object] = {}
    events: list[tuple[EventType, dict[str, object] | None]] = []

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return args

    def _run_flower_agent(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        flower_agent_module, "disable_process_dumping", lambda **_: None
    )
    monkeypatch.setattr(
        flower_agent_module,
        "warn_if_flwr_update_available",
        lambda **_: None,
    )
    monkeypatch.setattr(
        flower_agent_module, "load_superexec_auth_secret", lambda **_: b"secret"
    )
    monkeypatch.setattr(
        flower_agent_module,
        "event",
        lambda event_type, event_details=None: events.append(
            (event_type, event_details)
        ),
    )
    monkeypatch.setattr(flower_agent_module, "_parse_args", _Parser)
    monkeypatch.setattr(flower_agent_module, "run_flower_agent", _run_flower_agent)

    flower_agent_module.flower_agent()

    assert events == [(EventType.RUN_AGENT_ENTER, None)]
    assert captured == {
        "appio_api_address": "127.0.0.1:9091",
        "parent_pid": 4321,
        "health_server_address": "127.0.0.1:9099",
        "superexec_auth_secret": b"secret",
        "runtime_dependency_install": True,
    }


def test_flower_agent_allows_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flower Agent should allow missing auth secret like Flower SuperExec."""
    args = SimpleNamespace(
        insecure=True,
        appio_api_address="127.0.0.1:9091",
        superexec_auth_secret_file=None,
        parent_pid=None,
        health_server_address=None,
        runtime_dependency_install=False,
    )
    captured: dict[str, object] = {}

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return args

    def _run_flower_agent(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        flower_agent_module, "disable_process_dumping", lambda **_: None
    )
    monkeypatch.setattr(
        flower_agent_module,
        "warn_if_flwr_update_available",
        lambda **_: None,
    )
    monkeypatch.setattr(flower_agent_module, "_parse_args", _Parser)
    monkeypatch.setattr(flower_agent_module, "event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(flower_agent_module, "run_flower_agent", _run_flower_agent)

    flower_agent_module.flower_agent()

    assert captured["superexec_auth_secret"] is None
