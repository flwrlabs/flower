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
"""Tests for Flower SuperLink app CLI argument parsing."""


import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flwr.common.constant import ISOLATION_MODE_SUBPROCESS, TRANSPORT_TYPE_GRPC_RERE
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.version import package_version

from . import app as app_module
from .app import _parse_args_run_superlink


def test_parse_superlink_log_rotation_args_defaults() -> None:
    """SuperLink log rotation args should have expected defaults."""
    # Execute
    args = _parse_args_run_superlink().parse_args([])

    # Assert
    assert args.log_file is None
    assert args.log_rotation_interval_hours == 24
    assert args.log_rotation_backup_count == 7


def test_parse_superlink_log_rotation_args_custom_values() -> None:
    """SuperLink log rotation args should parse explicit values."""
    # Execute
    args = _parse_args_run_superlink().parse_args(
        [
            "--log-file",
            "/tmp/superlink.log",
            "--log-rotation-interval-hours",
            "12",
            "--log-rotation-backup-count",
            "14",
        ]
    )

    # Assert
    assert args.log_file == "/tmp/superlink.log"
    assert args.log_rotation_interval_hours == 12
    assert args.log_rotation_backup_count == 14


def test_parse_superlink_appio_tls_args() -> None:
    """SuperLink should parse AppIO-specific TLS args for ServerAppIo."""
    args = _parse_args_run_superlink().parse_args(
        [
            "--appio-ssl-certfile",
            "appio-cert.pem",
            "--appio-ssl-keyfile",
            "appio-key.pem",
            "--appio-ssl-ca-certfile",
            "appio-ca.pem",
        ]
    )

    assert args.appio_ssl_certfile == "appio-cert.pem"
    assert args.appio_ssl_keyfile == "appio-key.pem"
    assert args.appio_ssl_ca_certfile == "appio-ca.pem"


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_parse_superlink_version_flag(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The version flags should print the package version and exit."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_args_run_superlink().parse_args([flag])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"Flower version: {package_version}\n"


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_superlink_log_rotation_interval_requires_positive_int(
    value: str,
) -> None:
    """The interval must be a positive integer."""
    with pytest.raises(SystemExit):
        _parse_args_run_superlink().parse_args(["--log-rotation-interval-hours", value])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_superlink_log_rotation_backup_requires_positive_int(
    value: str,
) -> None:
    """The backup count must be a positive integer."""
    with pytest.raises(SystemExit):
        _parse_args_run_superlink().parse_args(["--log-rotation-backup-count", value])


def test_run_superlink_checks_for_update(monkeypatch: pytest.MonkeyPatch) -> None:
    """SuperLink should run the startup update check before parsing arguments."""

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

    monkeypatch.setattr(app_module, "_parse_args_run_superlink", _unexpected_parse_args)
    monkeypatch.setattr(app_module, "warn_if_flwr_update_available", _raise_sentinel)

    with pytest.raises(_SentinelError):
        app_module.run_superlink()

    assert captured == ["update", "flower-superlink"]


def _superlink_args(**overrides: object) -> SimpleNamespace:
    """Return SuperLink args for startup wiring tests."""
    args: dict[str, object] = {
        "account_auth_config": None,
        "appio_ssl_ca_certfile": "appio-ca.pem",
        "appio_ssl_certfile": "appio-cert.pem",
        "appio_ssl_keyfile": "appio-key.pem",
        "artifact_provider_config": None,
        "auth_list_public_keys": None,
        "control_api_address": "127.0.0.1:9093",
        "database": FLWR_IN_MEMORY_DB_NAME,
        "disable_oidc_tls_cert_verification": False,
        "enable_event_log": False,
        "enable_supernode_auth": False,
        "exec_api_address": None,
        "executor": None,
        "executor_config": None,
        "executor_dir": None,
        "fleet_api_address": None,
        "fleet_api_num_workers": 1,
        "fleet_api_type": TRANSPORT_TYPE_GRPC_RERE,
        "health_server_address": None,
        "insecure": False,
        "isolation": ISOLATION_MODE_SUBPROCESS,
        "log_file": None,
        "runtime_dependency_install": False,
        "serverappio_api_address": "127.0.0.1:9091",
        "simulation": True,
        "ssl_ca_certfile": "fleet-ca.pem",
        "ssl_certfile": "fleet-cert.pem",
        "ssl_keyfile": "fleet-key.pem",
        "superexec_auth_secret_file": None,
        "user_auth_config": None,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def test_run_superlink_uses_appio_certificates_for_serverappio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ServerAppIo and SuperExec should use AppIO TLS material, not Fleet TLS."""

    class _StopAfterRegister(Exception):
        pass

    fleet_certificates = (b"fleet-ca", b"fleet-cert", b"fleet-key")
    appio_certificates = (b"appio-ca", b"appio-cert", b"appio-key")
    state_factory = Mock()
    control_server = Mock()
    serverappio_server = Mock()
    serverappio_server.bound_address = "127.0.0.1:9091"
    popen = Mock()
    run_control = Mock(return_value=control_server)
    run_serverappio = Mock(return_value=serverappio_server)

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return _superlink_args()

    monkeypatch.setattr(sys, "argv", ["flower-superlink"])
    monkeypatch.setattr(app_module, "warn_if_flwr_update_available", Mock())
    monkeypatch.setattr(
        app_module, "_parse_args_run_superlink", Mock(return_value=_Parser())
    )
    monkeypatch.setattr(
        app_module, "try_obtain_server_certificates", lambda _args: fleet_certificates
    )
    monkeypatch.setattr(
        app_module,
        "try_obtain_optional_appio_server_certificates",
        lambda _args: appio_certificates,
    )
    monkeypatch.setattr(app_module, "ObjectStoreFactory", Mock())
    monkeypatch.setattr(
        app_module, "LinkStateFactory", Mock(return_value=state_factory)
    )
    monkeypatch.setattr(app_module, "run_control_api_grpc", run_control)
    monkeypatch.setattr(app_module, "run_serverappio_api_grpc", run_serverappio)
    monkeypatch.setattr("flwr.server.app.subprocess.Popen", popen)
    monkeypatch.setattr(
        app_module,
        "register_signal_handlers",
        Mock(side_effect=_StopAfterRegister),
    )

    with pytest.raises(_StopAfterRegister):
        app_module.run_superlink()

    run_control.assert_called_once()
    assert run_control.call_args.kwargs["certificates"] == fleet_certificates
    run_serverappio.assert_called_once()
    assert run_serverappio.call_args.kwargs["certificates"] == appio_certificates
    assert run_serverappio.call_args.kwargs["certificates"] != fleet_certificates
    command = popen.call_args.args[0]
    assert command[:3] == ["flower-superexec", "--root-certificates", "appio-ca.pem"]


def test_run_superlink_requires_appio_certificates_when_secure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secure SuperLink should fail if ServerAppIo TLS material is omitted."""
    fleet_certificates = (b"fleet-ca", b"fleet-cert", b"fleet-key")

    class _Parser:
        def parse_args(self) -> SimpleNamespace:
            """Return parsed arguments for the test path."""
            return _superlink_args(
                appio_ssl_ca_certfile=None,
                appio_ssl_certfile=None,
                appio_ssl_keyfile=None,
            )

    monkeypatch.setattr(sys, "argv", ["flower-superlink"])
    monkeypatch.setattr(app_module, "warn_if_flwr_update_available", Mock())
    monkeypatch.setattr(
        app_module, "_parse_args_run_superlink", Mock(return_value=_Parser())
    )
    monkeypatch.setattr(
        app_module, "try_obtain_server_certificates", lambda _args: fleet_certificates
    )
    monkeypatch.setattr(
        app_module, "try_obtain_optional_appio_server_certificates", lambda _args: None
    )

    with pytest.raises(SystemExit) as exc_info:
        app_module.run_superlink()

    assert "--appio-ssl-certfile" in str(exc_info.value)
