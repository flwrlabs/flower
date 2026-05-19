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


import argparse
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest

from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.interceptors import RuntimeVersionServerInterceptor
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.version import package_version
from flwr.superlink.federation import NoOpFederationManager

from . import app as app_module
from .app import _obtain_superlink_certificates, _parse_args_run_superlink
from .superlink.linkstate import LinkStateFactory


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


def test_obtain_superlink_certificates_keeps_appio_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperLink should load separate certificate tuples for Fleet and AppIO."""
    fleet_certificates = (b"fleet-ca", b"fleet-cert", b"fleet-key")
    appio_certificates = (b"appio-ca", b"appio-cert", b"appio-key")
    monkeypatch.setattr(
        app_module, "try_obtain_server_certificates", lambda _args: fleet_certificates
    )
    monkeypatch.setattr(
        app_module,
        "try_obtain_optional_appio_server_certificates",
        lambda _args: appio_certificates,
    )
    args = argparse.Namespace(insecure=False)

    certificates, appio_certificates_result = _obtain_superlink_certificates(args)

    assert certificates == fleet_certificates
    assert appio_certificates_result == appio_certificates


def test_obtain_superlink_certificates_allows_plaintext_appio_when_secure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperLink should allow plaintext ServerAppIo with secure Fleet/Control APIs."""
    fleet_certificates = (b"fleet-ca", b"fleet-cert", b"fleet-key")
    monkeypatch.setattr(
        app_module, "try_obtain_server_certificates", lambda _args: fleet_certificates
    )
    monkeypatch.setattr(
        app_module, "try_obtain_optional_appio_server_certificates", lambda _args: None
    )
    args = argparse.Namespace(insecure=False)

    certificates, appio_certificates = _obtain_superlink_certificates(args)

    assert certificates == fleet_certificates
    assert appio_certificates is None


def test_obtain_superlink_certificates_skips_cert_loading_when_insecure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperLink should not load any TLS certificates when insecure."""
    obtain_server_certificates_mock = Mock()
    obtain_appio_certificates_mock = Mock()
    monkeypatch.setattr(
        app_module, "try_obtain_server_certificates", obtain_server_certificates_mock
    )
    monkeypatch.setattr(
        app_module,
        "try_obtain_optional_appio_server_certificates",
        obtain_appio_certificates_mock,
    )
    args = argparse.Namespace(insecure=True)

    certificates, appio_certificates = _obtain_superlink_certificates(args)

    assert certificates is None
    assert appio_certificates is None
    obtain_server_certificates_mock.assert_not_called()
    obtain_appio_certificates_mock.assert_not_called()


def test_run_fleet_api_grpc_rere_adds_runtime_version_interceptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fleet gRPC-rere server should observe runtime-version metadata."""
    grpc_server = Mock()
    grpc_server.bound_address = "127.0.0.1:9092"
    create_grpc_server = Mock(return_value=grpc_server)
    existing_interceptor = Mock(spec=grpc.ServerInterceptor)
    monkeypatch.setattr(app_module, "generic_create_grpc_server", create_grpc_server)

    app_module._run_fleet_api_grpc_rere(  # pylint: disable=protected-access
        address="127.0.0.1:9092",
        state_factory=Mock(),
        objectstore_factory=Mock(),
        enable_supernode_auth=False,
        certificates=None,
        interceptors=[existing_interceptor],
    )

    interceptors = create_grpc_server.call_args.kwargs["interceptors"]
    assert interceptors[0] is existing_interceptor
    assert isinstance(interceptors[1], RuntimeVersionServerInterceptor)


@pytest.mark.parametrize(
    "database",
    [
        FLWR_IN_MEMORY_DB_NAME,
        ":memory:",
        "sqlite:///:memory:",
        "state.db",
    ],
)
def test_get_objectstore_linkstate_factories_uses_defaults(
    database: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-memory and SQLite databases should stay on default backend factories."""

    def _unexpected(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("EE resolver should not be called for default databases")

    monkeypatch.setattr(app_module, "get_ee_objectstore_factory", _unexpected)
    monkeypatch.setattr(app_module, "get_ee_linkstate_factory", _unexpected)

    federation_manager = NoOpFederationManager()
    objectstore_factory, state_factory = (
        # pylint: disable-next=protected-access
        app_module._get_objectstore_linkstate_factories(database, federation_manager)
    )

    assert isinstance(objectstore_factory, ObjectStoreFactory)
    assert isinstance(state_factory, LinkStateFactory)
    assert objectstore_factory.database == database
    assert state_factory.database == database


def test_get_objectstore_linkstate_factories_non_sqlite_without_ee_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-SQLite URL should fail if EE does not provide a backend resolver."""

    def _not_implemented(_database: str) -> object:
        raise NotImplementedError()

    monkeypatch.setattr(app_module, "get_ee_objectstore_factory", _not_implemented)

    with pytest.raises(ValueError, match="Unsupported value for `--database`"):
        app_module._get_objectstore_linkstate_factories(  # pylint: disable=protected-access
            "dummysql://user:pw@localhost/flwr", NoOpFederationManager()
        )


def test_get_objectstore_linkstate_factories_non_sqlite_uses_ee_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-SQLite URL should delegate backend factory creation to EE."""
    federation_manager = NoOpFederationManager()
    expected_objectstore_factory = ObjectStoreFactory(FLWR_IN_MEMORY_DB_NAME)
    expected_state_factory = LinkStateFactory(
        FLWR_IN_MEMORY_DB_NAME, federation_manager, expected_objectstore_factory
    )
    captured: list[object] = []

    def _objectstore_resolver(database: str) -> ObjectStoreFactory:
        captured.append(("objectstore", database))
        return expected_objectstore_factory

    def _linkstate_resolver(
        database: str,
        manager: NoOpFederationManager,
        objectstore_factory: ObjectStoreFactory,
    ) -> LinkStateFactory:
        captured.append(("linkstate", database, manager, objectstore_factory))
        return expected_state_factory

    monkeypatch.setattr(app_module, "get_ee_objectstore_factory", _objectstore_resolver)
    monkeypatch.setattr(app_module, "get_ee_linkstate_factory", _linkstate_resolver)

    objectstore_factory, state_factory = (
        # pylint: disable-next=protected-access
        app_module._get_objectstore_linkstate_factories(
            "dummysql://db.example/flwr", federation_manager
        )
    )

    assert captured == [
        ("objectstore", "dummysql://db.example/flwr"),
        (
            "linkstate",
            "dummysql://db.example/flwr",
            federation_manager,
            expected_objectstore_factory,
        ),
    ]
    assert objectstore_factory is expected_objectstore_factory
    assert state_factory is expected_state_factory
