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
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flwr.common.constant import (
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    ISOLATION_MODE_SUBPROCESS,
    SUPERNODE_RUNTIME_API_DEFAULT_SERVER_ADDRESS,
    TRANSPORT_TYPE_GRPC_RERE,
)
from flwr.supercore.constant import UVICORN_DEFAULT_HOST, UVICORN_DEFAULT_PORT
from flwr.supercore.version import package_version

from .flower_supernode import (
    _parse_args_run_supernode,
    _parse_supernode_lifespan_config,
)

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


def test_parse_supernode_appio_tls_args() -> None:
    """SuperNode should parse AppIO-named TLS args for its Runtime API."""
    args = _parse_args_run_supernode().parse_args(
        [
            "--appio-ssl-certfile",
            "appio-cert.pem",
            "--appio-ssl-keyfile",
            "appio-key.pem",
            "--appio-ssl-ca-certfile",
            "appio-ca.pem",
        ]
    )

    assert args.runtime_ssl_certfile == "appio-cert.pem"
    assert args.runtime_ssl_keyfile == "appio-key.pem"
    assert args.runtime_ssl_ca_certfile == "appio-ca.pem"


def test_parse_supernode_lifespan_config_returns_final_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperNode CLI parsing should return the final lifespan config."""
    monkeypatch.setattr(sys, "argv", ["flower-supernode", "--insecure"])

    config = _parse_supernode_lifespan_config()

    assert config.server_address == FLEET_API_GRPC_RERE_DEFAULT_ADDRESS
    assert config.transport == TRANSPORT_TYPE_GRPC_RERE
    assert config.root_certificates is None
    assert config.insecure is True
    assert config.authentication_keys is None
    assert config.max_retries is None
    assert config.max_wait_time is None
    assert not config.node_config
    assert config.isolation == ISOLATION_MODE_SUBPROCESS
    assert config.runtime_api_address == SUPERNODE_RUNTIME_API_DEFAULT_SERVER_ADDRESS
    assert config.runtime_certificates is None
    assert config.runtime_root_certificates_path is None
    assert config.health_server_address is None
    assert config.trusted_entities is None
    assert config.superexec_auth_secret is None
    assert config.runtime_dependency_install is False
    assert config.enable_http_api is False
    assert config.host == UVICORN_DEFAULT_HOST
    assert config.port == UVICORN_DEFAULT_PORT
    assert config.runtime_ssl_certfile is None
    assert config.runtime_ssl_keyfile is None


def test_parse_supernode_http_api_args() -> None:
    """SuperNode should parse its temporary Runtime HTTP API arguments."""
    args = _parse_args_run_supernode().parse_args(
        ["--enable-http-api", "--host", "0.0.0.0", "--port", "8080"]
    )

    assert args.enable_http_api is True
    assert args.host == "0.0.0.0"
    assert args.port == 8080


def test_parse_supernode_lifespan_config_preserves_appio_tls_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperNode lifespan config should preserve AppIO-specific TLS args."""
    runtime_certificates = (b"appio-ca", b"appio-cert", b"appio-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flower-supernode",
            "--insecure",
            "--appio-ssl-certfile",
            "appio-cert.pem",
            "--appio-ssl-keyfile",
            "appio-key.pem",
            "--appio-ssl-ca-certfile",
            "appio-ca.pem",
        ],
    )
    monkeypatch.setattr(
        flower_supernode_module,
        "try_obtain_optional_runtime_server_certificates",
        Mock(return_value=runtime_certificates),
    )

    config = _parse_supernode_lifespan_config()

    assert config.runtime_certificates == runtime_certificates
    assert config.runtime_root_certificates_path == "appio-ca.pem"
    assert config.runtime_ssl_certfile == "appio-cert.pem"
    assert config.runtime_ssl_keyfile == "appio-key.pem"


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


def test_flower_supernode_injects_state_factory_and_manages_http_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SuperNode should share state and stop its background HTTP API."""
    config = Mock()
    config.enable_http_api = True
    objectstore_factory = Mock()
    state_factory = Mock()
    start_client_internal = Mock()
    http_server = Mock()
    http_thread = Mock()
    start_http_api = Mock(return_value=(http_server, http_thread))

    monkeypatch.setattr(
        flower_supernode_module, "warn_if_flwr_update_available", Mock()
    )
    monkeypatch.setattr(flower_supernode_module, "event", Mock())
    monkeypatch.setattr(
        flower_supernode_module,
        "_parse_supernode_lifespan_config",
        Mock(return_value=config),
    )
    monkeypatch.setattr(
        flower_supernode_module,
        "ObjectStoreFactory",
        Mock(return_value=objectstore_factory),
    )
    node_state_factory = Mock(return_value=state_factory)
    monkeypatch.setattr(flower_supernode_module, "NodeStateFactory", node_state_factory)
    monkeypatch.setattr(
        flower_supernode_module, "start_client_internal", start_client_internal
    )
    monkeypatch.setattr(
        flower_supernode_module, "_start_supernode_http_api", start_http_api
    )

    flower_supernode_module.flower_supernode()

    node_state_factory.assert_called_once_with(objectstore_factory=objectstore_factory)
    assert start_client_internal.call_args.kwargs["state_factory"] is state_factory
    start_http_api.assert_called_once_with(config, state_factory)
    assert http_server.should_exit is True
    http_thread.join.assert_called_once_with()
