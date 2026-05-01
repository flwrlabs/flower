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
"""Tests for TLS helpers in supercore."""


import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from flwr.common.exit import ExitCode

from .tls import (
    get_client_tls_args,
    try_obtain_optional_appio_server_certificates,
    validate_and_resolve_root_certificates,
)


def test_load_root_certificates_returns_none_when_insecure() -> None:
    """The helper should return `None` for insecure connections."""
    assert validate_and_resolve_root_certificates(None, insecure=True) is None


def test_load_root_certificates_reads_file(tmp_path: Path) -> None:
    """The helper should read certificate bytes from disk."""
    cert_path = tmp_path / "root.pem"
    cert_path.write_bytes(b"root-cert")

    assert (
        validate_and_resolve_root_certificates(str(cert_path), insecure=False)
        == b"root-cert"
    )


def test_load_root_certificates_rejects_conflicting_flags() -> None:
    """The helper should reject insecure mode with root certificates."""
    with patch("flwr.supercore.tls.flwr_exit") as mock_exit:
        mock_exit.side_effect = RuntimeError
        with pytest.raises(RuntimeError):
            validate_and_resolve_root_certificates("/tmp/root.pem", insecure=True)

    mock_exit.assert_called_once()
    input_code = mock_exit.call_args.args[0]
    assert input_code == ExitCode.COMMON_TLS_ROOT_CERTIFICATES_INCOMPATIBLE


def test_load_root_certificates_rejects_invalid_path() -> None:
    """The helper should reject invalid certificate paths."""
    with patch("flwr.supercore.tls.flwr_exit") as mock_exit:
        mock_exit.side_effect = RuntimeError
        with pytest.raises(RuntimeError):
            validate_and_resolve_root_certificates(
                "/tmp/missing-root.pem", insecure=False
            )

    mock_exit.assert_called_once()
    input_code = mock_exit.call_args.args[0]
    assert input_code == ExitCode.COMMON_PATH_INVALID


def test_load_root_certificates_returns_none_when_no_path() -> None:
    """The helper should return `None` when no path is provided."""
    assert validate_and_resolve_root_certificates(None, insecure=False) is None


def test_get_client_tls_args_returns_insecure() -> None:
    """Client processes should use plaintext when requested."""
    assert get_client_tls_args(insecure=True, root_certificates_path=None) == [
        "--insecure"
    ]


def test_get_client_tls_args_returns_root_certificates() -> None:
    """Client processes should verify TLS with the configured CA path."""
    assert get_client_tls_args(
        insecure=False, root_certificates_path="/tmp/ca.pem"
    ) == [
        "--root-certificates",
        "/tmp/ca.pem",
    ]


def test_get_client_tls_args_omits_flags_for_system_trust() -> None:
    """Client processes should use system trust roots when no CA path is provided."""
    assert not get_client_tls_args(insecure=False, root_certificates_path=None)


def test_try_obtain_optional_appio_server_certificates_returns_none() -> None:
    """Optional AppIO server certificates should be omitted by default."""
    args = argparse.Namespace(
        appio_ssl_ca_certfile=None,
        appio_ssl_certfile=None,
        appio_ssl_keyfile=None,
    )

    assert try_obtain_optional_appio_server_certificates(args) is None


def test_try_obtain_optional_appio_server_certificates_reads_files(
    tmp_path: Path,
) -> None:
    """Optional AppIO server certificates should be read when all paths are provided."""
    cert_dir = tmp_path
    ca_cert = cert_dir / "ca.pem"
    server_cert = cert_dir / "server.pem"
    server_key = cert_dir / "server.key"
    ca_cert.write_bytes(b"ca")
    server_cert.write_bytes(b"cert")
    server_key.write_bytes(b"key")
    args = argparse.Namespace(
        appio_ssl_ca_certfile=str(ca_cert),
        appio_ssl_certfile=str(server_cert),
        appio_ssl_keyfile=str(server_key),
    )

    certificates = try_obtain_optional_appio_server_certificates(args)

    assert certificates == (b"ca", b"cert", b"key")


def test_try_obtain_optional_appio_server_certificates_rejects_partial_config() -> None:
    """Optional AppIO server certificates should reject partial TLS config."""
    args = argparse.Namespace(
        appio_ssl_ca_certfile="/tmp/ca.pem",
        appio_ssl_certfile=None,
        appio_ssl_keyfile=None,
    )

    with patch("flwr.supercore.tls.flwr_exit") as mock_exit:
        mock_exit.side_effect = RuntimeError
        with pytest.raises(RuntimeError):
            try_obtain_optional_appio_server_certificates(args)

    mock_exit.assert_called_once()
    input_code, message = mock_exit.call_args.args
    assert input_code == ExitCode.COMMON_TLS_SERVER_CERTIFICATES_INVALID
    assert "--appio-ssl-certfile" in message
    assert "--appio-ssl-keyfile" in message
    assert "--appio-ssl-ca-certfile" in message


def test_try_obtain_optional_appio_server_certificates_rejects_invalid_path() -> None:
    """Optional AppIO server certificates should reject invalid paths."""
    args = argparse.Namespace(
        appio_ssl_ca_certfile="/tmp/missing-ca.pem",
        appio_ssl_certfile="/tmp/missing-cert.pem",
        appio_ssl_keyfile="/tmp/missing-key.pem",
    )

    with patch("flwr.supercore.tls.flwr_exit") as mock_exit:
        mock_exit.side_effect = RuntimeError
        with pytest.raises(RuntimeError):
            try_obtain_optional_appio_server_certificates(args)

    mock_exit.assert_called_once()
    input_code, message = mock_exit.call_args.args
    assert input_code == ExitCode.COMMON_PATH_INVALID
    assert "--appio-ssl-ca-certfile" in message
