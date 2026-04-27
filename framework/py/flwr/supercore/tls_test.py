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


from pathlib import Path
from unittest.mock import patch

import pytest

from flwr.common.exit import ExitCode

from .tls import get_superexec_appio_tls_args, validate_and_resolve_root_certificates


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


def test_get_superexec_appio_tls_args_returns_insecure_without_certificates() -> None:
    """SuperExec should use plaintext when the AppIO server has no TLS certs."""
    assert get_superexec_appio_tls_args(None, None) == ["--insecure"]


def test_get_superexec_appio_tls_args_returns_root_certificates() -> None:
    """SuperExec should verify AppIO TLS with the configured CA path."""
    assert get_superexec_appio_tls_args((b"ca", b"cert", b"key"), "/tmp/ca.pem") == [
        "--root-certificates",
        "/tmp/ca.pem",
    ]


def test_get_superexec_appio_tls_args_omits_flags_for_system_trust() -> None:
    """SuperExec should use system trust roots when no CA path is provided."""
    assert not get_superexec_appio_tls_args((b"ca", b"cert", b"key"), None)
