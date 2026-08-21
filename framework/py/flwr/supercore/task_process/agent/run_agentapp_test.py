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
"""Tests for the AgentApp process environment."""

import os
import ssl
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from .run_agentapp import _set_runtime_environment


@pytest.mark.parametrize(("insecure", "scheme"), [(True, "http"), (False, "https")])
def test_set_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, insecure: bool, scheme: str
) -> None:
    """Expose the Runtime Responses base URL and AgentApp task token."""
    monkeypatch.delenv("FLWR_RUNTIME_BASE_URL", raising=False)
    monkeypatch.delenv("FLWR_RUNTIME_API_KEY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    certificate_path = _set_runtime_environment(
        "runtime.example:9092", "task-token", insecure=insecure
    )

    assert os.environ["FLWR_RUNTIME_BASE_URL"] == (
        f"{scheme}://runtime.example:9092/v1/runtime"
    )
    assert os.environ["FLWR_RUNTIME_API_KEY"] == "task-token"
    assert certificate_path is None
    assert "SSL_CERT_FILE" not in os.environ


def test_set_runtime_environment_exposes_root_certificates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add custom Runtime root certificates to the public trust bundle."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)

    certificate_path = _set_runtime_environment(
        "runtime.example:9092",
        "task-token",
        insecure=False,
        certificates=b"root-certificates",
    )

    assert certificate_path is not None
    try:
        assert os.environ["SSL_CERT_FILE"] == str(certificate_path)
        certificate_bundle = certificate_path.read_bytes()
        assert certificate_bundle.startswith(b"-----BEGIN CERTIFICATE-----")
        assert certificate_bundle.endswith(b"\nroot-certificates")
    finally:
        certificate_path.unlink(missing_ok=True)


@pytest.mark.parametrize("ca_env", ["SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_set_runtime_environment_preserves_inherited_root_certificates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ca_env: str
) -> None:
    """Keep inherited custom roots when adding the Runtime root certificate."""
    inherited_certificate = httpx.create_ssl_context(trust_env=False).get_ca_certs(
        binary_form=True
    )[0]
    inherited_certificate_pem = ssl.DER_cert_to_PEM_cert(inherited_certificate).encode(
        "ascii"
    )
    if ca_env == "SSL_CERT_FILE":
        inherited_ca_path = tmp_path / "inherited-ca.pem"
        inherited_ca_file = inherited_ca_path
    else:
        inherited_ca_path = tmp_path / "inherited-cas"
        inherited_ca_path.mkdir()
        inherited_ca_file = inherited_ca_path / "inherited-ca.pem"
    inherited_ca_file.write_bytes(inherited_certificate_pem)
    monkeypatch.setenv(ca_env, str(inherited_ca_path))
    monkeypatch.delenv(
        "SSL_CERT_DIR" if ca_env == "SSL_CERT_FILE" else "SSL_CERT_FILE",
        raising=False,
    )
    public_ca_context = Mock()
    public_ca_context.get_ca_certs.return_value = []
    monkeypatch.setattr(httpx, "create_ssl_context", lambda **_: public_ca_context)

    certificate_path = _set_runtime_environment(
        "runtime.example:9092",
        "task-token",
        insecure=False,
        certificates=b"runtime-root-certificate",
    )

    assert certificate_path is not None
    try:
        certificate_bundle = certificate_path.read_bytes()
        assert certificate_bundle.startswith(inherited_certificate_pem)
        assert certificate_bundle.endswith(b"\nruntime-root-certificate")
    finally:
        certificate_path.unlink(missing_ok=True)
