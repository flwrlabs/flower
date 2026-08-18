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
"""Tests for Runtime HTTP client construction."""

import ssl
from unittest.mock import Mock

import pytest

from .client import create_runtime_http_stub


def test_create_runtime_http_stub_uses_plain_http_when_insecure() -> None:
    """Create an unverified HTTP Runtime client in insecure mode."""
    stub_class = Mock()

    create_runtime_http_stub(
        stub_class=stub_class,
        runtime_api_address="127.0.0.1:8000",
        insecure=True,
        root_certificates=None,
        interceptors=[],
    )

    stub_class.assert_called_once_with(
        "http://127.0.0.1:8000", interceptors=[], verify=False
    )


@pytest.mark.parametrize("root_certificates", [b"certificate", "ca.pem"])
def test_create_runtime_http_stub_rejects_certificates_when_insecure(
    root_certificates: bytes | str,
) -> None:
    """Reject root certificates for a plaintext Runtime connection."""
    with pytest.raises(ValueError, match="root_certificates.*insecure"):
        create_runtime_http_stub(
            stub_class=Mock(),
            runtime_api_address="127.0.0.1:8000",
            insecure=True,
            root_certificates=root_certificates,
            interceptors=[],
        )


def test_create_runtime_http_stub_uses_certificate_path() -> None:
    """Pass a CA certificate path to the HTTP client."""
    stub_class = Mock()

    create_runtime_http_stub(
        stub_class=stub_class,
        runtime_api_address="runtime.example:443",
        insecure=False,
        root_certificates="ca.pem",
        interceptors=[],
    )

    stub_class.assert_called_once_with(
        "https://runtime.example:443", interceptors=[], verify="ca.pem"
    )


def test_create_runtime_http_stub_loads_certificate_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load in-memory CA certificates into an SSL context."""
    stub_class = Mock()
    context = Mock(spec=ssl.SSLContext)
    ssl_context = Mock(return_value=context)
    monkeypatch.setattr(ssl, "SSLContext", ssl_context)

    create_runtime_http_stub(
        stub_class=stub_class,
        runtime_api_address="runtime.example:443",
        insecure=False,
        root_certificates=b"certificate",
        interceptors=[],
    )

    ssl_context.assert_called_once_with(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations.assert_called_once_with(cadata="certificate")
    stub_class.assert_called_once_with(
        "https://runtime.example:443", interceptors=[], verify=context
    )
