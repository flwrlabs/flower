# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for module server."""


from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from flwr.common.grpc import valid_certificates
from flwr.server.client_manager import SimpleClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server


def _generate_test_certificates() -> tuple[bytes, bytes, bytes]:
    """Create in-memory test certificates for a TLS-enabled gRPC server."""
    now = datetime.now(timezone.utc)

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(server_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=7))
        .sign(private_key=server_key, algorithm=hashes.SHA256())
    )

    return (
        server_cert.public_bytes(serialization.Encoding.PEM),
        server_cert.public_bytes(serialization.Encoding.PEM),
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ),
    )


def test_valid_certificates_when_correct() -> None:
    """Test is validation function works correctly when passed valid list."""
    # Prepare
    certificates = (b"a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)

    # Assert
    assert is_valid


def test_valid_certificates_when_wrong() -> None:
    """Test is validation function works correctly when passed invalid list."""
    # Prepare
    certificates = ("not_a_byte_string", b"a_byte_string", b"a_byte_string")

    # Execute
    is_valid = valid_certificates(certificates)  # type: ignore

    # Assert
    assert not is_valid


def test_integration_start_and_shutdown_insecure_server() -> None:
    """Create server and check if FlowerServiceServicer is returned."""
    # Prepare
    client_manager = SimpleClientManager()

    # Execute
    server = start_grpc_server(client_manager=client_manager, server_address="[::]:0")

    # Teardown
    server.stop(1)


def test_integration_start_and_shutdown_secure_server() -> None:
    """Create server and check if FlowerServiceServicer is returned."""
    # Prepare
    client_manager = SimpleClientManager()

    certificates = _generate_test_certificates()

    # Execute
    server = start_grpc_server(
        client_manager=client_manager,
        server_address="[::]:0",
        certificates=certificates,
    )

    # Teardown
    server.stop(1)
