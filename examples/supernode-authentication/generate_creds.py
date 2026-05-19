import argparse
import datetime
import ipaddress
import os
import shutil
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.x509.oid import NameOID

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Certificate Validity
VALIDITY_DAYS = 365
KEY_SIZE = 4096

# CA Certificate Details
CA_COUNTRY = "DE"
CA_STATE = "HH"
CA_ORGANIZATION = "CA, Inc."

# Server Certificate Details
SERVER_COUNTRY = "DE"
SERVER_STATE = "HH"
SERVER_ORGANIZATION = "Flower"
SERVER_COMMON_NAME = "localhost"

# Subject Alternative Names (SANs)
SERVER_SAN_DNS = ["localhost"]
SERVER_SAN_IPS = [
    "127.0.0.1",
    "::1",
    # "xy.xy.xy.xy",  # Add your server's public IP here
]

# Output Directories
CERT_DIR = Path("certificates")
KEY_DIR = Path("keys")

# --------------------------------------------------------------------------


def generate_ca() -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Generate a self-signed CA certificate and private key."""
    # Generate Private Key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=KEY_SIZE)

    # Generate Name
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, CA_COUNTRY),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, CA_STATE),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, CA_ORGANIZATION),
        ]
    )

    # Generate Certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=VALIDITY_DAYS)
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(private_key, hashes.SHA256())
    )

    return private_key, cert


def generate_server_cert(
    ca_key: rsa.RSAPrivateKey, ca_cert: x509.Certificate
) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
    """Generate a server certificate signed by the CA."""
    # Generate Private Key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=KEY_SIZE)

    # Generate Name
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, SERVER_COUNTRY),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, SERVER_STATE),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, SERVER_ORGANIZATION),
            x509.NameAttribute(NameOID.COMMON_NAME, SERVER_COMMON_NAME),
        ]
    )

    # SANs (Subject Alternative Names)
    alt_names = [x509.DNSName(dns) for dns in SERVER_SAN_DNS] + [
        x509.IPAddress(ipaddress.ip_address(ip)) for ip in SERVER_SAN_IPS
    ]

    # Generate Certificate
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(days=VALIDITY_DAYS)
        )
        .add_extension(
            x509.SubjectAlternativeName(alt_names),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return private_key, cert


def generate_supernode_keys(num_supernodes: int) -> None:
    """Generate generic ECDSA key pairs for supernodes."""
    KEY_DIR.mkdir(exist_ok=True)

    # Clear existing keys
    for item in KEY_DIR.glob("*"):
        if item.is_file():
            item.unlink()

    print(f"Generating keys for {num_supernodes} supernodes...")

    for i in range(1, num_supernodes + 1):
        # Generate ECDSA key (P-384 to match -b 384)
        private_key = ec.generate_private_key(ec.SECP384R1())

        # Serialize Private Key (OpenSSH format)
        # Note: The original script used ssh-keygen Use OpenSSH format for both
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Serialize Public Key (OpenSSH format)
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.OpenSSH,
            format=serialization.PublicFormat.OpenSSH,
        )

        # Write files
        priv_path = KEY_DIR / f"supernode_credentials_{i}"
        pub_path = KEY_DIR / f"supernode_credentials_{i}.pub"

        # Write private key
        with open(priv_path, "wb") as f:
            f.write(private_bytes)
        # Set permissions for private key (600) - mainly for Unix, but good practice
        try:
            os.chmod(priv_path, 0o600)
        except Exception:
            pass

        # Write public key
        with open(pub_path, "wb") as f:
            f.write(public_bytes)

        print(f"  - SuperNode {i}: {priv_path}, {pub_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Flower credentials")
    parser.add_argument(
        "--supernodes",
        type=int,
        default=2,
        help="Number of SuperNode key pairs to generate",
    )
    args = parser.parse_args()

    # --- TLS Certificates ---
    CERT_DIR.mkdir(exist_ok=True)
    # Clear existing certificates
    for item in CERT_DIR.glob("*"):
        if item.is_file():
            item.unlink()

    print("Generating CA certificate...")
    ca_key, ca_cert = generate_ca()

    # Save CA
    with open(CERT_DIR / "ca.key", "wb") as f:
        f.write(
            ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(CERT_DIR / "ca.crt", "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    print("Generating Server certificate...")
    server_key, server_cert = generate_server_cert(ca_key, ca_cert)

    # Save Server
    with open(CERT_DIR / "server.key", "wb") as f:
        f.write(
            server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(CERT_DIR / "server.pem", "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificates generated in {CERT_DIR}/")

    # --- SuperNode Keys ---
    generate_supernode_keys(args.supernodes)
    print("Done.")


if __name__ == "__main__":
    main()
