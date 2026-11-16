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
"""Symmetric encryption."""


import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def generate_shared_key(
    private_key: ec.EllipticCurvePrivateKey, public_key: ec.EllipticCurvePublicKey
) -> bytes:
    """Generate a shared key from a private key (i.e., a secret key) and a public key.

    Generate shared key by exchange function and key derivation function Key derivation
    function is needed to obtain final shared key of exactly 32 bytes
    """
    # Generate a 32 byte urlsafe(for fernet) shared key
    # from own private key and another public key
    shared_key = private_key.exchange(ec.ECDH(), public_key)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=None,
    ).derive(shared_key)
    return base64.urlsafe_b64encode(derived_key)


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt plaintext using 32-byte key with Fernet."""
    # The input key must be url safe
    fernet = Fernet(key)
    return fernet.encrypt(plaintext)


def decrypt(key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt ciphertext using 32-byte key with Fernet."""
    # The input key must be url safe
    fernet = Fernet(key)
    return fernet.decrypt(ciphertext)
