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
"""SuperExec shared-secret auth helpers."""


from __future__ import annotations

import hashlib
import hmac
from typing import Protocol

from google.protobuf.message import Message as ProtoMessage

from flwr.supercore.constant import (
    MAX_TIMESTAMP_DIFF_SECONDS,
    MIN_TIMESTAMP_DIFF_SECONDS,
    SUPEREXEC_AUTH_SECRET_CONTEXT,
)
from flwr.supercore.date import now


class _NonceStore(Protocol):
    """Store used to reserve SuperExec authentication nonces."""

    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Atomically reserve a nonce until it expires."""


def canonicalize_superexec_auth_input(  # pylint: disable=R0913
    *,
    method: str,
    timestamp: int,
    nonce: str,
    body_sha256: str,
) -> bytes:
    """Serialize SuperExec auth fields to canonical bytes for HMAC input."""
    canonical = (
        f"method={method}\n"
        f"ts={timestamp}\n"
        f"nonce={nonce}\n"
        f"body_sha256={body_sha256}"
    )
    return canonical.encode("utf-8")


def compute_request_body_sha256(request: ProtoMessage) -> str:
    """Compute SHA256 of the deterministic protobuf request body."""
    payload = request.SerializeToString(deterministic=True)
    return hashlib.sha256(payload).hexdigest()


def derive_auth_secret(master_secret: bytes) -> bytes:
    """Derive an auth-scope secret from the master secret."""
    return hmac.new(
        master_secret, SUPEREXEC_AUTH_SECRET_CONTEXT, hashlib.sha256
    ).digest()


def compute_superexec_signature(  # pylint: disable=R0913
    *,
    auth_secret: bytes,
    method: str,
    timestamp: int,
    nonce: str,
    body_sha256: str,
) -> str:
    """Compute SuperExec HMAC-SHA256 signature as a lowercase hex string."""
    canonical = canonicalize_superexec_auth_input(
        method=method,
        timestamp=timestamp,
        nonce=nonce,
        body_sha256=body_sha256,
    )
    return hmac.new(auth_secret, canonical, hashlib.sha256).hexdigest()


def verify_superexec_signature(
    expected_signature: str, received_signature: str
) -> bool:
    """Verify signatures with constant-time comparison."""
    return hmac.compare_digest(expected_signature, received_signature)


def verify_superexec_request(  # pylint: disable=too-many-arguments
    *,
    request: ProtoMessage,
    method: str,
    auth_secret: bytes,
    timestamp_raw: str,
    nonce: str,
    body_sha256_header: str,
    signature: str,
    nonce_store: _NonceStore,
) -> bool:
    """Verify SuperExec authentication fields for a protobuf request."""
    # Require an integer timestamp within the accepted clock-skew and age window.
    try:
        timestamp = int(timestamp_raw)
    except ValueError:
        return False

    time_diff = now().timestamp() - timestamp
    if not MIN_TIMESTAMP_DIFF_SECONDS < time_diff < MAX_TIMESTAMP_DIFF_SECONDS:
        return False

    # Bind the authentication metadata to the deterministic protobuf payload.
    body_sha256 = compute_request_body_sha256(request)
    if body_sha256 != body_sha256_header:
        return False

    # Recompute and compare the HMAC without leaking timing information.
    expected_signature = compute_superexec_signature(
        auth_secret=auth_secret,
        method=method,
        timestamp=timestamp,
        nonce=nonce,
        body_sha256=body_sha256,
    )
    if not verify_superexec_signature(expected_signature, signature):
        return False

    # Reserve the nonce last so failed requests cannot consume nonce entries.
    return nonce_store.reserve_nonce(
        namespace=f"superexec:{method}",
        nonce=nonce,
        expires_at=float(timestamp + MAX_TIMESTAMP_DIFF_SECONDS),
    )
