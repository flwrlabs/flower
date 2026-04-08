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

from google.protobuf.message import Message as GrpcMessage

from flwr.supercore.constant import SUPEREXEC_AUTH_SECRET_CONTEXT


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


def compute_request_body_sha256(request: GrpcMessage) -> str:
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
