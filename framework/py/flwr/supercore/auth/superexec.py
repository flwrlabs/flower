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
from typing import Any

from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import SYSTEM_TIME_TOLERANCE, TIMESTAMP_TOLERANCE
from flwr.supercore.address import parse_address

SUPEREXEC_AUTH_AUDIENCE_HEADER = "flwr-superexec-audience"
SUPEREXEC_AUTH_TIMESTAMP_HEADER = "flwr-superexec-ts"
SUPEREXEC_AUTH_NONCE_HEADER = "flwr-superexec-nonce"
SUPEREXEC_AUTH_BODY_SHA256_HEADER = "flwr-superexec-body-sha256"
SUPEREXEC_AUTH_SIGNATURE_HEADER = "flwr-superexec-signature"

SUPEREXEC_AUTH_SECRET_CONTEXT = b"superexec-auth-v1"

MIN_TIMESTAMP_DIFF_SECONDS = -SYSTEM_TIME_TOLERANCE
MAX_TIMESTAMP_DIFF_SECONDS = TIMESTAMP_TOLERANCE + SYSTEM_TIME_TOLERANCE


def derive_superexec_audience(service_kind: str, address: str) -> str:
    """Derive a canonical audience string in `<service-kind>:<port>` form."""
    parsed = parse_address(address)
    if parsed is None:
        raise ValueError(f"Cannot parse address: {address}")
    _, port, _ = parsed
    if port <= 0:
        raise ValueError(f"Invalid audience port in address: {address}")
    return f"{service_kind}:{port}"


def canonicalize_superexec_auth_input(  # pylint: disable=R0913
    *,
    method: str,
    audience: str,
    timestamp: int,
    nonce: str,
    body_sha256: str,
) -> bytes:
    """Serialize SuperExec auth fields to canonical bytes for HMAC input."""
    canonical = (
        f"method={method}\n"
        f"audience={audience}\n"
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
    audience: str,
    timestamp: int,
    nonce: str,
    body_sha256: str,
) -> str:
    """Compute SuperExec HMAC-SHA256 signature as a lowercase hex string."""
    canonical = canonicalize_superexec_auth_input(
        method=method,
        audience=audience,
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


def extract_single_str_metadata(
    metadata: list[tuple[str, str | bytes]] | tuple[tuple[str, str | bytes], ...],
    key: str,
) -> str | None:
    """Return exactly one non-empty string metadata value for `key`."""
    values: list[Any] = [
        value for metadata_key, value in metadata if metadata_key == key
    ]
    if len(values) != 1:
        return None
    value = values[0]
    if not isinstance(value, str) or value == "":
        return None
    return value
