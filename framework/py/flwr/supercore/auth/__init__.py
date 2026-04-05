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
"""Shared auth policy definitions."""


from .policy import (
    CLIENTAPPIO_METHOD_AUTH_POLICY,
    SERVERAPPIO_METHOD_AUTH_POLICY,
    MethodTokenPolicy,
)
from .superexec import (
    MAX_TIMESTAMP_DIFF_SECONDS,
    MIN_TIMESTAMP_DIFF_SECONDS,
    SUPEREXEC_AUTH_AUDIENCE_HEADER,
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
    derive_superexec_audience,
    extract_single_str_metadata,
    verify_superexec_signature,
)
from .superexec_secret import (
    add_superexec_auth_secret_args,
    generate_superexec_auth_secret,
    load_superexec_auth_secret,
)

__all__ = [
    "CLIENTAPPIO_METHOD_AUTH_POLICY",
    "MAX_TIMESTAMP_DIFF_SECONDS",
    "MIN_TIMESTAMP_DIFF_SECONDS",
    "MethodTokenPolicy",
    "SERVERAPPIO_METHOD_AUTH_POLICY",
    "SUPEREXEC_AUTH_AUDIENCE_HEADER",
    "SUPEREXEC_AUTH_BODY_SHA256_HEADER",
    "SUPEREXEC_AUTH_NONCE_HEADER",
    "SUPEREXEC_AUTH_SIGNATURE_HEADER",
    "SUPEREXEC_AUTH_TIMESTAMP_HEADER",
    "add_superexec_auth_secret_args",
    "compute_request_body_sha256",
    "compute_superexec_signature",
    "derive_auth_secret",
    "derive_superexec_audience",
    "extract_single_str_metadata",
    "generate_superexec_auth_secret",
    "load_superexec_auth_secret",
    "verify_superexec_signature",
]
