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
"""FastAPI SuperExec authentication dependency for Runtime routes."""

from typing import NoReturn, cast

from fastapi import Request

from flwr.supercore.auth import derive_auth_secret, verify_superexec_request
from flwr.supercore.constant import (
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.corestate import CoreState
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.translation import get_protobuf_request


def _raise_authentication_failed(method: str) -> NoReturn:
    """Raise a Runtime authentication error for a SuperExec method."""
    raise FlowerError(
        ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED,
        f"SuperExec authentication failed for {method}.",
    )


def authenticate_superexec_request(
    request: Request,
    state: CoreState,
    method: str,
) -> None:
    """Validate SuperExec authentication headers when authentication is enabled."""
    master_secret = cast(
        bytes | None,
        getattr(request.app.state, "superexec_auth_secret", None),
    )
    if master_secret is None:
        return

    header_values = [
        request.headers.getlist(name)
        for name in (
            SUPEREXEC_AUTH_TIMESTAMP_HEADER,
            SUPEREXEC_AUTH_NONCE_HEADER,
            SUPEREXEC_AUTH_BODY_SHA256_HEADER,
            SUPEREXEC_AUTH_SIGNATURE_HEADER,
        )
    ]
    if any(len(values) != 1 or not values[0] for values in header_values):
        _raise_authentication_failed(method)

    timestamp_raw, nonce, body_sha256_header, signature = (
        values[0] for values in header_values
    )
    authenticated = verify_superexec_request(
        request=get_protobuf_request(request),
        method=method,
        auth_secret=derive_auth_secret(master_secret),
        timestamp_raw=timestamp_raw,
        nonce=nonce,
        body_sha256_header=body_sha256_header,
        signature=signature,
        nonce_store=state,
    )
    if not authenticated:
        _raise_authentication_failed(method)
