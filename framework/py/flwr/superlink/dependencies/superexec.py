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

from typing import Annotated, cast

from fastapi import Depends, Request

from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth import derive_auth_secret, verify_superexec_request
from flwr.supercore.constant import (
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.translation import get_protobuf_request

from .linkstate import get_linkstate


class SuperExecAuthDependency:
    """Authenticate one SuperExec Runtime operation."""

    def __init__(self, method: str) -> None:
        self.method = method

    def __call__(
        self,
        request: Request,
        state: Annotated[LinkState, Depends(get_linkstate)],
    ) -> None:
        """Validate SuperExec authentication headers when authentication is enabled."""
        master_secret = cast(
            bytes | None,
            getattr(request.app.state, "superexec_auth_secret", None),
        )
        if master_secret is None:
            return

        authenticated = verify_superexec_request(
            request=get_protobuf_request(request),
            method=self.method,
            auth_secret=derive_auth_secret(master_secret),
            timestamp_raw=request.headers.get(SUPEREXEC_AUTH_TIMESTAMP_HEADER),
            nonce=request.headers.get(SUPEREXEC_AUTH_NONCE_HEADER),
            body_sha256_header=request.headers.get(SUPEREXEC_AUTH_BODY_SHA256_HEADER),
            signature=request.headers.get(SUPEREXEC_AUTH_SIGNATURE_HEADER),
            nonce_store=state,
        )
        if not authenticated:
            raise FlowerError(
                ApiErrorCode.RUNTIME_AUTHENTICATION_FAILED,
                f"SuperExec authentication failed for {self.method}.",
            )
