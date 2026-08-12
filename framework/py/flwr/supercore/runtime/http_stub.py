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
"""Protobuf-over-HTTP Runtime API client."""

import secrets
from typing import TypeVar

import requests
from google.protobuf.message import Message

from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    ClaimTaskResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
)
from flwr.supercore.auth import (
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
)
from flwr.supercore.constant import (
    SUPEREXEC_AUTH_BODY_SHA256_HEADER,
    SUPEREXEC_AUTH_NONCE_HEADER,
    SUPEREXEC_AUTH_SIGNATURE_HEADER,
    SUPEREXEC_AUTH_TIMESTAMP_HEADER,
)
from flwr.supercore.date import now
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE

ResponseT = TypeVar("ResponseT", bound=Message)


class RuntimeHttpStub:
    """Protobuf-over-HTTP implementation of the Runtime API client."""

    def __init__(
        self,
        base_url: str,
        *,
        superexec_auth_secret: bytes | None = None,
        verify: bool | str = True,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth_secret = (
            derive_auth_secret(superexec_auth_secret)
            if superexec_auth_secret is not None
            else None
        )
        self._verify = verify
        self._timeout = timeout
        self._session = requests.Session()

    def PullPendingTasks(  # pylint: disable=invalid-name
        self, request: PullPendingTasksRequest
    ) -> PullPendingTasksResponse:
        """Pull pending tasks."""
        return self._post(
            path="/v1/runtime/pull-pending-tasks",
            method="/flwr.proto.Runtime/PullPendingTasks",
            request=request,
            response_type=PullPendingTasksResponse,
        )

    def ClaimTask(  # pylint: disable=invalid-name
        self, request: ClaimTaskRequest
    ) -> ClaimTaskResponse:
        """Claim a pending task."""
        return self._post(
            path="/v1/runtime/claim-task",
            method="/flwr.proto.Runtime/ClaimTask",
            request=request,
            response_type=ClaimTaskResponse,
        )

    def GetRun(  # pylint: disable=invalid-name
        self, request: GetRunRequest
    ) -> GetRunResponse:
        """Get run information."""
        return self._post(
            path="/v1/runtime/get-run",
            method="/flwr.proto.Runtime/GetRun",
            request=request,
            response_type=GetRunResponse,
        )

    def _post(
        self,
        *,
        path: str,
        method: str,
        request: Message,
        response_type: type[ResponseT],
    ) -> ResponseT:
        """Send a SuperExec-authenticated protobuf request over HTTP."""
        headers = {"content-type": PROTOBUF_MEDIA_TYPE}
        if self._auth_secret is not None:
            timestamp = int(now().timestamp())
            nonce = secrets.token_hex(16)
            body_sha256 = compute_request_body_sha256(request)
            headers.update(
                {
                    SUPEREXEC_AUTH_TIMESTAMP_HEADER: str(timestamp),
                    SUPEREXEC_AUTH_NONCE_HEADER: nonce,
                    SUPEREXEC_AUTH_BODY_SHA256_HEADER: body_sha256,
                    SUPEREXEC_AUTH_SIGNATURE_HEADER: compute_superexec_signature(
                        auth_secret=self._auth_secret,
                        method=method,
                        timestamp=timestamp,
                        nonce=nonce,
                        body_sha256=body_sha256,
                    ),
                }
            )

        response = self._session.post(
            f"{self._base_url}{path}",
            data=request.SerializeToString(deterministic=True),
            headers=headers,
            verify=self._verify,
            timeout=self._timeout,
        )
        response.raise_for_status()

        result = response_type()
        result.ParseFromString(response.content)
        return result

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()
