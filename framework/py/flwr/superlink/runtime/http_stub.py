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
"""SuperLink protobuf-over-HTTP Runtime API client."""

from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.runtime import RuntimeHttpStub as CoreRuntimeHttpStub


class RuntimeHttpStub(CoreRuntimeHttpStub):
    """Protobuf-over-HTTP client for the SuperLink Runtime API."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        base_url: str,
        *,
        task_token: str,
        superexec_auth_secret: bytes | None = None,
        verify: bool | str = True,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(
            base_url,
            superexec_auth_secret=superexec_auth_secret,
            verify=verify,
            timeout=timeout,
        )
        self._task_token = task_token

    def GetNodes(  # pylint: disable=invalid-name
        self, request: GetNodesRequest
    ) -> GetNodesResponse:
        """Get available nodes."""
        response = self._session.post(
            f"{self._base_url}/v1/runtime/get-nodes",
            data=request.SerializeToString(deterministic=True),
            headers={
                "content-type": PROTOBUF_MEDIA_TYPE,
                TASK_TOKEN_HEADER: self._task_token,
            },
            verify=self._verify,
            timeout=self._timeout,
        )
        response.raise_for_status()

        result = GetNodesResponse()
        result.ParseFromString(response.content)
        return result
