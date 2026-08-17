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
"""HTTP stub for the SuperLink Runtime API."""

from flwr.proto.log_pb2 import (  # pylint: disable=E0611
    PushLogsRequest,
    PushLogsResponse,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.supercore.runtime import RuntimeHttpStub as CoreRuntimeHttpStub


# Match the method names exposed by the generated gRPC RuntimeStub.
# pylint: disable=invalid-name
class RuntimeHttpStub(CoreRuntimeHttpStub):
    """Protobuf-over-HTTP client for SuperLink Runtime API methods."""

    def PushLogs(self, request: PushLogsRequest) -> PushLogsResponse:
        """Push task logs to the Runtime API."""
        return self._unary_unary(
            path="/v1/runtime/push-logs",
            rpc_method="/flwr.proto.Runtime/PushLogs",
            request=request,
            response_type=PushLogsResponse,
        )

    def GetNodes(self, request: GetNodesRequest) -> GetNodesResponse:
        """Get nodes available to the ServerApp."""
        return self._unary_unary(
            path="/v1/runtime/get-nodes",
            rpc_method="/flwr.proto.Runtime/GetNodes",
            request=request,
            response_type=GetNodesResponse,
        )
