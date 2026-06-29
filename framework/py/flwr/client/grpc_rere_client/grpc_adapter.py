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
"""GrpcAdapter implementation."""


import signal
import time
from logging import DEBUG
from typing import Any, TypeVar, cast

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common import log
from flwr.common.constant import (
    GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_NAME_KEY,
    GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_VERSION_KEY,
    GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY,
    GRPC_ADAPTER_METADATA_MESSAGE_MODULE_KEY,
    GRPC_ADAPTER_METADATA_MESSAGE_QUALNAME_KEY,
    GRPC_ADAPTER_METADATA_SHOULD_EXIT_KEY,
)
from flwr.proto.fab_pb2 import GetFabRequest, GetFabResponse  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    ActivateNodeResponse,
    DeactivateNodeRequest,
    DeactivateNodeResponse,
    PullMessagesRequest,
    PullMessagesResponse,
    PushMessagesRequest,
    PushMessagesResponse,
    RegisterNodeFleetRequest,
    RegisterNodeFleetResponse,
    UnregisterNodeFleetRequest,
    UnregisterNodeFleetResponse,
)
from flwr.proto.grpcadapter_pb2 import MessageContainer  # pylint: disable=E0611
from flwr.proto.grpcadapter_pb2_grpc import GrpcAdapterStub
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.supercore.constant import FORCE_EXIT_TIMEOUT_SECONDS
from flwr.supercore.version import package_name, package_version

T = TypeVar("T", bound=GrpcMessage)


class GrpcAdapter:
    """Adapter class to send and receive gRPC messages via the ``GrpcAdapterStub``.

    This class utilizes the ``GrpcAdapterStub`` to send and receive gRPC messages
    which are defined and used by the Fleet API, as defined in ``fleet.proto``.
    """

    def __init__(self, channel: grpc.Channel) -> None:
        self.stub = GrpcAdapterStub(channel)

    def _send_and_receive(
        self, request: GrpcMessage, response_type: type[T], **kwargs: Any
    ) -> T:
        # Serialize request
        req_cls = request.__class__
        container_req = MessageContainer(
            metadata={
                GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_NAME_KEY: package_name,
                GRPC_ADAPTER_METADATA_FLOWER_PACKAGE_VERSION_KEY: package_version,
                GRPC_ADAPTER_METADATA_FLOWER_VERSION_KEY: package_version,
                GRPC_ADAPTER_METADATA_MESSAGE_MODULE_KEY: req_cls.__module__,
                GRPC_ADAPTER_METADATA_MESSAGE_QUALNAME_KEY: req_cls.__qualname__,
            },
            grpc_message_name=req_cls.__qualname__,
            grpc_message_content=request.SerializeToString(),
        )

        # Send via the stub
        container_res = cast(
            MessageContainer, self.stub.SendReceive(container_req, **kwargs)
        )

        # Handle control message
        should_exit = (
            container_res.metadata.get(GRPC_ADAPTER_METADATA_SHOULD_EXIT_KEY, "false")
            == "true"
        )
        if should_exit:
            log(
                DEBUG,
                'Received shutdown signal: exit flag is set to ``"true"``. Exiting...',
            )
            signal.raise_signal(signal.SIGTERM)
            # Give some time to handle the signal
            time.sleep(FORCE_EXIT_TIMEOUT_SECONDS + 1)

        # Check the grpc_message_name of the response
        if container_res.grpc_message_name != response_type.__qualname__:
            raise ValueError(
                f"Invalid grpc_message_name. Expected {response_type.__qualname__}"
                f", but got {container_res.grpc_message_name}."
            )

        # Deserialize response
        response = response_type()
        response.ParseFromString(container_res.grpc_message_content)
        return response

    def RegisterNode(  # pylint: disable=C0103
        self, request: RegisterNodeFleetRequest, **kwargs: Any
    ) -> RegisterNodeFleetResponse:
        """."""
        return self._send_and_receive(request, RegisterNodeFleetResponse, **kwargs)

    def ActivateNode(  # pylint: disable=C0103
        self, request: ActivateNodeRequest, **kwargs: Any
    ) -> ActivateNodeResponse:
        """."""
        return self._send_and_receive(request, ActivateNodeResponse, **kwargs)

    def DeactivateNode(  # pylint: disable=C0103
        self, request: DeactivateNodeRequest, **kwargs: Any
    ) -> DeactivateNodeResponse:
        """."""
        return self._send_and_receive(request, DeactivateNodeResponse, **kwargs)

    def UnregisterNode(  # pylint: disable=C0103
        self, request: UnregisterNodeFleetRequest, **kwargs: Any
    ) -> UnregisterNodeFleetResponse:
        """."""
        return self._send_and_receive(request, UnregisterNodeFleetResponse, **kwargs)

    def SendNodeHeartbeat(  # pylint: disable=C0103
        self, request: SendNodeHeartbeatRequest, **kwargs: Any
    ) -> SendNodeHeartbeatResponse:
        """."""
        return self._send_and_receive(request, SendNodeHeartbeatResponse, **kwargs)

    def PullMessages(  # pylint: disable=C0103
        self, request: PullMessagesRequest, **kwargs: Any
    ) -> PullMessagesResponse:
        """."""
        return self._send_and_receive(request, PullMessagesResponse, **kwargs)

    def PushMessages(  # pylint: disable=C0103
        self, request: PushMessagesRequest, **kwargs: Any
    ) -> PushMessagesResponse:
        """."""
        return self._send_and_receive(request, PushMessagesResponse, **kwargs)

    def GetRun(  # pylint: disable=C0103
        self, request: GetRunRequest, **kwargs: Any
    ) -> GetRunResponse:
        """."""
        return self._send_and_receive(request, GetRunResponse, **kwargs)

    def GetFab(  # pylint: disable=C0103
        self, request: GetFabRequest, **kwargs: Any
    ) -> GetFabResponse:
        """."""
        return self._send_and_receive(request, GetFabResponse, **kwargs)

    def PushObject(  # pylint: disable=C0103
        self, request: PushObjectRequest, **kwargs: Any
    ) -> PushObjectResponse:
        """."""
        return self._send_and_receive(request, PushObjectResponse, **kwargs)

    def PullObject(  # pylint: disable=C0103
        self, request: PullObjectRequest, **kwargs: Any
    ) -> PullObjectResponse:
        """."""
        return self._send_and_receive(request, PullObjectResponse, **kwargs)

    def ConfirmMessageReceived(  # pylint: disable=C0103
        self, request: ConfirmMessageReceivedRequest, **kwargs: Any
    ) -> ConfirmMessageReceivedResponse:
        """."""
        return self._send_and_receive(request, ConfirmMessageReceivedResponse, **kwargs)
