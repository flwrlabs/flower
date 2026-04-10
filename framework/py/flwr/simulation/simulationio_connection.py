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
"""Flower simulation connection compatibility helper."""


from logging import DEBUG, WARNING
from typing import cast

import grpc

from flwr.common.constant import SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.retry_invoker import make_simple_grpc_retry_invoker, wrap_stub
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611
from flwr.supercore.interceptors import AppIoTokenClientInterceptor


class SimulationIoConnection:
    """`SimulationIoConnection` provides an interface to the ServerAppIo API.

    Parameters
    ----------
    serverappio_api_address : str (default: "127.0.0.1:9091")
        The address (URL, IPv6, IPv4) of the SuperLink ServerAppIo API service.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    token : str
        Executor token attached to all outgoing RPCs via metadata.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        serverappio_api_address: str = SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        root_certificates: bytes | None = None,
        *,
        token: str,
    ) -> None:
        if token == "":
            raise ValueError("`token` must be a non-empty string")
        self._addr = serverappio_api_address
        self._cert = root_certificates
        self._token = token
        self._grpc_stub: ServerAppIoStub | None = None
        self._channel: grpc.Channel | None = None
        self._retry_invoker = make_simple_grpc_retry_invoker()

    @property
    def _is_connected(self) -> bool:
        """Check if connected to the ServerAppIo API server."""
        return self._channel is not None

    @property
    def _stub(self) -> ServerAppIoStub:
        """ServerAppIo stub."""
        if not self._is_connected:
            self._connect()
        return cast(ServerAppIoStub, self._grpc_stub)

    def _connect(self) -> None:
        """Connect to the ServerAppIo API."""
        if self._is_connected:
            log(WARNING, "Already connected")
            return
        self._channel = create_channel(
            server_address=self._addr,
            insecure=(self._cert is None),
            root_certificates=self._cert,
            interceptors=[AppIoTokenClientInterceptor(token=self._token)],
        )
        self._channel.subscribe(on_channel_state_change)
        self._grpc_stub = ServerAppIoStub(self._channel)
        wrap_stub(self._grpc_stub, self._retry_invoker)
        log(DEBUG, "[ServerAppIO] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the ServerAppIo API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        channel: grpc.Channel = self._channel
        self._channel = None
        self._grpc_stub = None
        channel.close()
        log(DEBUG, "[ServerAppIO] Disconnected")
