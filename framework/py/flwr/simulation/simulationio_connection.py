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

from flwr.common.constant import SUPERLINK_RUNTIME_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.logger import log
from flwr.proto.runtime_pb2_grpc import RuntimeStub  # pylint: disable=E0611
from flwr.supercore.grpc import create_channel, on_channel_state_change
from flwr.supercore.interceptors import (
    RuntimeTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)
from flwr.supercore.retry import make_simple_grpc_retry_invoker, wrap_stub


class SimulationIoConnection:
    """`SimulationIoConnection` provides an interface to the Runtime API.

    Parameters
    ----------
    runtime_api_address : str (default: "127.0.0.1:9091")
        The address (URL, IPv6, IPv4) of the SuperLink Runtime API service.
    insecure : bool (default: False)
        If True, use plaintext (TLS disabled). If False, use TLS.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        Used only when `insecure` is False. If provided, these certificates are
        used to verify the server certificate. If None, gRPC default root
        certificates are used.
    token : str
        Executor token attached to all outgoing RPCs via metadata.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        runtime_api_address: str = SUPERLINK_RUNTIME_API_DEFAULT_CLIENT_ADDRESS,
        insecure: bool = False,
        root_certificates: bytes | None = None,
        *,
        token: str,
    ) -> None:
        if token == "":
            raise ValueError("`token` must be a non-empty string")
        self._addr = runtime_api_address
        self._insecure = insecure
        self._cert = root_certificates
        self._token = token
        self._grpc_stub: RuntimeStub | None = None
        self._channel: grpc.Channel | None = None
        self._retry_invoker = make_simple_grpc_retry_invoker()

    @property
    def _is_connected(self) -> bool:
        """Check if connected to the Runtime API server."""
        return self._channel is not None

    @property
    def _stub(self) -> RuntimeStub:
        """Runtime API stub."""
        if not self._is_connected:
            self._connect()
        return cast(RuntimeStub, self._grpc_stub)

    def _connect(self) -> None:
        """Connect to the Runtime API."""
        if self._is_connected:
            log(WARNING, "Already connected")
            return
        self._channel = create_channel(
            server_address=self._addr,
            insecure=self._insecure,
            root_certificates=self._cert,
            interceptors=[
                RuntimeVersionClientInterceptor(component_name="flwr-simulation"),
                RuntimeTokenClientInterceptor(token=self._token),
            ],
        )
        self._channel.subscribe(on_channel_state_change)
        self._grpc_stub = RuntimeStub(self._channel)
        wrap_stub(self._grpc_stub, self._retry_invoker)
        log(DEBUG, "[Runtime] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the Runtime API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        channel: grpc.Channel = self._channel
        self._channel = None
        self._grpc_stub = None
        channel.close()
        log(DEBUG, "[Runtime] Disconnected")
