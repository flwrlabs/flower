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

from flwr.common.constant import SUPERLINK_RUNTIME_API_DEFAULT_CLIENT_ADDRESS
from flwr.supercore import log
from flwr.supercore.interceptors import (
    RuntimeTokenHttpInterceptor,
    RuntimeVersionHttpInterceptor,
)
from flwr.supercore.retry import make_simple_http_retry_invoker
from flwr.supercore.runtime import RuntimeHttpClient


class SimulationIoConnection:
    """`SimulationIoConnection` provides an interface to the Runtime API.

    Parameters
    ----------
    runtime_api_address : str (default: "127.0.0.1:8000")
        The address (URL, IPv6, IPv4) of the SuperLink Runtime API service.
    insecure : bool (default: False)
        If True, use plaintext (TLS disabled). If False, use TLS.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        Used only when `insecure` is False. If provided, these certificates are
        used to verify the server certificate. If None, system root
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
        self._client: RuntimeHttpClient | None = None
        self._retry_invoker = make_simple_http_retry_invoker()

    @property
    def _is_connected(self) -> bool:
        """Check if connected to the Runtime API server."""
        return self._client is not None

    @property
    def _stub(self) -> RuntimeHttpClient:
        """Runtime API client."""
        if not self._is_connected:
            self._connect()
        return cast(RuntimeHttpClient, self._client)

    def _connect(self) -> None:
        """Connect to the Runtime API."""
        if self._is_connected:
            log(WARNING, "Already connected")
            return
        self._client = RuntimeHttpClient.from_server_address(
            server_address=self._addr,
            insecure=self._insecure,
            root_certificates=self._cert,
            interceptors=[
                RuntimeVersionHttpInterceptor(component_name="flwr-simulation"),
                RuntimeTokenHttpInterceptor(token=self._token),
            ],
            retry_invoker=self._retry_invoker,
        )
        log(DEBUG, "[Runtime] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the Runtime API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        client = cast(RuntimeHttpClient, self._client)
        self._client = None
        client.close()
        log(DEBUG, "[Runtime] Disconnected")
