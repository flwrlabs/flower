# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Flower SuperNode connection."""


from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, Message
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.typing import Fab, Run


class FleetConnection(ABC):
    """Abstract base class for SuperNode connections."""

    def __init__(  # pylint: disable=R0913, R0914, R0915, R0917
        self,
        server_address: str,
        insecure: bool,
        retry_invoker: RetryInvoker,
        max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
        root_certificates: bytes | str | None = None,
        authentication_keys: (
            tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
        ) = None,
    ) -> None:
        """Open a connection with the SuperLink.

        Parameters
        ----------
        server_address : str
            The IPv6 address of the server with `http://` or `https://`.
            If the Flower server runs on the same machine
            on port 8080, then `server_address` would be `"http://[::]:8080"`.
        insecure : bool
            Starts an insecure gRPC connection when True. Enables HTTPS connection
            when False, using system certificates if `root_certificates` is None.
        retry_invoker: RetryInvoker
            `RetryInvoker` object that will try to reconnect the client to the server
            after gRPC errors. If None, the client will only try to
            reconnect once after a failure.
        max_message_length : int
            Ignored, only present to preserve API-compatibility.
        root_certificates : Optional[Union[bytes, str]] (default: None)
            Path of the root certificate. If provided, a secure
            connection using the certificates will be established to an SSL-enabled
            Flower server. Bytes won't work for the REST API.
        authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
            Tuple containing the elliptic curve private key and public key for
            authentication from the cryptography library.
            Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
            Used to establish an authenticated connection with the server.
        """
        self.server_address = server_address
        self.insecure = insecure
        self.retry_invoker = retry_invoker
        self.max_message_length = max_message_length
        self.root_certificates = root_certificates
        self.authentication_keys = authentication_keys

    @abstractmethod
    def ping(self) -> None:
        """Ping the SuperLink."""

    @abstractmethod
    def create_node(self) -> int | None:
        """Request to create a node."""

    @abstractmethod
    def delete_node(self) -> None:
        """Request to delete a node."""

    @abstractmethod
    def receive(self) -> Message | None:
        """Receive a message."""

    @abstractmethod
    def send(self, message: Message) -> None:
        """Send a message."""

    @abstractmethod
    def get_run(self, run_id: int) -> Run:
        """Get run info."""

    @abstractmethod
    def get_fab(self, fab_hash: str, run_id: int) -> Fab:
        """Get FAB file."""

    def __enter__(self) -> FleetConnection:
        """Enter the context."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType,
    ) -> None:
        """Exit from the context."""
