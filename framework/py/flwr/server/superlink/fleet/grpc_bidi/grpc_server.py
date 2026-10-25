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
"""Implements utility function to create a gRPC server."""


import grpc

from flwr.proto.transport_pb2_grpc import (  # pylint: disable=E0611
    add_FlowerServiceServicer_to_server,
)
from flwr.server.client_manager import ClientManager
from flwr.server.superlink.fleet.grpc_bidi.flower_service_servicer import (
    FlowerServiceServicer,
)
from flwr.supercore.grpc import GRPC_MAX_MESSAGE_LENGTH, generic_create_grpc_server


def start_grpc_server(  # pylint: disable=too-many-arguments,R0917
    client_manager: ClientManager,
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    keepalive_time_ms: int = 210000,
    certificates: tuple[bytes, bytes, bytes] | None = None,
) -> grpc.Server:
    """Create and start a gRPC server running FlowerServiceServicer.

    If used in a main function server.wait_for_termination(timeout=None)
    should be called as otherwise the server will immediately stop.

    **SSL**
    To enable SSL you have to pass all of root_certificate, certificate,
    and private_key. Setting only some will make the process exit with code 1.

    Parameters
    ----------
    client_manager : ClientManager
        Instance of ClientManager
    server_address : str
        Server address in the form of HOST:PORT e.g. "[::]:8080"
    max_concurrent_workers : int
        Maximum number of clients the server can process before returning
        RESOURCE_EXHAUSTED status (default: 1000)
    max_message_length : int
        Maximum message length that the server can send or receive.
        Int valued in bytes. -1 means unlimited. (default: GRPC_MAX_MESSAGE_LENGTH)
    keepalive_time_ms : int
        Flower uses a default gRPC keepalive time of 210000ms (3 minutes 30 seconds)
        because some cloud providers (for example, Azure) agressively clean up idle
        TCP connections by terminating them after some time (4 minutes in the case
        of Azure). Flower does not use application-level keepalive signals and relies
        on the assumption that the transport layer will fail in cases where the
        connection is no longer active. `keepalive_time_ms` can be used to customize
        the keepalive interval for specific environments. The default Flower gRPC
        keepalive of 210000 ms (3 minutes 30 seconds) ensures that Flower can keep
        the long running streaming connection alive in most environments. The actual
        gRPC default of this setting is 7200000 (2 hours), which results in dropped
        connections in some cloud environments.

        These settings are related to the issue described here:
        - https://github.com/grpc/proposal/blob/master/A8-client-side-keepalive.md
        - https://github.com/grpc/grpc/blob/master/doc/keepalive.md
        - https://grpc.io/docs/guides/performance/

        Mobile Flower clients may choose to increase this value if their server
        environment allows long-running idle TCP connections.
        (default: 210000)
    certificates : Tuple[bytes, bytes, bytes] (default: None)
        Tuple containing root certificate, server certificate, and private key to
        start a secure SSL-enabled server. The tuple is expected to have three bytes
        elements in the following order:

            * CA certificate.
            * server certificate.
            * server private key.

    Returns
    -------
    server : grpc.Server
        An instance of a gRPC server which is already started

    Examples
    --------
    Starting a TLS-enabled server::

        from pathlib import Path
        start_grpc_server(
            client_manager=ClientManager(),
            server_address="localhost:8080",
            certificates=(
                Path("/crts/root.pem").read_bytes(),
                Path("/crts/localhost.crt").read_bytes(),
                Path("/crts/localhost.key").read_bytes(),
            ),
        )
    """
    servicer = FlowerServiceServicer(client_manager)
    add_servicer_to_server_fn = add_FlowerServiceServicer_to_server

    server = generic_create_grpc_server(
        servicer_and_add_fn=(servicer, add_servicer_to_server_fn),
        server_address=server_address,
        max_concurrent_workers=max_concurrent_workers,
        max_message_length=max_message_length,
        keepalive_time_ms=keepalive_time_ms,
        certificates=certificates,
    )

    server.start()

    return server
