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
"""Utility functions for gRPC."""


import concurrent.futures
import os
import sys
from collections.abc import Callable, Sequence
from logging import DEBUG, ERROR
from typing import Any

import grpc

from flwr.supercore.address import is_port_in_use

from .logger import log

GRPC_MAX_MESSAGE_LENGTH: int = 2_147_483_647  # == 2048 * 1024 * 1024 -1 (2GB)

INVALID_CERTIFICATES_ERR_MSG = """
    When setting any of root_certificate, certificate, or private_key,
    all of them need to be set.
"""

AddServicerToServerFn = Callable[..., Any]

if "GRPC_VERBOSITY" not in os.environ:
    os.environ["GRPC_VERBOSITY"] = "error"
# The following flags can be uncommented for debugging. Other possible values:
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
# os.environ["GRPC_TRACE"] = "tcp,http"


def create_channel(
    server_address: str,
    insecure: bool,
    root_certificates: bytes | None = None,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    interceptors: Sequence[grpc.UnaryUnaryClientInterceptor] | None = None,
) -> grpc.Channel:
    """Create a gRPC channel, either secure or insecure."""
    # Check for conflicting parameters
    if insecure and root_certificates is not None:
        raise ValueError(
            "Invalid configuration: 'root_certificates' should not be provided "
            "when 'insecure' is set to True. For an insecure connection, omit "
            "'root_certificates', or set 'insecure' to False for a secure connection."
        )

    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    channel_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    if insecure:
        channel = grpc.insecure_channel(server_address, options=channel_options)
        log(DEBUG, "Opened insecure gRPC connection (no certificates were passed)")
    else:
        try:
            ssl_channel_credentials = grpc.ssl_channel_credentials(root_certificates)
        except Exception as e:
            raise ValueError(f"Failed to create SSL channel credentials: {e}") from e
        channel = grpc.secure_channel(
            server_address, ssl_channel_credentials, options=channel_options
        )
        log(DEBUG, "Opened secure gRPC connection using certificates")

    if interceptors is not None:
        channel = grpc.intercept_channel(channel, *interceptors)

    return channel


def valid_certificates(certificates: tuple[bytes, bytes, bytes]) -> bool:
    """Validate certificates tuple."""
    is_valid = (
        all(isinstance(certificate, bytes) for certificate in certificates)
        and len(certificates) == 3
    )

    if not is_valid:
        log(ERROR, INVALID_CERTIFICATES_ERR_MSG)

    return is_valid


def generic_create_grpc_server(  # pylint: disable=too-many-arguments, R0914, R0917
    servicer_and_add_fn: tuple[Any, AddServicerToServerFn],
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    keepalive_time_ms: int = 210000,
    certificates: tuple[bytes, bytes, bytes] | None = None,
    interceptors: Sequence[grpc.ServerInterceptor] | None = None,
) -> grpc.Server:
    """Create a gRPC server with a single servicer.

    Parameters
    ----------
    servicer_and_add_fn : tuple
        A tuple holding a servicer implementation and a matching
        add_Servicer_to_server function.
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
    interceptors : Optional[Sequence[grpc.ServerInterceptor]] (default: None)
        A list of gRPC interceptors.

    Returns
    -------
    server : grpc.Server
        A non-running instance of a gRPC server. The actual used address can be accessed
        via `server.bound_address`.
    """
    # Check if port is in use
    if is_port_in_use(server_address):
        sys.exit(f"Port in server address {server_address} is already in use.")

    # Deconstruct tuple into servicer and function
    servicer, add_servicer_to_server_fn = servicer_and_add_fn

    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    options = [
        # Maximum number of concurrent incoming streams to allow on a http2
        # connection. Int valued.
        ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
        # Maximum message length that the channel can send.
        # Int valued, bytes. -1 means unlimited.
        ("grpc.max_send_message_length", max_message_length),
        # Maximum message length that the channel can receive.
        # Int valued, bytes. -1 means unlimited.
        ("grpc.max_receive_message_length", max_message_length),
        # The gRPC default for this setting is 7200000 (2 hours). Flower uses a
        # customized default of 210000 (3 minutes and 30 seconds) to improve
        # compatibility with popular cloud providers. Mobile Flower clients may
        # choose to increase this value if their server environment allows
        # long-running idle TCP connections.
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        # Setting this to zero will allow sending unlimited keepalive pings in between
        # sending actual data frames.
        ("grpc.http2.max_pings_without_data", 0),
        # Is it permissible to send keepalive pings from the client without
        # any outstanding streams. More explanation here:
        # https://github.com/flwrlabs/flower/pull/2197
        ("grpc.keepalive_permit_without_calls", 0),
    ]

    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers),
        # Set the maximum number of concurrent RPCs this server will service before
        # returning RESOURCE_EXHAUSTED status, or None to indicate no limit.
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=options,
        interceptors=interceptors,
    )
    add_servicer_to_server_fn(servicer, server)

    if certificates is not None:
        if not valid_certificates(certificates):
            sys.exit(1)

        root_certificate_b, certificate_b, private_key_b = certificates

        server_credentials = grpc.ssl_server_credentials(
            ((private_key_b, certificate_b),),
            root_certificates=root_certificate_b,
            # A boolean indicating whether or not to require clients to be
            # authenticated. May only be True if root_certificates is not None.
            # We are explicitly setting the current gRPC default to document
            # the option. For further reference see:
            # https://grpc.github.io/grpc/python/grpc.html#create-server-credentials
            require_client_auth=False,
        )
        port = server.add_secure_port(server_address, server_credentials)
    else:
        port = server.add_insecure_port(server_address)

    # Allow the port to be accessible outside the function
    server.bound_address = f"{server_address.rpartition(':')[0]}:{port}"
    return server


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)
