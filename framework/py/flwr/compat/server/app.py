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
"""Flower server app."""


import sys
from logging import INFO

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.constant import FLEET_API_GRPC_BIDI_DEFAULT_ADDRESS
from flwr.supercore.exit import register_signal_handlers
from flwr.common.logger import log, warn_deprecated_feature
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server, init_defaults, run_fl
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import start_grpc_server
from flwr.supercore.address import parse_address


def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    server_address: str = FLEET_API_GRPC_BIDI_DEFAULT_ADDRESS,
    server: Server | None = None,
    config: ServerConfig | None = None,
    strategy: Strategy | None = None,
    client_manager: ClientManager | None = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: tuple[bytes, bytes, bytes] | None = None,
) -> History:
    """Start a Flower server using the gRPC transport layer.

    Warning
    -------
    This function is deprecated since 1.13.0. Use the :code:`flower-superlink` command
    instead to start a SuperLink.

    Parameters
    ----------
    server_address : Optional[str]
        The IPv4 or IPv6 address of the server. Defaults to `"[::]:8080"`.
    server : Optional[flwr.server.Server] (default: None)
        A server implementation, either `flwr.server.Server` or a subclass
        thereof. If no instance is provided, then `start_server` will create
        one.
    config : Optional[ServerConfig] (default: None)
        Currently supported values are `num_rounds` (int, default: 1) and
        `round_timeout` in seconds (float, default: None).
    strategy : Optional[flwr.server.Strategy] (default: None).
        An implementation of the abstract base class
        `flwr.server.strategy.Strategy`. If no strategy is provided, then
        `start_server` will use `flwr.server.strategy.FedAvg`.
    client_manager : Optional[flwr.server.ClientManager] (default: None)
        An implementation of the abstract base class
        `flwr.server.ClientManager`. If no implementation is provided, then
        `start_server` will use
        `flwr.server.client_manager.SimpleClientManager`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower clients. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower clients need to be started with the
        same value (see `flwr.client.start_client`), otherwise clients will
        not know about the increased limit and block larger messages.
    certificates : Tuple[bytes, bytes, bytes] (default: None)
        Tuple containing root certificate, server certificate, and private key
        to start a secure SSL-enabled server. The tuple is expected to have
        three bytes elements in the following order:

            * CA certificate.
            * server certificate.
            * server private key.

    Returns
    -------
    hist : flwr.server.history.History
        Object containing training and evaluation metrics.

    Examples
    --------
    Starting an insecure server::

        start_server()

    Starting a TLS-enabled server::

        start_server(
            certificates=(
                Path("/crts/root.pem").read_bytes(),
                Path("/crts/localhost.crt").read_bytes(),
                Path("/crts/localhost.key").read_bytes()
            )
        )
    """
    msg = (
        "flwr.server.start_server() is deprecated."
        "\n\tInstead, use the `flower-superlink` CLI command to start a SuperLink "
        "as shown below:"
        "\n\n\t\t$ flower-superlink --insecure"
        "\n\n\tTo view usage and all available options, run:"
        "\n\n\t\t$ flower-superlink --help"
        "\n\n\tUsing `start_server()` is deprecated."
    )
    warn_deprecated_feature(name=msg)

    event(EventType.START_SERVER_ENTER)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Initialize server and server config
    initialized_server, initialized_config = init_defaults(
        server=server,
        config=config,
        strategy=strategy,
        client_manager=client_manager,
    )
    log(
        INFO,
        "Starting Flower server, config: %s",
        initialized_config,
    )

    # Start gRPC server
    grpc_server = start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_address=address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        initialized_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Graceful shutdown
    register_signal_handlers(
        event_type=EventType.START_SERVER_LEAVE,
        exit_message="Flower server terminated gracefully.",
        grpc_servers=[grpc_server],
    )

    # Start training
    hist = run_fl(
        server=initialized_server,
        config=initialized_config,
    )

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist
