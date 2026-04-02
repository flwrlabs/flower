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
"""Flower client app."""


import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from logging import ERROR, INFO, WARN

from cryptography.hazmat.primitives.asymmetric import ec
from grpc import RpcError

from flwr.app.error import Error
from flwr.app.user_config import UserConfig
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.client.client import Client
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.client.numpy_client import NumPyClient
from flwr.client.run_info_store import DeprecatedRunInfoStore
from flwr.client.typing import ClientFnExt
from flwr.clientapp.client_app import ClientApp, LoadClientAppError
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, Context, EventType, Message, event
from flwr.common.constant import MAX_RETRY_DELAY, ErrorCode
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log, warn_deprecated_feature
from flwr.common.retry_invoker import RetryInvoker, RetryState, exponential
from flwr.common.typing import Fab, Run, RunNotRunningException
from flwr.compat.client.grpc_client.connection import grpc_connection
from flwr.compat.common.constant import TRANSPORT_TYPE_GRPC_BIDI, TRANSPORT_TYPES_COMPAT
from flwr.supercore.address import parse_address
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supernode.nodestate import NodeStateFactory


def _check_actionable_client(
    client: Client | None, client_fn: ClientFnExt | None
) -> None:
    if client_fn is None and client is None:
        raise ValueError(
            "Both `client_fn` and `client` are `None`, but one is required"
        )

    if client_fn is not None and client is not None:
        raise ValueError(
            "Both `client_fn` and `client` are provided, but only one is allowed"
        )


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
def start_client(
    *,
    server_address: str,
    client_fn: ClientFnExt | None = None,
    client: Client | None = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: bytes | str | None = None,
    insecure: bool | None = None,
    transport: str | None = None,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
    max_retries: int | None = None,
    max_wait_time: float | None = None,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Warning
    -------
    This function is deprecated since 1.13.0. Use :code:`flower-supernode` command
    instead to start a SuperNode.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    client_fn : Optional[ClientFnExt]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : bool (default: True)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        **[Deprecated]** This argument is no longer supported and will be
        removed in a future release.
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Tuple containing the elliptic curve private key and public key for
        authentication from the cryptography library.
        Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
        Used to establish an authenticated connection with the server.
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.

    Examples
    --------
    Starting a gRPC client with an insecure server connection::

        start_client(
            server_address=localhost:8080,
            client_fn=client_fn,
        )

    Starting a TLS-enabled gRPC client using system certificates::

        def client_fn(context: Context):
            return FlowerClient().to_client()

        start_client(
            server_address=localhost:8080,
            client_fn=client_fn,
            insecure=False,
        )

    Starting a TLS-enabled gRPC client using provided certificates::

        from pathlib import Path

        start_client(
            server_address=localhost:8080,
            client_fn=client_fn,
            root_certificates=Path("/crts/root.pem").read_bytes(),
        )
    """
    msg = (
        "flwr.client.start_client() is deprecated."
        "\n\tInstead, use the `flower-supernode` CLI command to start a SuperNode "
        "as shown below:"
        "\n\n\t\t$ flower-supernode --insecure --superlink='<IP>:<PORT>'"
        "\n\n\tTo view all available options, run:"
        "\n\n\t\t$ flower-supernode --help"
        "\n\n\tUsing `start_client()` is deprecated."
    )
    warn_deprecated_feature(name=msg)

    if transport is not None and transport != "grpc-bidi":
        raise ValueError(
            f"Transport type {transport} is not supported. "
            "Use 'grpc-bidi' or None (default) instead."
        )

    event(EventType.START_CLIENT_ENTER)
    start_client_internal(
        server_address=server_address,
        node_config={},
        load_client_app_fn=None,
        client_fn=client_fn,
        client=client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
        authentication_keys=authentication_keys,
        max_retries=max_retries,
        max_wait_time=max_wait_time,
    )
    event(EventType.START_CLIENT_LEAVE)


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
def start_client_internal(
    *,
    server_address: str,
    node_config: UserConfig,
    load_client_app_fn: Callable[[str, str, str], ClientApp] | None = None,
    client_fn: ClientFnExt | None = None,
    client: Client | None = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: bytes | str | None = None,
    insecure: bool | None = None,
    transport: str | None = None,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
    max_retries: int | None = None,
    max_wait_time: float | None = None,
) -> None:
    """Start a Flower client node which connects to a Flower server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower
        server runs on the same machine on port 8080, then `server_address`
        would be `"[::]:8080"`.
    node_config: UserConfig
        The configuration of the node.
    load_client_app_fn : Optional[Callable[[], ClientApp]] (default: None)
        A function that can be used to load a `ClientApp` instance.
    client_fn : Optional[ClientFnExt]
        A callable that instantiates a Client. (default: None)
    client : Optional[flwr.client.Client]
        An implementation of the abstract base
        class `flwr.client.Client` (default: None)
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        Configure the transport layer. Allowed values:
        - 'grpc-bidi': gRPC, bidirectional streaming
        - 'grpc-rere': gRPC, request-response (experimental)
        - 'rest': HTTP (experimental)
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        Tuple containing the elliptic curve private key and public key for
        authentication from the cryptography library.
        Source: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ec/
        Used to establish an authenticated connection with the server.
    max_retries: Optional[int] (default: None)
        The maximum number of times the client will try to connect to the
        server before giving up in case of a connection error. If set to None,
        there is no limit to the number of tries.
    max_wait_time: Optional[float] (default: None)
        The maximum duration before the client stops trying to
        connect to the server in case of connection error.
        If set to None, there is no limit to the total time.
    """
    if insecure is None:
        insecure = root_certificates is None

    if load_client_app_fn is None:
        _check_actionable_client(client, client_fn)

        if client_fn is None:
            # Wrap `Client` instance in `client_fn`
            def single_client_factory(
                context: Context,  # pylint: disable=unused-argument
            ) -> Client:
                if client is None:  # Added this to keep mypy happy
                    raise ValueError(
                        "Both `client_fn` and `client` are `None`, but one is required"
                    )
                return client  # Always return the same instance

            client_fn = single_client_factory

        def _load_client_app(_1: str, _2: str, _3: str) -> ClientApp:
            return ClientApp(client_fn=client_fn)

        load_client_app_fn = _load_client_app

    # At this point, only `load_client_app_fn` should be used
    # Both `client` and `client_fn` must not be used directly

    # Initialize connection context manager
    connection, address, connection_error_type = _init_connection(
        transport, server_address
    )

    def _on_sucess(retry_state: RetryState) -> None:
        if retry_state.tries > 1:
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _on_backoff(retry_state: RetryState) -> None:
        if retry_state.tries == 1:
            log(WARN, "Connection attempt failed, retrying...")
        else:
            log(
                WARN,
                "Connection attempt failed, retrying in %.2f seconds",
                retry_state.actual_wait,
            )

    retry_invoker = RetryInvoker(
        wait_gen_factory=lambda: exponential(max_delay=MAX_RETRY_DELAY),
        recoverable_exceptions=connection_error_type,
        max_tries=max_retries + 1 if max_retries is not None else None,
        max_time=max_wait_time,
        on_giveup=lambda retry_state: (
            log(
                WARN,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )
            if retry_state.tries > 1
            else None
        ),
        on_success=_on_sucess,
        on_backoff=_on_backoff,
    )

    # DeprecatedRunInfoStore gets initialized when the first connection is established
    run_info_store: DeprecatedRunInfoStore | None = None
    state_factory = NodeStateFactory(objectstore_factory=ObjectStoreFactory())
    state = state_factory.state()

    runs: dict[int, Run] = {}

    while True:
        sleep_duration: int = 0
        with connection(
            address,
            insecure,
            retry_invoker,
            grpc_max_message_length,
            root_certificates,
            authentication_keys,
        ) as conn:
            receive, send, create_node, delete_node, get_run, get_fab = conn

            # Register node when connecting the first time
            if run_info_store is None:
                if create_node is None:
                    if transport not in ["grpc-bidi", None]:
                        raise NotImplementedError(
                            "All transports except `grpc-bidi` require "
                            "an implementation for `create_node()`.'"
                        )
                    # gRPC-bidi doesn't have the concept of node_id,
                    # so we set it to -1
                    run_info_store = DeprecatedRunInfoStore(
                        node_id=-1,
                        node_config={},
                    )
                else:
                    # Call create_node fn to register node
                    # and store node_id in state
                    if (node_id := create_node()) is None:
                        raise ValueError(
                            "Failed to register SuperNode with the SuperLink"
                        )
                    state.set_node_id(node_id)
                    run_info_store = DeprecatedRunInfoStore(
                        node_id=state.get_node_id(),
                        node_config=node_config,
                    )

            # pylint: disable=too-many-nested-blocks
            while True:
                try:
                    # Receive
                    message = receive()
                    if message is None:
                        time.sleep(3)  # Wait for 3s before asking again
                        continue

                    log(INFO, "")
                    if len(message.metadata.group_id) > 0:
                        log(
                            INFO,
                            "[RUN %s, ROUND %s]",
                            message.metadata.run_id,
                            message.metadata.group_id,
                        )
                    log(
                        INFO,
                        "Received: %s message %s",
                        message.metadata.message_type,
                        message.metadata.message_id,
                    )

                    # Handle control message
                    out_message, sleep_duration = handle_control_message(message)
                    if out_message:
                        send(out_message)
                        break

                    # Get run info
                    run_id = message.metadata.run_id
                    if run_id not in runs:
                        if get_run is not None:
                            runs[run_id] = get_run(run_id)
                        # If get_run is None, i.e., in grpc-bidi mode
                        else:
                            runs[run_id] = Run.create_empty(run_id=run_id)

                    run: Run = runs[run_id]
                    if get_fab is not None and run.fab_hash:
                        fab = get_fab(run.fab_hash, run_id)  # pylint: disable=E1102
                        # If `ClientApp` runs in the same process, install the FAB
                        install_from_fab(fab.content, skip_prompt=True)
                        fab_id, fab_version = get_fab_metadata(fab.content)
                    else:
                        fab = None
                        fab_id, fab_version = run.fab_id, run.fab_version

                    run.fab_id, run.fab_version = fab_id, fab_version

                    # Register context for this run
                    run_info_store.register_context(
                        run_id=run_id,
                        run=run,
                        fab=fab,
                    )

                    # Retrieve context for this run
                    context = run_info_store.retrieve_context(run_id=run_id)
                    # Create an error reply message that will never be used to prevent
                    # the used-before-assignment linting error
                    reply_message = Message(
                        Error(code=ErrorCode.UNKNOWN, reason="Unknown"),
                        reply_to=message,
                    )

                    # Handle app loading and task message
                    try:
                        # Load ClientApp instance
                        client_app: ClientApp = load_client_app_fn(
                            fab_id, fab_version, run.fab_hash
                        )

                        # Execute ClientApp
                        reply_message = client_app(message=message, context=context)
                    except Exception as ex:  # pylint: disable=broad-exception-caught

                        # Legacy grpc-bidi
                        if transport in ["grpc-bidi", None]:
                            log(ERROR, "Client raised an exception.", exc_info=ex)
                            # Raise exception, crash process
                            raise ex

                        # Don't update/change DeprecatedRunInfoStore

                        e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
                        # Ex fmt: "<class 'ZeroDivisionError'>:<'division by zero'>"
                        reason = str(type(ex)) + ":<'" + str(ex) + "'>"
                        exc_entity = "ClientApp"
                        if isinstance(ex, LoadClientAppError):
                            reason = (
                                "An exception was raised when attempting to load "
                                "`ClientApp`"
                            )
                            e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION
                            exc_entity = "SuperNode"

                        log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)

                        # Create error message
                        reply_message = Message(
                            Error(code=e_code, reason=reason),
                            reply_to=message,
                        )
                    else:
                        # No exception, update node state
                        run_info_store.update_context(
                            run_id=run_id,
                            context=context,
                        )

                    # Send
                    send(reply_message)
                    log(INFO, "Sent reply")

                except RunNotRunningException:
                    log(INFO, "")
                    log(
                        INFO,
                        "SuperNode aborted sending the reply message. "
                        "Run ID %s is not in `RUNNING` status.",
                        run_id,
                    )
                    log(INFO, "")
            # pylint: enable=too-many-nested-blocks

            # Unregister node
            if delete_node is not None:
                delete_node()  # pylint: disable=not-callable

        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break

        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)


def start_numpy_client(
    *,
    server_address: str,
    client: NumPyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: bytes | None = None,
    insecure: bool | None = None,
    transport: str | None = None,
) -> None:
    """Start a Flower NumPyClient which connects to a gRPC server.

    Warning
    -------
    This function is deprecated since 1.7.0. Use :code:`flwr.client.start_client`
    instead and first convert your :code:`NumPyClient` to type
    :code:`flwr.client.Client` by executing its :code:`to_client()` method.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower server runs on
        the same machine on port 8080, then `server_address` would be
        `"[::]:8080"`.
    client : flwr.client.NumPyClient
        An implementation of the abstract base class `flwr.client.NumPyClient`.
    grpc_max_message_length : int (default: 536_870_912, this equals 512MB)
        The maximum length of gRPC messages that can be exchanged with the
        Flower server. The default should be sufficient for most models.
        Users who train very large models might need to increase this
        value. Note that the Flower server needs to be started with the
        same value (see `flwr.server.start_server`), otherwise it will not
        know about the increased limit and block larger messages.
    root_certificates : bytes (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : Optional[str] (default: None)
        **[Deprecated]** This argument is no longer supported and will be
        removed in a future release.

    Examples
    --------
    Starting a gRPC client with an insecure server connection::

        start_numpy_client(
            server_address=localhost:8080,
            client=FlowerClient(),
        )

    Starting a TLS-enabled gRPC client using system certificates::

        start_numpy_client(
            server_address=localhost:8080,
            client=FlowerClient(),
            insecure=False,
        )

    Starting a TLS-enabled gRPC client using provided certificates::

        from pathlib import Path

        start_numpy_client(
            server_address=localhost:8080,
            client=FlowerClient(),
            root_certificates=Path("/crts/root.pem").read_bytes(),
        )
    """
    mssg = (
        "flwr.client.start_numpy_client() is deprecated. \n\tInstead, use "
        "`flwr.client.start_client()` by ensuring you first call "
        "the `.to_client()` method as shown below: \n"
        "\tflwr.client.start_client(\n"
        "\t\tserver_address='<IP>:<PORT>',\n"
        "\t\tclient=FlowerClient().to_client(),"
        " # <-- where FlowerClient is of type flwr.client.NumPyClient object\n"
        "\t)\n"
        "\tUsing `start_numpy_client()` is deprecated."
    )

    warn_deprecated_feature(name=mssg)

    # Calling this function is deprecated. A warning is thrown.
    # We first need to convert the supplied client to `Client.`

    wrp_client = client.to_client()

    start_client(
        server_address=server_address,
        client=wrp_client,
        grpc_max_message_length=grpc_max_message_length,
        root_certificates=root_certificates,
        insecure=insecure,
        transport=transport,
    )


def _init_connection(transport: str | None, server_address: str) -> tuple[
    Callable[
        [
            str,
            bool,
            RetryInvoker,
            int,
            bytes | str | None,
            tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None,
        ],
        AbstractContextManager[
            tuple[
                Callable[[], Message | None],
                Callable[[Message], None],
                Callable[[], int | None] | None,
                Callable[[], None] | None,
                Callable[[int], Run] | None,
                Callable[[str, int], Fab] | None,
            ]
        ],
    ],
    str,
    type[Exception],
]:
    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"SuperLink address ({server_address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Set the default transport layer
    if transport is None:
        transport = TRANSPORT_TYPE_GRPC_BIDI

    # Use gRPC bidirectional streaming
    if transport == TRANSPORT_TYPE_GRPC_BIDI:
        connection, error_type = grpc_connection, RpcError
    else:
        raise ValueError(
            f"Unknown transport type: {transport} (possible: {TRANSPORT_TYPES_COMPAT})"
        )

    return connection, address, error_type
