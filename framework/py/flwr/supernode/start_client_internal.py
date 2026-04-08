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
"""Main loop for Flower SuperNode."""


import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from logging import ERROR, INFO
from typing import cast

import grpc
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.hazmat.primitives.serialization.ssh import load_ssh_public_key
from grpc import RpcError

from flwr.app.user_config import UserConfig
from flwr.client.grpc_adapter_client.connection import grpc_adapter
from flwr.client.grpc_rere_client.connection import grpc_request_response
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, Context, Error, Message, RecordDict
from flwr.common.config import get_fused_config_from_fab
from flwr.common.constant import (
    CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
    ISOLATION_MODE_SUBPROCESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    TRANSPORT_TYPES,
    ErrorCode,
    ExecPluginType,
)
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import log
from flwr.common.retry_invoker import RetryInvoker, make_simple_grpc_retry_invoker
from flwr.common.telemetry import EventType
from flwr.common.typing import Fab, Run, RunNotRunningException
from flwr.proto.clientappio_pb2_grpc import add_ClientAppIoServicer_to_server
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.address import parse_address, resolve_bind_address
from flwr.supercore.grpc_health import run_health_server_grpc_no_tls
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    iterate_object_tree,
    no_object_id_recompute,
)
from flwr.supercore.inflatable.inflatable_utils import (
    pull_objects,
    push_object_contents_from_iterable,
)
from flwr.supercore.interceptors import create_clientappio_token_auth_server_interceptor
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory
from flwr.supercore.primitives.asymmetric_ed25519 import (
    create_message_to_sign,
    decode_base64url,
    verify_signature,
)
from flwr.supercore.version import package_version
from flwr.supernode.nodestate import NodeState, NodeStateFactory
from flwr.supernode.servicer.clientappio import ClientAppIoServicer

FAB_VERIFICATION_ERROR = Error(ErrorCode.INVALID_FAB, "The FAB could not be verified.")


# pylint: disable=import-outside-toplevel
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-arguments
def start_client_internal(
    *,
    server_address: str,
    node_config: UserConfig,
    root_certificates: bytes | str | None = None,
    insecure: bool | None = None,
    transport: str,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
    max_retries: int | None = None,
    max_wait_time: float | None = None,
    isolation: str = ISOLATION_MODE_SUBPROCESS,
    clientappio_api_address: str = CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS,
    health_server_address: str | None = None,
    trusted_entities: dict[str, str] | None = None,
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
    root_certificates : Optional[Union[bytes, str]] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    insecure : Optional[bool] (default: None)
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    transport : str
        Configure the transport layer. Allowed values:
        - 'grpc-rere': gRPC, request-response
        - 'grpc-adapter': gRPC via 3rd party adapter (experimental)
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
    isolation : str (default: ISOLATION_MODE_SUBPROCESS)
        Isolation mode for `ClientApp`. Possible values are `subprocess` and
        `process`. If `subprocess`, the `ClientApp` runs in a subprocess started
        by the SueprNode and communicates using gRPC at the address
        `clientappio_api_address`. If `process`, the `ClientApp` runs in a separate
        isolated process and communicates using gRPC at the address
        `clientappio_api_address`.
    clientappio_api_address : str
        (default: `CLIENTAPPIO_API_DEFAULT_SERVER_ADDRESS`)
        The SuperNode gRPC server address.
    health_server_address : Optional[str] (default: None)
        The address of the health server. If `None` is provided, the health server will
        NOT be started.
    trusted_entities : Optional[dict[str, str]] (default: None)
        A dictionary mapping public key IDs to public keys.
        Only apps verified by at least one of these
        entities can run on a supernode.
    """
    if insecure is None:
        insecure = root_certificates is None

    # Insecure HTTP is incompatible with authentication
    if insecure and authentication_keys is not None:
        url_v = f"https://flower.ai/docs/framework/v{package_version}/en/"
        page = "how-to-authenticate-supernodes.html"
        flwr_exit(
            ExitCode.SUPERNODE_STARTED_WITHOUT_TLS_BUT_NODE_AUTH_ENABLED,
            "Insecure connection is enabled, but the SuperNode's private key is "
            "provided for authentication. SuperNode authentication requires a "
            "secure TLS connection with the SuperLink. Please enable TLS by "
            "providing the certificate via `--root-certificates`. Please refer "
            f"to the Flower documentation for more information: {url_v}{page}",
        )

    # Initialize factories
    object_store_factory = ObjectStoreFactory()
    state_factory = NodeStateFactory(objectstore_factory=object_store_factory)

    # Launch ClientAppIo API server
    grpc_servers = []
    clientappio_server = run_clientappio_api_grpc(
        address=clientappio_api_address,
        state_factory=state_factory,
        objectstore_factory=object_store_factory,
        certificates=None,
    )
    grpc_servers.append(clientappio_server)

    # Launch gRPC health server
    if health_server_address is not None:
        health_server = run_health_server_grpc_no_tls(health_server_address)
        grpc_servers.append(health_server)

    # Register handlers for graceful shutdown
    register_signal_handlers(
        event_type=EventType.RUN_SUPERNODE_LEAVE,
        exit_message="SuperNode terminated gracefully.",
        grpc_servers=grpc_servers,
    )

    # Initialize NodeState and ObjectStore
    state = state_factory.state()
    store = object_store_factory.store()

    # Launch the SuperExec if the isolation mode is `subprocess`
    if isolation == ISOLATION_MODE_SUBPROCESS:
        command = ["flower-superexec", "--insecure"]
        command += [
            "--appio-api-address",
            resolve_bind_address(clientappio_api_address),
        ]
        command += ["--plugin-type", ExecPluginType.CLIENT_APP]
        command += ["--parent-pid", str(os.getpid())]
        # pylint: disable-next=consider-using-with
        subprocess.Popen(command)

    with _init_connection(
        transport=transport,
        server_address=server_address,
        insecure=insecure,
        root_certificates=root_certificates,
        authentication_keys=authentication_keys,
        max_retries=max_retries,
        max_wait_time=max_wait_time,
    ) as conn:
        (
            node_id,
            receive,
            send,
            get_run,
            get_fab,
            pull_object,
            push_object,
            confirm_message_received,
        ) = conn
        # Store node_id in state
        state.set_node_id(node_id)
        log(INFO, "SuperNode ID: %s", node_id)

        # pylint: disable=too-many-nested-blocks
        while True:
            run_id = _pull_and_store_message(
                state=state,
                object_store=store,
                node_config=node_config,
                receive=receive,
                get_run=get_run,
                get_fab=get_fab,
                pull_object=pull_object,
                confirm_message_received=confirm_message_received,
                trusted_entities=trusted_entities,
            )

            # No message has been pulled therefore we can skip the push stage.
            if run_id is None:
                # If no message was received, wait for a while
                time.sleep(3)
                continue

            _push_messages(
                state=state,
                object_store=store,
                send=send,
                push_object=push_object,
            )


def _insert_message(msg: Message, state: NodeState, store: ObjectStore) -> None:
    """Insert a message into the NodeState and ObjectStore."""
    with no_object_id_recompute():
        # Store message in state
        msg.metadata.__dict__["_message_id"] = msg.object_id  # Set message_id
        state.store_message(msg)

        # Preregister objects in ObjectStore
        store.preregister(msg.metadata.run_id, get_object_tree(msg))

        # Store all objects in ObjectStore
        all_objects = get_all_nested_objects(msg)
        for obj_id, obj in all_objects.items():
            store.put(obj_id, obj.deflate())


def _pull_and_store_message(  # pylint: disable=too-many-positional-arguments
    state: NodeState,
    object_store: ObjectStore,
    node_config: UserConfig,
    receive: Callable[[], tuple[Message, ObjectTree] | None],
    get_run: Callable[[int], Run],
    get_fab: Callable[[str, int], Fab],
    pull_object: Callable[[int, str], bytes],
    confirm_message_received: Callable[[int, str], None],
    trusted_entities: dict[str, str] | None,
) -> int | None:
    """Pull a message from the SuperLink and store it in the state.

    Return None if no message is received, otherwise return the processed run_id.
    """
    # pylint: disable=too-many-nested-blocks
    message = None
    try:
        # Pull message
        if (recv := receive()) is None:
            return None
        message, object_tree = recv

        # Log message reception
        log(INFO, "")
        if message.metadata.group_id:
            log(
                INFO,
                "[RUN %s, ROUND %s]",
                message.metadata.run_id,
                message.metadata.group_id,
            )
        else:
            log(INFO, "[RUN %s]", message.metadata.run_id)
        log(
            INFO,
            "Receiving: %s message (ID: %s)",
            message.metadata.message_type,
            message.metadata.message_id,
        )

        # Ensure the run and FAB are available
        run_id = message.metadata.run_id

        # Check if the message is from an unknown run
        if (run_info := state.get_run(run_id)) is None:
            # Pull run info from SuperLink
            run_info = get_run(run_id)

            # Pull and store the FAB
            fab = get_fab(run_info.fab_hash, run_id)

            # Verify the received FAB
            # FAB must be signed if trust entities provided
            if trusted_entities:
                if not fab.verifications.get("valid_license", ""):
                    log(
                        ERROR,
                        "The FAB could not be verified. App verification is "
                        "not supported by the connected SuperLink.",
                    )
                    reply = Message(FAB_VERIFICATION_ERROR, reply_to=message)
                    _insert_message(reply, state, object_store)
                    return run_id

                fab_verified = _verify_fab(fab, trusted_entities)
                if not fab_verified:
                    # Insert an error message in the state
                    # when FAB verification fails
                    log(
                        ERROR,
                        "FAB verification failed: the provided trusted entities "
                        "could not verify the FAB. An error reply "
                        "has been generated.",
                    )
                    reply = Message(FAB_VERIFICATION_ERROR, reply_to=message)
                    _insert_message(reply, state, object_store)
                    return run_id

            # Initialize the context
            run_cfg = get_fused_config_from_fab(fab.content, run_info)
            run_ctx = Context(
                run_id=run_id,
                node_id=state.get_node_id(),
                node_config=node_config,
                state=RecordDict(),
                run_config=run_cfg,
            )

            # Store in the state
            state.store_context(run_ctx)
            state.store_run(run_info)
            state.store_fab(fab)

        # Preregister the object tree of the message
        obj_ids_to_pull = object_store.preregister(run_id, object_tree)

        # Store the message in the state (note this message has no content)
        state.store_message(message)

        try:
            # Pull and store objects of the message in the ObjectStore
            obj_contents = pull_objects(
                obj_ids_to_pull,
                pull_object_fn=lambda obj_id: pull_object(run_id, obj_id),
            )
            for obj_id in list(obj_contents.keys()):
                object_store.put(obj_id, obj_contents.pop(obj_id))

            # Confirm that the message was received
            confirm_message_received(run_id, message.metadata.message_id)
            log(INFO, "Received successfully")
        except Exception as err:  # pylint: disable=broad-except
            log(
                ERROR,
                "Failed to receive message %s: %s",
                message.metadata.message_id,
                err,
            )
            state.delete_messages(message_ids=[message.metadata.message_id])
            object_store.delete(message.metadata.message_id)

    except RunNotRunningException:
        if message is None:
            log(
                INFO,
                "Run transitioned to a non-`RUNNING` status while receiving a message. "
                "Ignoring the message.",
            )
        else:
            log(
                INFO,
                "Run ID %s is not in `RUNNING` status. Ignoring message %s.",
                run_id,
                message.metadata.message_id,
            )
        return None

    return run_id


def _push_messages(
    state: NodeState,
    object_store: ObjectStore,
    send: Callable[[Message, ObjectTree, float], set[str]],
    push_object: Callable[[int, str, bytes], None],
) -> None:
    """Push reply messages to the SuperLink."""
    # This is to ensure that only one message is processed at a time
    # Wait until a reply message is available
    while not (reply_messages := state.get_messages(is_reply=True)):
        time.sleep(0.5)

    for message in reply_messages:
        # Log message sending
        log(INFO, "")
        if message.metadata.group_id:
            log(
                INFO,
                "[RUN %s, ROUND %s]",
                message.metadata.run_id,
                message.metadata.group_id,
            )
        else:
            log(INFO, "[RUN %s]", message.metadata.run_id)
        log(
            INFO,
            "Sending: %s message (ID: %s)",
            message.metadata.message_type,
            message.metadata.message_id,
        )

        # Get the object tree for the message
        object_tree = object_store.get_object_tree(message.metadata.message_id)

        # Define the iterator for yielding object contents
        # This will yield (object_id, content) pairs
        def yield_object_contents(
            _obj_tree: ObjectTree, obj_id_set: set[str]
        ) -> Iterator[tuple[str, bytes]]:
            for tree in iterate_object_tree(_obj_tree):
                if tree.object_id not in obj_id_set:
                    continue
                while (content := object_store.get(tree.object_id)) == b"":
                    # Wait for the content to be available
                    time.sleep(0.5)
                # At this point, content is guaranteed to be available
                # therefore we can yield it after casting it to bytes
                yield tree.object_id, cast(bytes, content)

        # Send the message
        try:
            clientapp_runtime = state.get_message_processing_duration(
                message_id=message.metadata.reply_to_message_id,
            )
            # Send the reply message with its ObjectTree and ClientApp runtime
            # Get the IDs of objects to send
            ids_obj_to_send = send(message, object_tree, clientapp_runtime)

            # Push object contents from the ObjectStore
            run_id = message.metadata.run_id
            push_object_contents_from_iterable(
                yield_object_contents(object_tree, ids_obj_to_send),
                # Use functools.partial to bind run_id explicitly,
                # avoiding late binding issues and satisfying flake8 (B023)
                # Equivalent to:
                # lambda object_id, content: push_object(run_id, object_id, content)
                push_object_fn=partial(push_object, run_id),
            )
            log(INFO, "Sent successfully")
        except RunNotRunningException:
            log(
                INFO,
                "Run ID %s is not in `RUNNING` status. Ignoring reply message %s.",
                message.metadata.run_id,
                message.metadata.message_id,
            )
        except Exception as err:  # pylint: disable=broad-except
            log(
                ERROR,
                "Failed to send message %s: %s",
                message.metadata.message_id,
                err,
            )
        finally:
            # Delete the message from the state
            state.delete_messages(
                message_ids=[
                    message.metadata.message_id,
                    message.metadata.reply_to_message_id,
                ]
            )

            # Delete all its objects from the ObjectStore
            # No need to delete objects of the message it replies to, as it is
            # already deleted when the ClientApp calls `ConfirmMessageReceived`
            object_store.delete(message.metadata.message_id)


@contextmanager
def _init_connection(  # pylint: disable=too-many-positional-arguments
    transport: str,
    server_address: str,
    insecure: bool,
    root_certificates: bytes | str | None = None,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
    max_retries: int | None = None,
    max_wait_time: float | None = None,
) -> Iterator[
    tuple[
        int,
        Callable[[], tuple[Message, ObjectTree] | None],
        Callable[[Message, ObjectTree, float], set[str]],
        Callable[[int], Run],
        Callable[[str, int], Fab],
        Callable[[int, str], bytes],
        Callable[[int, str, bytes], None],
        Callable[[int, str], None],
    ]
]:
    """Establish a connection to the Fleet API server at SuperLink."""
    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"SuperLink address ({server_address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    # Use either gRPC bidirectional streaming or REST request/response
    if transport == TRANSPORT_TYPE_REST:
        try:
            from requests.exceptions import ConnectionError as RequestsConnectionError

            from flwr.client.rest_client.connection import http_request_response
        except ModuleNotFoundError:
            flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)
        if server_address[:4] != "http":
            flwr_exit(ExitCode.SUPERNODE_REST_ADDRESS_INVALID)
        connection, error_type = http_request_response, RequestsConnectionError
    elif transport == TRANSPORT_TYPE_GRPC_RERE:
        connection, error_type = grpc_request_response, RpcError
    elif transport == TRANSPORT_TYPE_GRPC_ADAPTER:
        connection, error_type = grpc_adapter, RpcError
    else:
        raise ValueError(
            f"Unknown transport type: {transport} (possible: {TRANSPORT_TYPES})"
        )

    # Create RetryInvoker
    retry_invoker = _make_fleet_connection_retry_invoker(
        max_retries=max_retries,
        max_wait_time=max_wait_time,
        connection_error_type=error_type,
    )

    # Establish connection
    with connection(
        address,
        insecure,
        retry_invoker,
        GRPC_MAX_MESSAGE_LENGTH,
        root_certificates,
        authentication_keys,
    ) as conn:
        yield conn


def _make_fleet_connection_retry_invoker(
    max_retries: int | None = None,
    max_wait_time: float | None = None,
    connection_error_type: type[Exception] = RpcError,
) -> RetryInvoker:
    """Create a retry invoker for fleet connection."""
    retry_invoker = make_simple_grpc_retry_invoker()
    retry_invoker.recoverable_exceptions = connection_error_type
    if max_retries is not None:
        retry_invoker.max_tries = max_retries + 1
    if max_wait_time is not None:
        retry_invoker.max_time = max_wait_time

    return retry_invoker


def run_clientappio_api_grpc(
    address: str,
    state_factory: NodeStateFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: tuple[bytes, bytes, bytes] | None,
) -> grpc.Server:
    """Run ClientAppIo API gRPC server."""
    clientappio_servicer: grpc.Server = ClientAppIoServicer(
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
    )
    auth_interceptor = create_clientappio_token_auth_server_interceptor(
        state_provider=state_factory.state
    )
    clientappio_add_servicer_to_server_fn = add_ClientAppIoServicer_to_server
    clientappio_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(
            clientappio_servicer,
            clientappio_add_servicer_to_server_fn,
        ),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=[auth_interceptor],
    )
    address = clientappio_grpc_server.bound_address
    log(INFO, "Flower Deployment Runtime: Starting ClientAppIo API on %s", address)
    clientappio_grpc_server.start()
    return clientappio_grpc_server


def _verify_fab(fab: Fab, trusted_entities: dict[str, str]) -> bool:
    """Verify a FAB using its verification data and the provided trusted entities.

    The FAB is considered verified if at least one trusted entity matches the
    information contained in its verification records.
    """
    verifications = fab.verifications
    verif_full = {
        k: json.loads(v) for k, v in verifications.items() if k != "valid_license"
    }
    fab_verified = False
    for public_key_id, verif in verif_full.items():
        if public_key_id in trusted_entities:
            verifier_public_key = load_ssh_public_key(
                trusted_entities[public_key_id].encode("utf-8")
            )
            message_to_verify = create_message_to_sign(
                hashlib.sha256(fab.content).digest(),
                verif["signed_at"],
            )
            assert isinstance(verifier_public_key, ed25519.Ed25519PublicKey)
            if verify_signature(
                verifier_public_key,
                message_to_verify,
                decode_base64url(verif["signature"]),
            ):
                fab_verified = True
                break
    return fab_verified
