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
"""Contextmanager for a REST request-response channel to the Flower server."""


from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging import ERROR, WARN
from typing import TypeVar

from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.message import Message as GrpcMessage
from requests.exceptions import ConnectionError as RequestsConnectionError

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.constant import HEARTBEAT_DEFAULT_INTERVAL
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.common.message import Message, remove_content_from_message
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.serde import (
    fab_from_proto,
    message_from_proto,
    message_to_proto,
    run_from_proto,
)
from flwr.common.typing import Fab, Run
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
from flwr.proto.heartbeat_pb2 import (  # pylint: disable=E0611
    SendNodeHeartbeatRequest,
    SendNodeHeartbeatResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    ObjectTree,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.supercore.heartbeat import HeartbeatSender
from flwr.supercore.inflatable.inflatable_protobuf_utils import (
    make_confirm_message_received_fn_protobuf,
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes

try:
    import requests
except ModuleNotFoundError:
    flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)


PATH_REGISTER_NODE: str = "/api/v0/fleet/register-node"
PATH_ACTIVATE_NODE: str = "/api/v0/fleet/activate-node"
PATH_DEACTIVATE_NODE: str = "/api/v0/fleet/deactivate-node"
PATH_UNREGISTER_NODE: str = "/api/v0/fleet/unregister-node"
PATH_PULL_MESSAGES: str = "/api/v0/fleet/pull-messages"
PATH_PUSH_MESSAGES: str = "/api/v0/fleet/push-messages"
PATH_PULL_OBJECT: str = "/api/v0/fleet/pull-object"
PATH_PUSH_OBJECT: str = "/api/v0/fleet/push-object"
PATH_SEND_NODE_HEARTBEAT: str = "api/v0/fleet/send-node-heartbeat"
PATH_GET_RUN: str = "/api/v0/fleet/get-run"
PATH_GET_FAB: str = "/api/v0/fleet/get-fab"
PATH_CONFIRM_MESSAGE_RECEIVED: str = "/api/v0/fleet/confirm-message-received"

T = TypeVar("T", bound=GrpcMessage)


@contextmanager
def http_request_response(  # pylint: disable=R0913,R0914,R0915,R0917
    server_address: str,
    insecure: bool,  # pylint: disable=unused-argument
    retry_invoker: RetryInvoker,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,  # pylint: disable=W0613
    root_certificates: bytes | str | None = None,  # pylint: disable=unused-argument
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
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
    """Primitives for request/response-based interaction with a server.

    One notable difference to the grpc_connection context manager is that
    `receive` can return `None`.

    Parameters
    ----------
    server_address : str
        The IPv6 address of the server with `http://` or `https://`.
        If the Flower server runs on the same machine
        on port 8080, then `server_address` would be `"http://[::]:8080"`.
    insecure : bool
        Unused argument present for compatibilty.
    retry_invoker: RetryInvoker
        `RetryInvoker` object that will try to reconnect the client to the server
        after REST connection errors. If None, the client will only try to
        reconnect once after a failure.
    max_message_length : int
        Ignored, only present to preserve API-compatibility.
    root_certificates : Optional[Union[bytes, str]] (default: None)
        Path of the root certificate. If provided, a secure
        connection using the certificates will be established to an SSL-enabled
        Flower server. Bytes won't work for the REST API.
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        SuperNode authentication is not supported for this transport type.

    Returns
    -------
    node_id : int
    receive : Callable[[], Optional[tuple[Message, ObjectTree]]]
    send : Callable[[Message, ObjectTree, float], set[str]]
    get_run : Callable[[int], Run]
    get_fab : Callable[[str, int], Fab]
    pull_object : Callable[[str], bytes]
    push_object : Callable[[str, bytes], None]
    confirm_message_received : Callable[[str], None]
    """
    log(
        WARN,
        """
        EXPERIMENTAL: `rest` is an experimental feature, it might change
        considerably in future versions of Flower
        """,
    )

    base_url = server_address

    # NEVER SET VERIFY TO FALSE
    # Otherwise any server can fake its identity
    # Please refer to:
    # https://requests.readthedocs.io/en/latest/user/advanced/#ssl-cert-verification
    verify: bool | str = True
    if isinstance(root_certificates, str):
        verify = root_certificates
    elif isinstance(root_certificates, bytes):
        log(
            ERROR,
            "For the REST API, the root certificates "
            "must be provided as a string path to the client.",
        )
    if authentication_keys is not None:
        log(ERROR, "SuperNode authentication is not supported for this transport type.")

    # REST does NOT support node authentication
    self_registered = False
    if authentication_keys is None:
        self_registered = True
        authentication_keys = generate_key_pairs()
    node_pk = public_key_to_bytes(authentication_keys[1])

    # Shared variables for inner functions
    node: Node | None = None

    # Remove should_giveup from RetryInvoker as REST does not support gRPC status codes
    retry_invoker.should_giveup = None

    ###########################################################################
    # SuperNode functions
    ###########################################################################

    def _request(
        req: GrpcMessage, res_type: type[T], api_path: str, retry: bool = True
    ) -> T | None:
        # Serialize the request
        req_bytes = req.SerializeToString()

        # Send the request
        def post() -> requests.Response:
            return requests.post(
                f"{base_url}/{api_path}",
                data=req_bytes,
                headers={
                    "Accept": "application/protobuf",
                    "Content-Type": "application/protobuf",
                },
                verify=verify,
                timeout=None,
            )

        if retry:
            res: requests.Response = retry_invoker.invoke(post)
        else:
            res = post()

        # Check status code and headers
        if res.status_code != 200:
            return None
        if "content-type" not in res.headers:
            log(
                WARN,
                "[Node] POST /%s: missing header `Content-Type`",
                api_path,
            )
            return None
        if res.headers["content-type"] != "application/protobuf":
            log(
                WARN,
                "[Node] POST /%s: header `Content-Type` has wrong value",
                api_path,
            )
            return None

        # Deserialize ProtoBuf from bytes
        grpc_res = res_type()
        grpc_res.ParseFromString(res.content)
        return grpc_res

    def _pull_object_protobuf(request: PullObjectRequest) -> PullObjectResponse:
        res = _request(
            req=request,
            res_type=PullObjectResponse,
            api_path=PATH_PULL_OBJECT,
        )
        if res is None:
            raise ValueError(f"{PullObjectResponse.__name__} is None.")
        return res

    def _push_object_protobuf(request: PushObjectRequest) -> PushObjectResponse:
        res = _request(
            req=request,
            res_type=PushObjectResponse,
            api_path=PATH_PUSH_OBJECT,
        )
        if res is None:
            raise ValueError(f"{PushObjectResponse.__name__} is None.")
        return res

    def _confirm_message_received_protobuf(
        request: ConfirmMessageReceivedRequest,
    ) -> ConfirmMessageReceivedResponse:
        res = _request(
            req=request,
            res_type=ConfirmMessageReceivedResponse,
            api_path=PATH_CONFIRM_MESSAGE_RECEIVED,
        )
        if res is None:
            raise ValueError(f"{ConfirmMessageReceivedResponse.__name__} is None.")
        return res

    def send_node_heartbeat() -> bool:
        # Get Node
        if node is None:
            log(ERROR, "Node instance missing")
            return False

        # Construct the heartbeat request
        req = SendNodeHeartbeatRequest(
            node=node, heartbeat_interval=HEARTBEAT_DEFAULT_INTERVAL
        )

        # Send the request
        res = _request(
            req, SendNodeHeartbeatResponse, PATH_SEND_NODE_HEARTBEAT, retry=False
        )
        if res is None:
            return False

        # Check if success
        if not res.success:
            raise RuntimeError(
                "Heartbeat failed unexpectedly. The SuperLink does not "
                "recognize this SuperNode."
            )
        return True

    heartbeat_sender = HeartbeatSender(send_node_heartbeat)

    def register_node() -> None:
        """Register node with SuperLink."""
        req = RegisterNodeFleetRequest(public_key=node_pk)

        # Send the request
        res = _request(req, RegisterNodeFleetResponse, PATH_REGISTER_NODE)
        if res is None:
            raise RuntimeError("Failed to register node")

    def activate_node() -> int:
        """Activate node and start heartbeat."""
        req = ActivateNodeRequest(
            public_key=node_pk,
            heartbeat_interval=HEARTBEAT_DEFAULT_INTERVAL,
        )

        # Send the request
        res = _request(req, ActivateNodeResponse, PATH_ACTIVATE_NODE)
        if res is None:
            raise RuntimeError("Failed to activate node")

        # Remember the node and start the heartbeat sender
        nonlocal node
        node = Node(node_id=res.node_id)
        heartbeat_sender.start()
        return node.node_id

    def deactivate_node() -> None:
        """Deactivate node and stop heartbeat."""
        nonlocal node
        if node is None:
            raise RuntimeError("Node instance missing")

        # Stop the heartbeat sender
        heartbeat_sender.stop()

        # Send DeactivateNode request
        req = DeactivateNodeRequest(node_id=node.node_id)

        # Send the request
        res = _request(req, DeactivateNodeResponse, PATH_DEACTIVATE_NODE)
        if res is None:
            raise RuntimeError("Failed to deactivate node")

    def unregister_node() -> None:
        """Unregister node from SuperLink."""
        nonlocal node
        if node is None:
            raise RuntimeError("Node instance missing")

        # Send UnregisterNode request
        req = UnregisterNodeFleetRequest(node_id=node.node_id)

        # Send the request
        res = _request(req, UnregisterNodeFleetResponse, PATH_UNREGISTER_NODE)
        if res is None:
            raise RuntimeError("Failed to unregister node")

        # Cleanup
        node = None

    def receive() -> tuple[Message, ObjectTree] | None:
        """Pull a message with its ObjectTree from SuperLink."""
        # Get Node
        if node is None:
            raise RuntimeError("Node instance missing")

        # Try to pull a message with its object tree from SuperLink
        req = PullMessagesRequest(node=node)
        res = _request(req, PullMessagesResponse, PATH_PULL_MESSAGES)
        if res is None:
            raise ValueError("PushMessagesResponse is None.")

        # If no messages are available, return None
        if len(res.messages_list) == 0:
            return None

        # Get the current Message and its object tree
        message_proto = res.messages_list[0]
        object_tree = res.message_object_trees[0]

        # Construct the Message
        in_message = message_from_proto(message_proto)

        # Return the Message and its object tree
        return in_message, object_tree

    def send(
        message: Message, object_tree: ObjectTree, clientapp_runtime: float
    ) -> set[str]:
        """Send the message with its ObjectTree to SuperLink."""
        # Get Node
        if node is None:
            raise RuntimeError("Node instance missing")
        # Remove the content from the message if it has
        if message.has_content():
            message = remove_content_from_message(message)

        # Send the message with its ObjectTree to SuperLink
        req = PushMessagesRequest(
            node=node,
            messages_list=[message_to_proto(message)],
            message_object_trees=[object_tree],
            clientapp_runtime_list=[clientapp_runtime],
        )
        res = _request(req, PushMessagesResponse, PATH_PUSH_MESSAGES)
        if res is None:
            raise ValueError("PushMessagesResponse is None.")

        # Get and return the object IDs to push
        return set(res.objects_to_push)

    def get_run(run_id: int) -> Run:
        # Construct the request
        req = GetRunRequest(node=node, run_id=run_id)

        # Send the request
        res = _request(req, GetRunResponse, PATH_GET_RUN)
        if res is None:
            return Run.create_empty(run_id)

        return run_from_proto(res.run)

    def get_fab(fab_hash: str, run_id: int) -> Fab:
        # Construct the request
        req = GetFabRequest(node=node, hash_str=fab_hash, run_id=run_id)

        # Send the request
        res = _request(req, GetFabResponse, PATH_GET_FAB)
        if res is None:
            return Fab("", b"", {})

        return fab_from_proto(res.fab)

    def pull_object(run_id: int, object_id: str) -> bytes:
        """Pull the object from the SuperLink."""
        # Check Node
        if node is None:
            raise RuntimeError("Node instance missing")

        fn = make_pull_object_fn_protobuf(
            pull_object_protobuf=_pull_object_protobuf,
            node=node,
            run_id=run_id,
        )
        return fn(object_id)

    def push_object(run_id: int, object_id: str, contents: bytes) -> None:
        """Push the object to the SuperLink."""
        # Check Node
        if node is None:
            raise RuntimeError("Node instance missing")

        fn = make_push_object_fn_protobuf(
            push_object_protobuf=_push_object_protobuf,
            node=node,
            run_id=run_id,
        )
        fn(object_id, contents)

    def confirm_message_received(run_id: int, object_id: str) -> None:
        """Confirm that the message has been received."""
        # Check Node
        if node is None:
            raise RuntimeError("Node instance missing")

        fn = make_confirm_message_received_fn_protobuf(
            confirm_message_received_protobuf=_confirm_message_received_protobuf,
            node=node,
            run_id=run_id,
        )
        fn(object_id)

    try:
        if self_registered:
            register_node()
        node_id = activate_node()
        # Yield methods
        yield (
            node_id,
            receive,
            send,
            get_run,
            get_fab,
            pull_object,
            push_object,
            confirm_message_received,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log(ERROR, exc)
    # Cleanup
    finally:
        try:
            if node is not None:
                # Disable retrying
                retry_invoker.max_tries = 1
                deactivate_node()
                if self_registered:
                    unregister_node()
        except RequestsConnectionError:
            pass
