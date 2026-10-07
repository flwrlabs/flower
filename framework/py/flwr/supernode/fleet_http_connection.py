# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Fleet HTTP connection for the SuperNode worker loop."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging import ERROR
from pathlib import Path

import httpx
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.app.message import Message, remove_content_from_message
from flwr.common.constant import HEARTBEAT_DEFAULT_INTERVAL
from flwr.common.serde import (
    fab_from_proto,
    message_from_proto,
    message_to_proto,
    run_from_proto,
)
from flwr.proto.fab_pb2 import GetFabRequest  # pylint: disable=E0611
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    DeactivateNodeRequest,
    PullMessagesRequest,
    PushMessagesRequest,
    RegisterNodeFleetRequest,
    UnregisterNodeFleetRequest,
)
from flwr.proto.heartbeat_pb2 import SendNodeHeartbeatRequest  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.error import FlowerError
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.fab import Fab
from flwr.supercore.heartbeat import HeartbeatSender
from flwr.supercore.inflatable.inflatable_protobuf_utils import (
    make_confirm_message_received_fn_protobuf,
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)
from flwr.supercore.interceptors.http.runtime_version import (
    RuntimeVersionHttpInterceptor,
)
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes
from flwr.supercore.retry import make_simple_http_retry_invoker
from flwr.supercore.run import Run

from .fleet_http_client import FleetHttpClient
from .node_auth_http_interceptor import NodeAuthHttpInterceptor


@contextmanager
def http_request_response(  # pylint: disable=R0913,R0917,R0914,R0912,R0915
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
        Callable[[Message, ObjectTree, float], tuple[set[str], str]],
        Callable[[int], Run],
        Callable[[str, int], Fab],
        Callable[[int, str], bytes],
        Callable[[int, str, str, bytes], None],
        Callable[[int, str], None],
    ]
]:
    """Provide the worker callbacks over Fleet protobuf HTTP."""
    if isinstance(root_certificates, str):
        root_certificates = str(Path(root_certificates).expanduser())

    self_registered = authentication_keys is None
    if authentication_keys is None:
        authentication_keys = generate_key_pairs()
    private_key, public_key = authentication_keys
    node_pk = public_key_to_bytes(public_key)

    retry_invoker = make_simple_http_retry_invoker()
    if max_retries is not None:
        retry_invoker.max_tries = max_retries + 1
    if max_wait_time is not None:
        retry_invoker.max_time = max_wait_time
    heartbeat_retry_invoker = make_simple_http_retry_invoker()
    heartbeat_retry_invoker.max_tries = 1

    with FleetHttpClient.from_server_address(
        server_address,
        insecure,
        root_certificates,
        interceptors=[
            RuntimeVersionHttpInterceptor(component_name="SuperNode"),
            NodeAuthHttpInterceptor(private_key, public_key),
        ],
    ) as client:
        node: Node | None = None

        def send_node_heartbeat() -> bool:
            if node is None:
                return False
            try:
                response = heartbeat_retry_invoker.invoke(
                    client.SendNodeHeartbeat,
                    SendNodeHeartbeatRequest(
                        node=node, heartbeat_interval=HEARTBEAT_DEFAULT_INTERVAL
                    ),
                )
            except httpx.TransportError:
                return False
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (503, 504):
                    return False
                raise
            if not response.success:
                raise RuntimeError(
                    "Heartbeat failed unexpectedly. The SuperLink does not "
                    "recognize this SuperNode."
                )
            return True

        heartbeat_sender = HeartbeatSender(send_node_heartbeat)

        def receive() -> tuple[Message, ObjectTree] | None:
            if node is None:
                return None
            response = retry_invoker.invoke(
                client.PullMessages, PullMessagesRequest(node=node)
            )
            if not response.messages_list:
                return None
            return (
                message_from_proto(response.messages_list[0]),
                response.message_object_trees[0],
            )

        def send(
            message: Message, object_tree: ObjectTree, clientapp_runtime: float
        ) -> tuple[set[str], str]:
            if node is None:
                return set(), ""
            if message.has_content():
                message = remove_content_from_message(message)
            response = retry_invoker.invoke(
                client.PushMessages,
                PushMessagesRequest(
                    node=node,
                    messages_list=[message_to_proto(message)],
                    message_object_trees=[object_tree],
                    clientapp_runtime_list=[clientapp_runtime],
                ),
            )
            return set(response.objects_to_push), response.session_id

        def get_run(run_id: int) -> Run:
            response = retry_invoker.invoke(
                client.GetRun, GetRunRequest(node=node, run_id=run_id)
            )
            return run_from_proto(response.run)

        def get_fab(fab_hash: str, run_id: int) -> Fab:
            response = retry_invoker.invoke(
                client.GetFab,
                GetFabRequest(node=node, hash_str=fab_hash, run_id=run_id),
            )
            return fab_from_proto(response.fab)

        def pull_object(run_id: int, object_id: str) -> bytes:
            if node is None:
                raise RuntimeError("Node instance missing")
            return make_pull_object_fn_protobuf(
                lambda request: retry_invoker.invoke(client.PullObject, request),
                node,
                run_id,
            )(object_id)

        def push_object(
            run_id: int, session_id: str, object_id: str, contents: bytes
        ) -> None:
            if node is None:
                raise RuntimeError("Node instance missing")
            make_push_object_fn_protobuf(
                lambda request: retry_invoker.invoke(client.PushObject, request),
                node,
                run_id,
                session_id,
            )(object_id, contents)

        def confirm_message_received(run_id: int, object_id: str) -> None:
            if node is None:
                raise RuntimeError("Node instance missing")
            make_confirm_message_received_fn_protobuf(
                lambda request: retry_invoker.invoke(
                    client.ConfirmMessageReceived, request
                ),
                node,
                run_id,
            )(object_id)

        connection_initialized = False
        try:
            if self_registered:
                retry_invoker.invoke(
                    client.RegisterNode, RegisterNodeFleetRequest(public_key=node_pk)
                )
            response = retry_invoker.invoke(
                client.ActivateNode,
                ActivateNodeRequest(
                    public_key=node_pk,
                    heartbeat_interval=HEARTBEAT_DEFAULT_INTERVAL,
                ),
            )
            node = Node(node_id=response.node_id)
            heartbeat_sender.start()
            connection_initialized = True
            yield (
                node.node_id,
                receive,
                send,
                get_run,
                get_fab,
                pull_object,
                push_object,
                confirm_message_received,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if not connection_initialized:
                message = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    if flower_error := FlowerError.from_json(exc.response.text):
                        message = f"[code: {flower_error.code}] {flower_error.message}"
                        if flower_error.public_details:
                            message += f"\n{flower_error.public_details}"
                flwr_exit(
                    ExitCode.SUPERNODE_CONNECTION_ERROR,
                    "Failed to initialize the connection to the SuperLink.\n" + message,
                )
            log(ERROR, exc)
        finally:
            if node is not None:
                retry_invoker.max_tries = 1
                if heartbeat_sender.is_running:
                    heartbeat_sender.stop()
                try:
                    client.DeactivateNode(DeactivateNodeRequest(node_id=node.node_id))
                except (httpx.HTTPError, RuntimeError):
                    pass
                if self_registered:
                    try:
                        client.UnregisterNode(
                            UnregisterNodeFleetRequest(node_id=node.node_id)
                        )
                    except (httpx.HTTPError, RuntimeError):
                        pass
