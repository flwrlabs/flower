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
"""InflatableObject gRPC utils."""


from collections.abc import Callable

from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611

from .inflatable_utils import ObjectIdNotPreregisteredError, ObjectUnavailableError

ConfirmMessageReceivedProtobuf = Callable[
    [ConfirmMessageReceivedRequest], ConfirmMessageReceivedResponse
]


def make_pull_object_fn_protobuf(
    pull_object_protobuf: Callable[[PullObjectRequest], PullObjectResponse],
    node: Node,
    run_id: int,
) -> Callable[[str], bytes]:
    """Create a pull object function that uses gRPC to pull objects.

    Parameters
    ----------
    pull_object_protobuf : Callable[[PullObjectRequest], PullObjectResponse]
        A callable that takes a `PullObjectRequest` and returns a `PullObjectResponse`.
        This function is typically backed by a gRPC client stub.
    node : Node
        The node making the request.
    run_id : int
        The run ID for the current operation.

    Returns
    -------
    Callable[[str], bytes]
        A function that takes an object ID and returns the object content as bytes.
        The function raises `ObjectIdNotPreregisteredError` if the object ID is not
        pre-registered, or `ObjectUnavailableError` if the object is not yet available.
    """

    def pull_object_fn(object_id: str) -> bytes:
        request = PullObjectRequest(node=node, run_id=run_id, object_id=object_id)
        response: PullObjectResponse = pull_object_protobuf(request)
        if not response.object_found:
            raise ObjectIdNotPreregisteredError(object_id)
        if not response.object_available:
            raise ObjectUnavailableError(object_id)
        return response.object_content

    return pull_object_fn


def make_push_object_fn_protobuf(
    push_object_protobuf: Callable[[PushObjectRequest], PushObjectResponse],
    node: Node,
    run_id: int,
) -> Callable[[str, bytes], None]:
    """Create a push object function that uses gRPC to push objects.

    Parameters
    ----------
    push_object_protobuf : Callable[[PushObjectRequest], PushObjectResponse]
        A callable that takes a `PushObjectRequest` and returns a `PushObjectResponse`.
        This function is typically backed by a gRPC client stub.
    node : Node
        The node making the request.
    run_id : int
        The run ID for the current operation.

    Returns
    -------
    Callable[[str, bytes], None]
        A function that takes an object ID and its content as bytes, and pushes it
        to the servicer. The function raises `ObjectIdNotPreregisteredError` if
        the object ID is not pre-registered.
    """

    def push_object_fn(object_id: str, object_content: bytes) -> None:
        request = PushObjectRequest(
            node=node, run_id=run_id, object_id=object_id, object_content=object_content
        )
        response: PushObjectResponse = push_object_protobuf(request)
        if not response.stored:
            raise ObjectIdNotPreregisteredError(object_id)

    return push_object_fn


def make_confirm_message_received_fn_protobuf(
    confirm_message_received_protobuf: ConfirmMessageReceivedProtobuf,
    node: Node,
    run_id: int,
) -> Callable[[str], None]:
    """Create a confirm message received function that uses protobuf.

    Parameters
    ----------
    confirm_message_received_protobuf : ConfirmMessageReceivedProtobuf
        A callable that takes a `ConfirmMessageReceivedRequest` and returns a
        `ConfirmMessageReceivedResponse`, confirming message receipt.
        This function is typically backed by a gRPC client stub.
    node : Node
        The node making the request.
    run_id : int
        The run ID for the current message.

    Returns
    -------
    Callable[[str], None]
        A wrapper function that takes an object ID and confirms that
        the message has been received.
    """

    def confirm_message_received_fn(object_id: str) -> None:
        request = ConfirmMessageReceivedRequest(
            node=node, run_id=run_id, message_object_id=object_id
        )
        confirm_message_received_protobuf(request)

    return confirm_message_received_fn
