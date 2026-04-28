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
"""Contextmanager for a gRPC streaming channel to the Flower server."""


import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue
from typing import cast

from cryptography.hazmat.primitives.asymmetric import ec

from flwr.app.message_type import MessageType
from flwr.common import (
    DEFAULT_TTL,
    GRPC_MAX_MESSAGE_LENGTH,
    ConfigRecord,
    Message,
    Metadata,
    RecordDict,
    now,
)
from flwr.common import recorddict_compat as compat
from flwr.common import serde
from flwr.common.constant import MessageTypeLegacy
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.message import make_message
from flwr.common.retry_invoker import RetryInvoker
from flwr.common.typing import Fab, Run
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    Reason,
    ServerMessage,
)
from flwr.proto.transport_pb2_grpc import FlowerServiceStub  # pylint: disable=E0611


@contextmanager
def grpc_connection(  # pylint: disable=R0913,R0915,too-many-positional-arguments
    server_address: str,
    insecure: bool,
    retry_invoker: RetryInvoker,  # pylint: disable=unused-argument
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: bytes | str | None = None,
    authentication_keys: (
        tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey] | None
    ) = None,
) -> Iterator[
    tuple[
        Callable[[], Message | None],
        Callable[[Message], None],
        Callable[[], int | None] | None,
        Callable[[], None] | None,
        Callable[[int], Run] | None,
        Callable[[str, int], Fab] | None,
    ]
]:
    """Establish a gRPC connection to a gRPC server.

    Parameters
    ----------
    server_address : str
        The IPv4 or IPv6 address of the server. If the Flower server runs on the same
        machine on port 8080, then `server_address` would be `"0.0.0.0:8080"` or
        `"[::]:8080"`.
    insecure : bool
        Starts an insecure gRPC connection when True. Enables HTTPS connection
        when False, using system certificates if `root_certificates` is None.
    retry_invoker: RetryInvoker
        Unused argument present for compatibilty.
    max_message_length : int
        The maximum length of gRPC messages that can be exchanged with the Flower
        server. The default should be sufficient for most models. Users who train
        very large models might need to increase this value. Note that the Flower
        server needs to be started with the same value
        (see `flwr.server.start_server`), otherwise it will not know about the
        increased limit and block larger messages.
        (default: 536_870_912, this equals 512MB)
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string or a path string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    authentication_keys : Optional[Tuple[PrivateKey, PublicKey]] (default: None)
        SuperNode authentication is not supported for this transport type.

    Returns
    -------
    receive, send : Callable, Callable

    Examples
    --------
    Establishing a TLS-enabled connection to the server::

        from pathlib import Path
        with grpc_connection(
            server_address,
            max_message_length=max_message_length,
            root_certificates=Path("/crts/root.pem").read_bytes(),
        ) as conn:
            receive, send = conn
            server_message = receive()
            # do something here
            send(client_message)
    """
    if isinstance(root_certificates, str):
        root_certificates = Path(root_certificates).read_bytes()
    if authentication_keys is not None:
        log(ERROR, "SuperNode authentication is not supported for this transport type.")

    channel = create_channel(
        server_address=server_address,
        insecure=insecure,
        root_certificates=root_certificates,
        max_message_length=max_message_length,
    )
    channel.subscribe(on_channel_state_change)

    queue: Queue[ClientMessage] = Queue(  # pylint: disable=unsubscriptable-object
        maxsize=1
    )
    stub = FlowerServiceStub(channel)

    server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))

    def receive() -> Message:
        # Receive ServerMessage proto
        proto = next(server_message_iterator)

        # ServerMessage proto --> *Ins --> RecordDict
        field = proto.WhichOneof("msg")
        message_type = ""
        if field == "get_properties_ins":
            recorddict = compat.getpropertiesins_to_recorddict(
                serde.get_properties_ins_from_proto(proto.get_properties_ins)
            )
            message_type = MessageTypeLegacy.GET_PROPERTIES
        elif field == "get_parameters_ins":
            recorddict = compat.getparametersins_to_recorddict(
                serde.get_parameters_ins_from_proto(proto.get_parameters_ins)
            )
            message_type = MessageTypeLegacy.GET_PARAMETERS
        elif field == "fit_ins":
            recorddict = compat.fitins_to_recorddict(
                serde.fit_ins_from_proto(proto.fit_ins), False
            )
            message_type = MessageType.TRAIN
        elif field == "evaluate_ins":
            recorddict = compat.evaluateins_to_recorddict(
                serde.evaluate_ins_from_proto(proto.evaluate_ins), False
            )
            message_type = MessageType.EVALUATE
        elif field == "reconnect_ins":
            recorddict = RecordDict()
            recorddict.config_records["config"] = ConfigRecord(
                {"seconds": proto.reconnect_ins.seconds}
            )
            message_type = "reconnect"
        else:
            raise ValueError(
                "Unsupported instruction in ServerMessage, "
                "cannot deserialize from ProtoBuf"
            )

        # Construct Message
        return make_message(
            metadata=Metadata(
                run_id=0,
                message_id=str(uuid.uuid4()),
                src_node_id=0,
                dst_node_id=0,
                reply_to_message_id="",
                group_id="",
                created_at=now().timestamp(),
                ttl=DEFAULT_TTL,
                message_type=message_type,
            ),
            content=recorddict,
        )

    def send(message: Message) -> None:
        # Retrieve RecordDict and message_type
        recorddict = message.content
        message_type = message.metadata.message_type

        # RecordDict --> *Res --> *Res proto -> ClientMessage proto
        if message_type == MessageTypeLegacy.GET_PROPERTIES:
            getpropres = compat.recorddict_to_getpropertiesres(recorddict)
            msg_proto = ClientMessage(
                get_properties_res=serde.get_properties_res_to_proto(getpropres)
            )
        elif message_type == MessageTypeLegacy.GET_PARAMETERS:
            getparamres = compat.recorddict_to_getparametersres(recorddict, False)
            msg_proto = ClientMessage(
                get_parameters_res=serde.get_parameters_res_to_proto(getparamres)
            )
        elif message_type == MessageType.TRAIN:
            fitres = compat.recorddict_to_fitres(recorddict, False)
            msg_proto = ClientMessage(fit_res=serde.fit_res_to_proto(fitres))
        elif message_type == MessageType.EVALUATE:
            evalres = compat.recorddict_to_evaluateres(recorddict)
            msg_proto = ClientMessage(evaluate_res=serde.evaluate_res_to_proto(evalres))
        elif message_type == "reconnect":
            reason = cast(
                Reason.ValueType, recorddict.config_records["config"]["reason"]
            )
            msg_proto = ClientMessage(
                disconnect_res=ClientMessage.DisconnectRes(reason=reason)
            )
        else:
            raise ValueError(f"Invalid message type: {message_type}")

        # Send ClientMessage proto
        return queue.put(msg_proto, block=False)

    try:
        # Yield methods
        yield (receive, send, None, None, None, None)
    finally:
        # Make sure to have a final
        channel.close()
        log(DEBUG, "gRPC channel closed")
