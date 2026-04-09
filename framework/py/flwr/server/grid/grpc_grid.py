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
"""Flower gRPC Grid."""


import time
from time import perf_counter
from collections.abc import Iterable
from logging import DEBUG, ERROR, WARNING
from typing import cast

import grpc

from flwr.app.error import Error
from flwr.common import Message, Metadata, RecordDict, now
from flwr.common.profiling import (
    get_active_profiler,
    get_current_round,
    publish_profile_summary,
    record_network_delivery_metrics_from_messages,
    record_profile_metrics_from_messages,
)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None
from flwr.common.constant import (
    FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES,
    SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
    SUPERLINK_NODE_ID,
    ErrorCode,
    MessageType,
)
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.inflatable import (
    InflatableObject,
    get_all_nested_objects,
    get_object_tree,
    iterate_object_tree,
    no_object_id_recompute,
)
from flwr.common.inflatable_protobuf_utils import (
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)
from flwr.common.inflatable_utils import (
    ObjectIdNotPreregisteredError,
    ObjectUnavailableError,
    inflate_object_from_contents,
    pull_objects,
    push_objects,
)
from flwr.common.logger import log, warn_deprecated_feature
from flwr.common.message import make_message, remove_content_from_message
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import message_from_proto, message_to_proto, run_from_proto
from flwr.common.system_metrics import DiskIoSnapshot, read_disk_io_mb
from flwr.common.typing import Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse  # pylint: disable=E0611
from flwr.proto.serverappio_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    GetNodesResponse,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub  # pylint: disable=E0611

from .grid import Grid

ERROR_MESSAGE_PUSH_MESSAGES_RESOURCE_EXHAUSTED = """

[Grid.push_messages] gRPC error occurred:

The 2GB gRPC limit has been reached. Consider reducing the number of messages pushed
at once, or push messages individually, for example:

> msgs = [msg1, msg2, msg3]
> msg_ids = []
> for msg in msgs:
>     msg_id = grid.push_messages([msg])
>     msg_ids.extend(msg_id)
"""

ERROR_MESSAGE_PULL_MESSAGES_RESOURCE_EXHAUSTED = """

[Grid.pull_messages] gRPC error occurred:

The 2GB gRPC limit has been reached. Consider reducing the number of messages pulled
at once, or pull messages individually, for example:

> msgs_ids = [msg_id1, msg_id2, msg_id3]
> msgs = []
> for msg_id in msg_ids:
>     msg = grid.pull_messages([msg_id])
>     msgs.extend(msg)
"""


class GrpcGrid(Grid):
    """`GrpcGrid` provides an interface to the ServerAppIo API.

    Parameters
    ----------
    serverappio_service_address : str (default: "[::]:9091")
        The address (URL, IPv6, IPv4) of the SuperLink ServerAppIo API service.
    root_certificates : Optional[bytes] (default: None)
        The PEM-encoded root certificates as a byte string.
        If provided, a secure connection using the certificates will be
        established to an SSL-enabled Flower server.
    """

    _deprecation_warning_logged = False

    def __init__(  # pylint: disable=too-many-arguments
        self,
        serverappio_service_address: str = SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        root_certificates: bytes | None = None,
        pull_interval: float = 3.0,
    ) -> None:
        self._addr = serverappio_service_address
        self._cert = root_certificates
        self._run: Run | None = None
        self._grpc_stub: ServerAppIoStub | None = None
        self._channel: grpc.Channel | None = None
        self.node = Node(node_id=SUPERLINK_NODE_ID)
        self._retry_invoker = _make_simple_grpc_retry_invoker()
        self.pull_interval = pull_interval
        super().__init__()

    @property
    def _is_connected(self) -> bool:
        """Check if connected to the ServerAppIo API server."""
        return self._channel is not None

    def _connect(self) -> None:
        """Connect to the ServerAppIo API.

        This will not call GetRun.
        """
        if self._is_connected:
            log(WARNING, "Already connected")
            return
        self._channel = create_channel(
            server_address=self._addr,
            insecure=(self._cert is None),
            root_certificates=self._cert,
        )
        self._channel.subscribe(on_channel_state_change)
        self._grpc_stub = ServerAppIoStub(self._channel)
        _wrap_stub(self._grpc_stub, self._retry_invoker)
        log(DEBUG, "[flwr-serverapp] Connected to %s", self._addr)

    def _disconnect(self) -> None:
        """Disconnect from the ServerAppIo API."""
        if not self._is_connected:
            log(DEBUG, "Already disconnected")
            return
        channel: grpc.Channel = self._channel
        self._channel = None
        self._grpc_stub = None
        channel.close()
        log(DEBUG, "[flwr-serverapp] Disconnected")

    def set_run(self, run_id: int) -> None:
        """Set the run."""
        # Get the run info
        req = GetRunRequest(run_id=run_id)
        res: GetRunResponse = self._stub.GetRun(req)
        if not res.HasField("run"):
            raise RuntimeError(f"Cannot find the run with ID: {run_id}")
        self._run = run_from_proto(res.run)

    @property
    def run(self) -> Run:
        """Run information."""
        return Run(**vars(self._run))

    @property
    def _stub(self) -> ServerAppIoStub:
        """ServerAppIo stub."""
        if not self._is_connected:
            self._connect()
        return cast(ServerAppIoStub, self._grpc_stub)

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.message_id != ""
            and message.metadata.reply_to_message_id == ""
            and message.metadata.ttl > 0
        ):
            raise ValueError(f"Invalid message: {message}")

    def create_message(  # pylint: disable=too-many-arguments,R0917
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.
        """
        if not GrpcGrid._deprecation_warning_logged:
            GrpcGrid._deprecation_warning_logged = True
            warn_deprecated_feature(
                "`Driver.create_message` / `Grid.create_message` is deprecated."
                "Use `Message` constructor instead."
            )
        return Message(content, dst_node_id, message_type, ttl=ttl, group_id=group_id)

    def get_node_ids(self) -> Iterable[int]:
        """Get node IDs."""
        # Call GrpcServerAppIoStub method
        res: GetNodesResponse = self._stub.GetNodes(
            GetNodesRequest(run_id=cast(Run, self._run).run_id)
        )
        return [node.node_id for node in res.nodes]

    def _try_push_messages(
        self,
        run_id: int,
        messages: Iterable[Message],
        *,
        max_concurrent_pushes: int | None = None,
    ) -> list[str]:
        """Push all messages and its associated objects."""
        # Prepare all Messages to be sent in a single request
        proto_messages = []
        object_trees = []
        all_objects: dict[str, InflatableObject] = {}
        for msg in messages:
            proto_messages.append(message_to_proto(remove_content_from_message(msg)))
            all_objects.update(get_all_nested_objects(msg))
            object_trees.append(get_object_tree(msg))
            del msg

        # Call GrpcServerAppIoStub method
        res: PushAppMessagesResponse = self._stub.PushMessages(
            PushAppMessagesRequest(
                messages_list=proto_messages,
                run_id=run_id,
                message_object_trees=object_trees,
            )
        )

        # Push objects
        push_objects(
            all_objects,
            push_object_fn=make_push_object_fn_protobuf(
                push_object_protobuf=self._stub.PushObject,
                node=self.node,
                run_id=run_id,
            ),
            object_ids_to_push=set(res.objects_to_push),
            max_concurrent_pushes=(
                max_concurrent_pushes
                if max_concurrent_pushes is not None
                else FLWR_PRIVATE_MAX_CONCURRENT_OBJ_PUSHES
            ),
        )
        return cast(list[str], res.message_ids)

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        # Construct Messages
        run_id = cast(Run, self._run).run_id
        message_ids: list[str] = []
        if not messages:
            return message_ids
        try:
            with no_object_id_recompute():
                for msg in messages:
                    # Populate metadata
                    msg.metadata.__dict__["_run_id"] = run_id
                    msg.metadata.__dict__["_src_node_id"] = self.node.node_id
                    msg.metadata.__dict__["_message_id"] = msg.object_id
                    # Check message
                    self._check_message(msg)
                # Try pushing messages and their objects.
                # In some deployment environments, preregistered object IDs can
                # become temporarily inconsistent during high-load startup windows.
                # Retry once by re-registering object trees before failing hard.
                try:
                    message_ids = self._try_push_messages(run_id, messages)
                except ObjectIdNotPreregisteredError as err:
                    log(
                        WARNING,
                        "Object preregistration mismatch while pushing messages. "
                        "Retrying once (run_id=%s): %s",
                        run_id,
                        err,
                    )
                    message_ids = self._try_push_messages(
                        run_id,
                        messages,
                        # Force serialized object push on retry to minimize race windows
                        # in object preregistration/object upload under load.
                        max_concurrent_pushes=1,
                    )

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:  # pylint: disable=E1101
                log(ERROR, ERROR_MESSAGE_PUSH_MESSAGES_RESOURCE_EXHAUSTED)
                return []
            raise

        if None in message_ids:
            log(
                WARNING,
                "Not all messages could be pushed to the SuperLink. The returned "
                "list has `None` for those messages (the order is preserved as "
                "passed to `push_messages`). This could be due to a malformed "
                "message.",
            )

        return message_ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        run_id = cast(Run, self._run).run_id
        try:
            # Pull Messages
            res: PullAppMessagesResponse = self._stub.PullMessages(
                PullAppMessagesRequest(
                    message_ids=message_ids,
                    run_id=run_id,
                )
            )
            # Pull Messages from store
            inflated_msgs: list[Message] = []
            for msg_proto, msg_tree in zip(
                res.messages_list, res.message_object_trees, strict=True
            ):
                msg_id = msg_proto.metadata.message_id
                try:
                    all_object_contents = pull_objects(
                        object_ids=[
                            tree.object_id for tree in iterate_object_tree(msg_tree)
                        ],
                        pull_object_fn=make_pull_object_fn_protobuf(
                            pull_object_protobuf=self._stub.PullObject,
                            node=self.node,
                            run_id=run_id,
                        ),
                    )
                except ObjectUnavailableError as e:
                    # An ObjectUnavailableError indicates that the object is not yet
                    # available. If this point has been reached, it means that the
                    # Grid has tried to pull the object for the maximum number of times
                    # or for the maximum time allowed, so we return an inflated message
                    # with an error
                    inflated_msgs.append(
                        make_message(
                            metadata=Metadata(
                                run_id=run_id,
                                message_id="",
                                src_node_id=self.node.node_id,
                                dst_node_id=self.node.node_id,
                                message_type=MessageType.SYSTEM,
                                group_id="",
                                ttl=0,
                                reply_to_message_id=msg_proto.metadata.reply_to_message_id,
                                created_at=now().timestamp(),
                            ),
                            error=Error(
                                code=ErrorCode.MESSAGE_UNAVAILABLE, reason=(str(e))
                            ),
                        )
                    )
                    continue

                # Confirm that the message has been received
                self._stub.ConfirmMessageReceived(
                    ConfirmMessageReceivedRequest(
                        node=self.node, run_id=run_id, message_object_id=msg_id
                    )
                )
                message = cast(
                    Message, inflate_object_from_contents(msg_id, all_object_contents)
                )
                # Network delivery metrics are attached to the lightweight reply message
                # held in LinkState (and serialized in `msg_proto`). Preserve only the
                # downstream timing as internal metadata so strategy reply validation is
                # not affected by extra MetricRecords.
                proto_msg = message_from_proto(msg_proto)
                net_delivery_record = proto_msg.content.metric_records.get(
                    "_flwr_network_delivery"
                )
                if net_delivery_record is not None:
                    downstream_ms = net_delivery_record.get("downstream_ms")
                    if isinstance(downstream_ms, (int, float)):
                        message.metadata.__dict__["_network_downstream_ms"] = float(
                            downstream_ms
                        )
                message.metadata.__dict__["_message_id"] = msg_id
                inflated_msgs.append(message)

            return inflated_msgs

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:  # pylint: disable=E1101
                log(ERROR, ERROR_MESSAGE_PULL_MESSAGES_RESOURCE_EXHAUSTED)
                return []
            raise

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: float | None = None,
    ) -> Iterable[Message]:
        """Push messages to specified node IDs and pull the reply messages.

        This method sends a list of messages to their destination node IDs and then
        waits for the replies. It continues to pull replies until either all replies are
        received or the specified timeout duration is exceeded.
        """
        profiler = get_active_profiler()
        if profiler is None:
            log(DEBUG, "[profiling] No active profiler in GrpcGrid.send_and_receive")
        proc = psutil.Process() if psutil is not None else None
        start = perf_counter() if profiler is not None else None

        def _disk_delta(
            start_snapshot: DiskIoSnapshot | None,
            end_snapshot: DiskIoSnapshot | None,
        ) -> tuple[float | None, float | None, str | None]:
            if (
                start_snapshot is None
                or end_snapshot is None
                or not start_snapshot.source
                or start_snapshot.source != end_snapshot.source
            ):
                return None, None, None
            read_delta = (
                end_snapshot.read_mb - start_snapshot.read_mb
                if end_snapshot.read_mb is not None
                and start_snapshot.read_mb is not None
                else None
            )
            write_delta = (
                end_snapshot.write_mb - start_snapshot.write_mb
                if end_snapshot.write_mb is not None
                and start_snapshot.write_mb is not None
                else None
            )
            return read_delta, write_delta, end_snapshot.source

        overall_mem_start_mb = None
        overall_disk_start = None
        if proc is not None:
            overall_mem_start_mb = proc.memory_info().rss / (1024**2)
            overall_disk_start = read_disk_io_mb(proc)

        # Push messages
        push_start = perf_counter() if profiler is not None else None
        mem_start_mb = None
        disk_start = None
        if proc is not None:
            mem_start_mb = proc.memory_info().rss / (1024**2)
            disk_start = read_disk_io_mb(proc)
        msg_ids = set(self.push_messages(messages))
        if profiler is not None and push_start is not None:
            mem_end_mb = None
            mem_delta_mb = None
            disk_read_mb = None
            disk_write_mb = None
            disk_source = None
            if proc is not None:
                mem_end_mb = proc.memory_info().rss / (1024**2)
                if mem_start_mb is not None:
                    mem_delta_mb = mem_end_mb - mem_start_mb
                disk_end = read_disk_io_mb(proc)
                disk_read_mb, disk_write_mb, disk_source = _disk_delta(
                    disk_start, disk_end
                )
            profiler.record(
                scope="server",
                task="network_downstream",
                round=get_current_round(),
                node_id=None,
                duration_ms=(perf_counter() - push_start) * 1000.0,
                metadata={
                    "expected_replies": len(msg_ids),
                    "memory_start_mb": mem_start_mb,
                    "memory_end_mb": mem_end_mb,
                    "memory_delta_mb": mem_delta_mb,
                    "memory_mb": mem_end_mb,
                    "disk_read_mb": disk_read_mb,
                    "disk_write_mb": disk_write_mb,
                    "disk_source": disk_source,
                },
            )
        del messages
        expected_replies = len(msg_ids)

        # Pull messages
        pull_start = perf_counter() if profiler is not None else None
        mem_start_mb = None
        disk_start = None
        pull_active_ms = 0.0
        wait_sleep_ms = 0.0
        if proc is not None:
            mem_start_mb = proc.memory_info().rss / (1024**2)
            disk_start = read_disk_io_mb(proc)
        end_time = time.time() + (timeout if timeout is not None else 0.0)
        ret: list[Message] = []
        while timeout is None or time.time() < end_time:
            iter_start = perf_counter() if profiler is not None else None
            res_msgs = self.pull_messages(msg_ids)
            if iter_start is not None:
                pull_active_ms += (perf_counter() - iter_start) * 1000.0
            ret.extend(res_msgs)
            msg_ids.difference_update(
                {msg.metadata.reply_to_message_id for msg in res_msgs}
            )
            if profiler is not None and res_msgs:
                record_network_delivery_metrics_from_messages(
                    res_msgs, delivered_at_ms=time.time() * 1000.0
                )
                record_profile_metrics_from_messages(res_msgs)
                publish_profile_summary()
            if len(msg_ids) == 0:
                break
            # Sleep
            sleep_start = perf_counter() if profiler is not None else None
            time.sleep(self.pull_interval)
            if sleep_start is not None:
                wait_sleep_ms += (perf_counter() - sleep_start) * 1000.0
        if profiler is not None and pull_start is not None:
            mem_end_mb = None
            mem_delta_mb = None
            disk_read_mb = None
            disk_write_mb = None
            disk_source = None
            if proc is not None:
                mem_end_mb = proc.memory_info().rss / (1024**2)
                if mem_start_mb is not None:
                    mem_delta_mb = mem_end_mb - mem_start_mb
                disk_end = read_disk_io_mb(proc)
                disk_read_mb, disk_write_mb, disk_source = _disk_delta(
                    disk_start, disk_end
                )
            profiler.record(
                scope="server",
                task="network_upstream",
                round=get_current_round(),
                node_id=None,
                duration_ms=pull_active_ms,
                metadata={
                    "received": len(ret),
                    "wait_sleep_ms": wait_sleep_ms,
                    "memory_start_mb": mem_start_mb,
                    "memory_end_mb": mem_end_mb,
                    "memory_delta_mb": mem_delta_mb,
                    "memory_mb": mem_end_mb,
                    "disk_read_mb": disk_read_mb,
                    "disk_write_mb": disk_write_mb,
                    "disk_source": disk_source,
                },
            )
        if profiler is not None and start is not None:
            duration_ms = (perf_counter() - start) * 1000.0
            mem_end_mb = None
            mem_delta_mb = None
            disk_read_mb = None
            disk_write_mb = None
            disk_source = None
            if proc is not None:
                mem_end_mb = proc.memory_info().rss / (1024**2)
                if overall_mem_start_mb is not None:
                    mem_delta_mb = mem_end_mb - overall_mem_start_mb
                disk_end = read_disk_io_mb(proc)
                disk_read_mb, disk_write_mb, disk_source = _disk_delta(
                    overall_disk_start, disk_end
                )
            profiler.record(
                scope="server",
                task="send_and_receive",
                round=get_current_round(),
                node_id=None,
                duration_ms=max(
                    duration_ms - wait_sleep_ms, 0.0
                ),
                metadata={
                    "expected_replies": expected_replies,
                    "wait_sleep_ms": wait_sleep_ms,
                    "memory_start_mb": overall_mem_start_mb,
                    "memory_end_mb": mem_end_mb,
                    "memory_delta_mb": mem_delta_mb,
                    "memory_mb": mem_end_mb,
                    "disk_read_mb": disk_read_mb,
                    "disk_write_mb": disk_write_mb,
                    "disk_source": disk_source,
                },
            )
            publish_profile_summary()

        return ret

    def close(self) -> None:
        """Disconnect from the SuperLink if connected."""
        # Check if `connect` was called before
        if not self._is_connected:
            return
        # Disconnect
        self._disconnect()
