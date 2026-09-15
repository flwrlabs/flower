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
"""Flower ClientApp process."""

from logging import DEBUG, ERROR, INFO
from time import perf_counter, time

import grpc

from flwr.app.error import Error
from flwr.cli.install import install_from_fab
from flwr.clientapp.client_app import ClientApp, LoadClientAppError
from flwr.clientapp.utils import get_load_client_app_fn
from flwr.common import Context, Message
from flwr.common.config import get_flwr_dir
from flwr.common.constant import ErrorCode
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.inflatable import (
    get_all_nested_objects,
    get_object_tree,
    iterate_object_trees_breadth_first,
    no_object_id_recompute,
)
from flwr.common.inflatable_protobuf_utils import (
    make_confirm_message_received_fn_protobuf,
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)
from flwr.common.inflatable_utils import pull_and_inflate_object_from_tree, push_objects
from flwr.common.logger import log
from flwr.common.message import remove_content_from_message
from flwr.common.record import ConfigRecord, MetricRecord
from flwr.common.retry_invoker import _make_simple_grpc_retry_invoker, _wrap_stub
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    message_to_proto,
    run_from_proto,
)
from flwr.common.telemetry import EventType, event
from flwr.common.typing import Fab, Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
)
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.heartbeat import HeartbeatSender, make_app_heartbeat_fn_grpc
from flwr.supercore.utils import mask_string


def run_clientapp(  # pylint: disable=R0913, R0914, R0917
    clientappio_api_address: str,
    token: str,
    flwr_dir: str | None = None,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
) -> None:
    """Run Flower ClientApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    event(EventType.FLWR_CLIENTAPP_RUN_ENTER)

    channel = create_channel(
        server_address=clientappio_api_address,
        insecure=(certificates is None),
        root_certificates=certificates,
    )
    channel.subscribe(on_channel_state_change)
    heartbeat_channel = create_channel(
        server_address=clientappio_api_address,
        insecure=(certificates is None),
        root_certificates=certificates,
    )
    heartbeat_sender = None

    def on_exit() -> None:
        if heartbeat_sender is not None and heartbeat_sender.is_running:
            heartbeat_sender.stop()
        heartbeat_channel.close()
        channel.close()

    register_signal_handlers(
        event_type=EventType.FLWR_CLIENTAPP_RUN_LEAVE,
        exit_handlers=[on_exit],
    )

    # Resolve directory where FABs are installed
    flwr_dir_ = get_flwr_dir(flwr_dir)
    try:
        stub = ClientAppIoStub(channel)
        _wrap_stub(stub, _make_simple_grpc_retry_invoker())

        # Keep heartbeat traffic independent from large object transfers on the main
        # AppIO channel so transport flow control cannot starve lease renewal.
        heartbeat_stub = ClientAppIoStub(heartbeat_channel)
        _wrap_stub(heartbeat_stub, _make_simple_grpc_retry_invoker())
        heartbeat_sender = HeartbeatSender(
            make_app_heartbeat_fn_grpc(heartbeat_stub, token)
        )
        heartbeat_sender.start()
        log(
            DEBUG,
            "[flwr-clientapp] Heartbeat sender started for token %s",
            mask_string(token),
        )

        # Pull Message, Context, Run and (optional) FAB from SuperNode
        inputs_start = perf_counter()
        message, context, run, fab = pull_clientappinputs(stub=stub, token=token)
        log(
            DEBUG,
            "[flwr-clientapp] Inputs ready: token=%s message_id=%s elapsed_ms=%.1f",
            mask_string(token),
            mask_string(message.metadata.message_id),
            (perf_counter() - inputs_start) * 1000.0,
        )

        try:

            # Install FAB, if provided
            if fab:
                log(DEBUG, "[flwr-clientapp] Start FAB installation.")
                install_from_fab(fab.content, flwr_dir=flwr_dir_, skip_prompt=True)

            load_client_app_fn = get_load_client_app_fn(
                default_app_ref="",
                app_path=None,
                multi_app=True,
                flwr_dir=str(flwr_dir_),
            )

            # Load ClientApp
            log(DEBUG, "[flwr-clientapp] Start `ClientApp` Loading.")
            client_app: ClientApp = load_client_app_fn(
                run.fab_id, run.fab_version, fab.hash_str if fab else ""
            )

            # Execute ClientApp
            handler_start = perf_counter()
            reply_message = client_app(message=message, context=context)
            log(
                DEBUG,
                "[flwr-clientapp] ClientApp handler complete: token=%s "
                "message_id=%s elapsed_ms=%.1f error=%s",
                mask_string(token),
                mask_string(message.metadata.message_id),
                (perf_counter() - handler_start) * 1000.0,
                reply_message.has_error(),
            )

        except Exception as ex:  # pylint: disable=broad-exception-caught
            # Don't update/change NodeState

            e_code = ErrorCode.CLIENT_APP_RAISED_EXCEPTION
            # Ex fmt: "<class 'ZeroDivisionError'>:<'division by zero'>"
            reason = str(type(ex)) + ":<'" + str(ex) + "'>"
            exc_entity = "ClientApp"
            if isinstance(ex, LoadClientAppError):
                reason = "An exception was raised when attempting to load `ClientApp`"
                e_code = ErrorCode.LOAD_CLIENT_APP_EXCEPTION

            log(ERROR, "%s raised an exception", exc_entity, exc_info=ex)

            # Create error message
            reply_message = Message(Error(code=e_code, reason=reason), reply_to=message)

        # Push Message and Context to SuperNode
        output_start = perf_counter()
        _ = push_clientappoutputs(
            stub=stub, token=token, message=reply_message, context=context
        )
        log(
            DEBUG,
            "[flwr-clientapp] Outputs complete: token=%s reply_to=%s elapsed_ms=%.1f",
            mask_string(token),
            mask_string(reply_message.metadata.reply_to_message_id),
            (perf_counter() - output_start) * 1000.0,
        )

    except grpc.RpcError as e:
        log(ERROR, "GRPC error occurred: %s", str(e))

    flwr_exit(
        code=ExitCode.SUCCESS,
        event_type=EventType.FLWR_CLIENTAPP_RUN_LEAVE,
    )


def pull_clientappinputs(
    stub: ClientAppIoStub, token: str
) -> tuple[Message, Context, Run, Fab | None]:
    """Pull ClientAppInputs from SuperNode."""
    masked_token = mask_string(token)
    log(INFO, "[flwr-clientapp] Pull `ClientAppInputs` for token %s", masked_token)
    inputs_started_at_ms = time() * 1000.0
    inputs_start = perf_counter()
    input_bytes = 0
    try:
        # Pull Context, Run and (optional) FAB
        res: PullAppInputsResponse = stub.PullClientAppInputs(
            PullAppInputsRequest(token=token)
        )
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab) if res.fab else None
        log(
            DEBUG,
            "[flwr-clientapp] PullClientAppInputs complete: token=%s run_id=%s "
            "fab_bytes=%s elapsed_ms=%.1f",
            masked_token,
            context.run_id,
            len(fab.content) if fab else 0,
            (perf_counter() - inputs_start) * 1000.0,
        )

        # Pull and inflate the message
        message_start = perf_counter()
        pull_msg_res: PullAppMessagesResponse = stub.PullMessage(
            PullAppMessagesRequest(token=token)
        )
        run_id = context.run_id
        node = Node(node_id=context.node_id)
        object_tree = pull_msg_res.message_object_trees[0]
        input_bytes += len(res.SerializeToString()) + len(
            pull_msg_res.SerializeToString()
        )
        pull_object_fn = make_pull_object_fn_protobuf(stub.PullObject, node, run_id)

        def tracked_pull_object(object_id: str) -> bytes:
            nonlocal input_bytes
            content = pull_object_fn(object_id)
            input_bytes += len(content)
            return content

        confirm_message_received = make_confirm_message_received_fn_protobuf(
            stub.ConfirmMessageReceived, node, run_id
        )

        def tracked_confirm_message_received(object_id: str) -> None:
            confirm_message_received(object_id)

        message = pull_and_inflate_object_from_tree(
            object_tree,
            tracked_pull_object,
            tracked_confirm_message_received,
            return_type=Message,
        )
        log(
            DEBUG,
            "[flwr-clientapp] PullMessage and inflate complete: token=%s run_id=%s "
            "message_id=%s elapsed_ms=%.1f",
            masked_token,
            run_id,
            mask_string(object_tree.object_id),
            (perf_counter() - message_start) * 1000.0,
        )

        # Set the message ID
        # The deflated message doesn't contain the message_id (its own object_id)
        message.metadata.__dict__["_message_id"] = object_tree.object_id
        if bool(context.run_config.get("profile.enabled", False)):
            context.__dict__["_transport_input_profile"] = {
                "instruction_id": object_tree.object_id,
                "group_id": message.metadata.group_id,
                "timestamp_ms": inputs_started_at_ms,
                "duration_ms": (perf_counter() - inputs_start) * 1000.0,
                "network_bytes": input_bytes,
            }
        return message, context, run, fab
    except grpc.RpcError as e:
        log(ERROR, "[PullClientAppInputs] gRPC error occurred: %s", str(e))
        raise e


def push_clientappoutputs(
    stub: ClientAppIoStub, token: str, message: Message, context: Context
) -> PushAppOutputsResponse:
    """Push ClientAppOutputs to SuperNode."""
    masked_token = mask_string(token)
    log(INFO, "[flwr-clientapp] Push `ClientAppOutputs` for token %s", masked_token)
    output_start = perf_counter()
    output_started_at_ms = time() * 1000.0
    if message.has_content():
        message.content.metric_records["_flwr_full_path"] = MetricRecord(
            {"upstream_started_at_ms": output_started_at_ms}
        )
    # Set message ID
    message.metadata.__dict__["_message_id"] = message.object_id
    proto_message = message_to_proto(remove_content_from_message(message))

    try:

        with no_object_id_recompute():
            # Get object tree and all objects to push
            object_tree = get_object_tree(message)

            # Push Message
            # This is temporary. The message should not contain its content
            output_wall_start = perf_counter()
            push_msg_res = stub.PushMessage(
                PushAppMessagesRequest(
                    token=token,
                    messages_list=[proto_message],
                    message_object_trees=[object_tree],
                )
            )
            del proto_message
            log(
                DEBUG,
                "[flwr-clientapp] PushMessage complete: token=%s reply_to=%s "
                "objects_to_push=%s elapsed_ms=%.1f",
                masked_token,
                mask_string(message.metadata.reply_to_message_id),
                len(push_msg_res.objects_to_push),
                (perf_counter() - output_start) * 1000.0,
            )

            # Retrieve the object IDs to push
            object_ids_to_push = set(push_msg_res.objects_to_push)

            # Push all objects
            all_objects = get_all_nested_objects(message)
            object_push_order = [
                tree.object_id
                for tree in iterate_object_trees_breadth_first([object_tree])
            ]
            all_objects = {
                object_id: all_objects[object_id] for object_id in object_push_order
            }
            del message
            output_bytes = 0
            push_object_fn = make_push_object_fn_protobuf(
                stub.PushObject,
                Node(node_id=context.node_id),
                run_id=context.run_id,
            )

            def tracked_push_object(object_id: str, content: bytes) -> None:
                nonlocal output_bytes
                output_bytes += len(content)
                push_object_fn(object_id, content)

            push_objects(
                all_objects,
                tracked_push_object,
                object_ids_to_push=object_ids_to_push,
            )
            log(
                INFO,
                "[flwr-clientapp] Reply objects pushed: token=%s objects=%s "
                "elapsed_ms=%.1f",
                masked_token,
                len(object_ids_to_push),
                (perf_counter() - output_start) * 1000.0,
            )

        input_profile = context.__dict__.pop("_transport_input_profile", None)
        if input_profile is not None:
            context.state.config_records["_flwr_transport_profile"] = ConfigRecord(
                {
                    "instruction_id": str(input_profile.get("instruction_id", "")),
                    "group_id": str(input_profile.get("group_id", "")),
                    "input_timestamp_ms": float(input_profile.get("timestamp_ms", 0.0)),
                    "input_duration_ms": float(input_profile.get("duration_ms", 0.0)),
                    "input_network_bytes": int(input_profile.get("network_bytes", 0)),
                    "output_timestamp_ms": output_started_at_ms,
                    "output_duration_ms": (perf_counter() - output_wall_start) * 1000.0,
                    "output_network_bytes": output_bytes,
                }
            )
        proto_context = context_to_proto(context)
        context.state.config_records.pop("_flwr_transport_profile", None)

        # Push Context and the internal transport measurements.
        res: PushAppOutputsResponse = stub.PushClientAppOutputs(
            PushAppOutputsRequest(token=token, context=proto_context)
        )
        log(
            INFO,
            "[flwr-clientapp] PushClientAppOutputs complete: token=%s elapsed_ms=%.1f",
            masked_token,
            (perf_counter() - output_start) * 1000.0,
        )
        return res
    except grpc.RpcError as e:
        log(ERROR, "[PushClientAppOutputs] gRPC error occurred: %s", str(e))
        raise e
