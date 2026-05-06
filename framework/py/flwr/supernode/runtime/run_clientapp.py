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


from logging import DEBUG, ERROR

import grpc

from flwr.app.error import Error
from flwr.cli.install import install_from_fab
from flwr.clientapp.client_app import ClientApp, LoadClientAppError
from flwr.clientapp.utils import get_load_client_app_fn
from flwr.common import Context, Message
from flwr.common.config import get_project_dir
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL, ErrorCode, SubStatus
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log
from flwr.common.message import remove_content_from_message
from flwr.common.retry_invoker import make_simple_grpc_retry_invoker, wrap_stub
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
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    no_object_id_recompute,
)
from flwr.supercore.inflatable.inflatable_protobuf_utils import (
    make_confirm_message_received_fn_protobuf,
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)
from flwr.supercore.inflatable.inflatable_utils import (
    pull_and_inflate_object_from_tree,
    push_objects,
)
from flwr.supercore.interceptors import AppIoTokenClientInterceptor
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
    install_app_dependencies,
)


def run_clientapp(  # pylint: disable=R0913, R0914, R0915, R0917
    clientappio_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower ClientApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    event(EventType.FLWR_CLIENTAPP_RUN_ENTER)

    channel = create_channel(
        server_address=clientappio_api_address,
        insecure=insecure,
        root_certificates=certificates,
        interceptors=[AppIoTokenClientInterceptor(token)],
    )
    channel.subscribe(on_channel_state_change)
    heartbeat_sender = None
    runtime_env_dir = None

    def on_exit() -> None:
        if heartbeat_sender is not None and heartbeat_sender.is_running:
            heartbeat_sender.stop()
        channel.close()
        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_CLIENTAPP_RUN_LEAVE,
        exit_handlers=[on_exit],
    )

    try:
        stub = ClientAppIoStub(channel)
        wrap_stub(stub, make_simple_grpc_retry_invoker())

        # Start app heartbeat
        heartbeat_sender = HeartbeatSender(make_app_heartbeat_fn_grpc(stub, token))
        heartbeat_sender.start()

        # Pull Message, Context, Run and FAB from SuperNode
        message, context, run, fab = pull_appinputs(stub)
        reply_message: Message | None = None
        sub_status = SubStatus.FAILED
        details = "ClientApp task failed due to unknown reason"

        try:

            # Install FAB
            log(DEBUG, "[flwr-clientapp] Start FAB installation.")
            install_from_fab(fab.content, skip_prompt=True)

            app_path = get_project_dir(run.fab_id, run.fab_version, fab.hash_str)
            if runtime_dependency_install:
                log(DEBUG, "[flwr-clientapp] Installing app dependencies.")
                runtime_env_dir = install_app_dependencies(
                    app_path,
                    launch_id=token,
                    run_id=run.run_id,
                    index_context={
                        "component": "clientapp",
                        "project_dir": str(app_path),
                        "run_id": run.run_id,
                        "launch_id": token,
                        "fab_id": run.fab_id,
                        "fab_version": run.fab_version,
                        "fab_hash": fab.hash_str,
                    },
                )
            else:
                log(
                    DEBUG,
                    "[flwr-clientapp] Runtime dependency installation is disabled.",
                )

            load_client_app_fn = get_load_client_app_fn(
                default_app_ref="",
                app_path=None,
                multi_app=True,
            )

            # Load ClientApp
            log(DEBUG, "[flwr-clientapp] Start `ClientApp` Loading.")
            client_app: ClientApp = load_client_app_fn(
                run.fab_id, run.fab_version, fab.hash_str
            )

            # Execute ClientApp
            reply_message = client_app(message=message, context=context)
            sub_status = SubStatus.COMPLETED
            details = ""

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

            sub_status = SubStatus.FAILED
            details = reason
        finally:
            if reply_message is None:
                reply_message = Message(
                    Error(
                        code=ErrorCode.CLIENT_APP_CRASHED,
                        reason=details,
                    ),
                    reply_to=message,
                )

            push_appoutputs(
                stub=stub,
                message=reply_message,
                context=context,
                sub_status=sub_status,
                details=details,
            )

    except grpc.RpcError as e:
        log(ERROR, "GRPC error occurred: %s", str(e))

    flwr_exit(
        code=ExitCode.SUCCESS,
        event_type=EventType.FLWR_CLIENTAPP_RUN_LEAVE,
    )


def pull_appinputs(stub: ClientAppIoStub) -> tuple[Message, Context, Run, Fab]:
    """Pull AppInputs from SuperNode."""
    try:
        # Pull Context, Run and FAB
        res: PullAppInputsResponse = stub.PullAppInputs(PullAppInputsRequest())
        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab)

        # Pull and inflate the message
        pull_msg_res: PullAppMessagesResponse = stub.PullMessage(
            PullAppMessagesRequest()
        )
        run_id = context.run_id
        node = Node(node_id=context.node_id)
        object_tree = pull_msg_res.message_object_trees[0]
        message = pull_and_inflate_object_from_tree(
            object_tree,
            make_pull_object_fn_protobuf(stub.PullObject, node, run_id),
            make_confirm_message_received_fn_protobuf(
                stub.ConfirmMessageReceived, node, run_id
            ),
            return_type=Message,
        )

        # Set the message ID
        # The deflated message doesn't contain the message_id (its own object_id)
        message.metadata.__dict__["_message_id"] = object_tree.object_id
        return message, context, run, fab
    except grpc.RpcError as e:
        log(ERROR, "[PullAppInputs] gRPC error occurred: %s", str(e))
        raise e


def push_appoutputs(  # pylint: disable=R0913, R0917
    stub: ClientAppIoStub,
    message: Message,
    context: Context,
    sub_status: str,
    details: str,
) -> PushAppOutputsResponse:
    """Push AppOutputs to SuperNode."""
    # Set message ID
    message.metadata.__dict__["_message_id"] = message.object_id
    proto_message = message_to_proto(remove_content_from_message(message))
    proto_context = context_to_proto(context)

    try:

        with no_object_id_recompute():
            # Get object tree and all objects to push
            object_tree = get_object_tree(message)

            # Push Message
            # This is temporary. The message should not contain its content
            push_msg_res = stub.PushMessage(
                PushAppMessagesRequest(
                    messages_list=[proto_message], message_object_trees=[object_tree]
                )
            )
            del proto_message

            # Retrieve the object IDs to push
            object_ids_to_push = set(push_msg_res.objects_to_push)

            # Push all objects
            all_objects = get_all_nested_objects(message)
            del message
            push_objects(
                all_objects,
                make_push_object_fn_protobuf(
                    stub.PushObject,
                    Node(node_id=context.node_id),
                    run_id=context.run_id,
                ),
                object_ids_to_push=object_ids_to_push,
            )

        # Push Context
        res: PushAppOutputsResponse = stub.PushAppOutputs(
            PushAppOutputsRequest(
                context=proto_context, sub_status=sub_status, details=details
            )
        )
        return res
    except grpc.RpcError as e:
        log(ERROR, "[PushAppOutputs] gRPC error occurred: %s", str(e))
        raise e
