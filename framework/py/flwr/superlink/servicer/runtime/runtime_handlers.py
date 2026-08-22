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
"""SuperLink Runtime API functions."""

# pylint: disable=unused-argument

from itertools import chain
from logging import DEBUG, ERROR, INFO

from flwr.app import Message
from flwr.common.constant import SUPERLINK_NODE_ID, Status
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullPendingTasksRequest,
    PullPendingTasksResponse,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.server.utils.validator import validate_message
from flwr.supercore import log
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import AUTOMATION_BATCH_LIMIT, TaskType
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    no_object_id_recompute,
)
from flwr.supercore.object_store import NoObjectInStoreError
from flwr.superlink.servicer.control.control_handlers import process_due_automations
from flwr.superlink.servicer.control.control_handlers import (
    start_automation as start_control_automation,
)

RUNTIME_ENDPOINT_UNAVAILABLE_MESSAGE = (
    "Some Runtime API endpoints are only available for Deployment Runtime runs."
)


def pull_pending_tasks(
    request: PullPendingTasksRequest,
    state: LinkState,
) -> PullPendingTasksResponse:
    """Process due automations, then pull pending tasks."""
    log(DEBUG, "Runtime.PullPendingTasks")
    process_due_automations(state, limit=AUTOMATION_BATCH_LIMIT)
    tasks = state.get_tasks(
        statuses=[Status.PENDING],
        order_by="pending_at",
        ascending=True,
        claimable=True,
    )
    return PullPendingTasksResponse(tasks=tasks)


def get_nodes(
    request: GetNodesRequest,
    state: LinkState,
    task: Task,
) -> GetNodesResponse:
    """Get available nodes."""
    log(DEBUG, "Runtime.GetNodes")
    run_id = _get_authenticated_serverapp_run_id(task)
    all_ids: set[int] = state.get_nodes(run_id)
    nodes: list[Node] = [Node(node_id=node_id) for node_id in all_ids]
    return GetNodesResponse(nodes=nodes)


def push_messages(
    request: PushAppMessagesRequest,
    state: LinkState,
    task: Task,
) -> PushAppMessagesResponse:
    """Push a set of Messages."""
    log(DEBUG, "Runtime.PushMessages")
    run_id = _get_authenticated_serverapp_run_id(task)

    _raise_if(
        validation_error=len(request.messages_list) == 0,
        request_name="PushMessages",
        detail="`messages_list` must not be empty",
    )
    session_id = state.start_session(run_id)
    message_ids: list[str] = []
    missing_objects_lists: list[list[str]] = []
    for message_proto, object_tree in zip(
        request.messages_list, request.message_object_trees, strict=True
    ):
        message = message_from_proto(message_proto=message_proto)
        validation_errors = validate_message(message, is_reply_message=False)
        _raise_if(
            validation_error=bool(validation_errors),
            request_name="PushMessages",
            detail=", ".join(validation_errors),
        )
        _raise_if(
            validation_error=run_id != message.metadata.run_id,
            request_name="PushMessages",
            detail="`Message.metadata` has mismatched `run_id`",
        )
        stored, missing_objects = state.store_message_and_object_tree(
            message, object_tree, session_id
        )
        if stored:
            message_ids.append(message.metadata.message_id)
            missing_objects_lists.append(missing_objects)
        else:
            message_ids.append("")

    objects_to_push = list(dict.fromkeys(chain(*missing_objects_lists)))
    return PushAppMessagesResponse(
        message_ids=message_ids,
        objects_to_push=objects_to_push,
        session_id=session_id,
    )


def pull_messages(  # pylint: disable=R0914
    request: PullAppMessagesRequest,
    state: LinkState,
    task: Task,
) -> PullAppMessagesResponse:
    """Pull a set of Messages."""
    log(DEBUG, "Runtime.PullMessages")
    run_id = _get_authenticated_serverapp_run_id(task)
    messages_res: list[Message] = state.get_message_res(
        message_ids=set(request.message_ids)
    )

    store = state.object_store
    for msg_res in messages_res:
        if msg_res.metadata.src_node_id == SUPERLINK_NODE_ID:
            with no_object_id_recompute():
                all_objects = get_all_nested_objects(msg_res)
                store.preregister(run_id, get_object_tree(msg_res))
                for obj_id, obj in all_objects.items():
                    store.put(obj_id, obj.deflate())

    message_ins_ids_to_delete = {
        msg_res.metadata.reply_to_message_id for msg_res in messages_res
    }
    state.delete_messages(message_ins_ids=message_ins_ids_to_delete)

    messages_list = []
    trees = []
    while messages_res:
        msg = messages_res.pop(0)
        if msg.metadata.src_node_id != SUPERLINK_NODE_ID:
            _raise_if(
                validation_error=run_id != msg.metadata.run_id,
                request_name="PullMessages",
                detail="`message.metadata` has mismatched `run_id`",
            )

        try:
            msg_object_id = msg.metadata.message_id
            obj_tree = store.get_object_tree(msg_object_id)
            messages_list.append(message_to_proto(msg))
            trees.append(obj_tree)
        except NoObjectInStoreError as err:
            log(ERROR, err.message)
            state.delete_messages(message_ins_ids={msg_object_id})

    return PullAppMessagesResponse(
        messages_list=messages_list, message_object_trees=trees
    )


def get_connector(
    request: GetConnectorRequest,
    state: LinkState,
    task: Task,
) -> GetConnectorResponse:
    """Return credentials authorized for the authenticated connector task."""
    log(DEBUG, "Runtime.GetConnector")
    if task.type != TaskType.CONNECTOR or not task.connector_ref:
        raise FlowerError(
            ApiErrorCode.RUNTIME_CONNECTOR_CREDENTIALS_NOT_AVAILABLE,
            "Connector credentials are not available to this task.",
        )
    connector_ref = task.connector_ref

    runs = state.get_run_info(run_ids=[task.run_id])
    run = runs[0] if runs else None
    if run is None or not run.flwr_aid:
        raise FlowerError(ApiErrorCode.CONNECTOR_NOT_FOUND, "Connector not found.")

    connector = state.get_connector(
        flwr_aid=run.flwr_aid,
        connector_ref=connector_ref,
    )
    if connector is None:
        raise FlowerError(ApiErrorCode.CONNECTOR_NOT_FOUND, "Connector not found.")

    return GetConnectorResponse(
        connector_ref=connector.connector_ref,
        credentials_json=connector.credentials_json,
        config_json=connector.config_json,
    )


def pull_task_input(
    request: PullTaskInputRequest,
    state: LinkState,
    task: Task,
) -> PullTaskInputResponse:
    """Pull ServerApp process inputs."""
    log(DEBUG, "Runtime.PullTaskInput")
    run_id = task.run_id

    runs = state.get_run_info(run_ids=[run_id])
    run = runs[0] if runs else None
    fab = state.get_fab(run.fab_hash) if run and run.fab_hash else None
    series_context = None
    if run and run.series_id:
        series_context = state.get_run_series_context(run.series_id)
    if run and fab and series_context and state.activate_task(task.task_id):
        series_context.run_id = run_id
        series_context.series_id = run.series_id
        log(INFO, "Started task %d of run %d", task.task_id, run_id)
        return PullTaskInputResponse(
            context=context_to_proto(series_context),
            run=run_to_proto(run),
            fab=fab_to_proto(fab),
            federation_config=state.get_federation_config(run_id),
            task_id=task.task_id,
        )

    raise FlowerError(
        ApiErrorCode.RUNTIME_TASK_START_FAILED,
        f"Failed to start task {task.task_id} of run {run_id}",
    )


def push_task_output(
    request: PushTaskOutputRequest,
    state: LinkState,
    task: Task,
) -> PushTaskOutputResponse:
    """Push ServerApp process outputs."""
    log(DEBUG, "Runtime.PushTaskOutput")
    run_id = task.run_id

    if request.HasField("clientapp_runtime"):
        state.add_clientapp_runtime(run_id, request.clientapp_runtime)

    series_id = None
    output_context = None
    if request.HasField("context"):
        runs = state.get_run_info(run_ids=[run_id])
        run = runs[0] if runs else None
        if run and run.series_id and run.primary_task_id == task.task_id:
            series_id = run.series_id
            output_context = context_from_proto(request.context)

    # Finishing releases an AgentApp's run-series reservation, so persist its
    # context first. Other task types retain the existing finish-then-store order.
    if series_id is not None and output_context is not None:
        if task.type == TaskType.AGENT_APP:
            state.set_run_series_context(series_id, output_context)

    if state.finish_task(
        task.task_id, sub_status=request.sub_status, details=request.details
    ):
        log(INFO, "Finished task %d of run %d", task.task_id, run_id)
        if (
            series_id is not None
            and output_context is not None
            and task.type != TaskType.AGENT_APP
        ):
            state.set_run_series_context(series_id, output_context)
    else:
        log(ERROR, "Failed to finish task %d of run %s", task.task_id, run_id)
    return PushTaskOutputResponse()


def start_automation(
    request: StartAutomationRequest,
    state: LinkState,
    task: Task,
) -> StartAutomationResponse:
    """Start an automation."""
    if task.type not in (TaskType.AGENT_APP, TaskType.SERVER_APP):
        raise FlowerError(
            ApiErrorCode.RUNTIME_AUTOMATION_CREATION_NOT_ALLOWED,
            "Only AgentApp and ServerApp tasks can create automations.",
        )

    run = state.get_run_info(run_ids=[task.run_id])[0]
    del request.start_run_request.connector_refs[:]
    request.start_run_request.connector_refs.extend(
        state.get_run_connector_refs(run_id=run.run_id)
    )
    return start_control_automation(
        request,
        AccountInfo(
            flwr_aid=run.flwr_aid,
            account_name=run.account_name,
        ),
        state,
    )


def push_object(
    request: PushObjectRequest,
    state: LinkState,
    task: Task,
) -> PushObjectResponse:
    """Push an object to the ObjectStore."""
    log(DEBUG, "Runtime.PushObject")
    run_id = _get_authenticated_serverapp_run_id(task)
    if request.node.node_id != SUPERLINK_NODE_ID:
        raise FlowerError(
            ApiErrorCode.RUNTIME_UNEXPECTED_NODE_ID, "Unexpected node ID."
        )
    stored = state.store_object(
        run_id,
        request.session_id,
        request.object_id,
        request.object_content,
    )
    return PushObjectResponse(stored=stored)


def pull_object(
    request: PullObjectRequest,
    state: LinkState,
    task: Task,
) -> PullObjectResponse:
    """Pull an object from the ObjectStore."""
    log(DEBUG, "Runtime.PullObject")
    run_id = _get_authenticated_serverapp_run_id(task)
    if request.node.node_id != SUPERLINK_NODE_ID:
        raise FlowerError(
            ApiErrorCode.RUNTIME_UNEXPECTED_NODE_ID, "Unexpected node ID."
        )

    content = state.get_object(run_id, request.object_id)
    if content is not None:
        return PullObjectResponse(
            object_found=True,
            object_available=content != b"",
            object_content=content,
        )
    return PullObjectResponse(object_found=False, object_available=False)


def confirm_message_received(
    request: ConfirmMessageReceivedRequest,
    state: LinkState,
    task: Task,
) -> ConfirmMessageReceivedResponse:
    """Confirm message received."""
    log(DEBUG, "Runtime.ConfirmMessageReceived")
    _ = _get_authenticated_serverapp_run_id(task)
    state.object_store.delete(request.message_object_id)
    return ConfirmMessageReceivedResponse()


def _get_authenticated_serverapp_run_id(task: Task) -> int:
    """Return the authenticated run ID if it can use these Runtime endpoints."""
    if task.type != TaskType.SERVER_APP:
        raise FlowerError(
            ApiErrorCode.RUNTIME_ENDPOINT_UNAVAILABLE,
            RUNTIME_ENDPOINT_UNAVAILABLE_MESSAGE,
        )
    return task.run_id


def _raise_if(validation_error: bool, request_name: str, detail: str) -> None:
    """Raise a `ValueError` with a detailed message if a validation error occurs."""
    if validation_error:
        raise ValueError(f"Malformed {request_name}: {detail}")
