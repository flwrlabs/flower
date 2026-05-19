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
"""Flower AgentApp process."""


from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from logging import ERROR
from pathlib import Path
from queue import Queue
from typing import Any, Protocol, cast

import grpc

from flwr.agentapp import AgentApp
from flwr.agentapp.session import AgentSession
from flwr.agentapp.utils import get_load_agent_app_fn
from flwr.cli.install import install_from_fab
from flwr.common import Context, EventType
from flwr.common.config import get_project_dir
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL, SubStatus
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.logger import log, stop_log_uploader
from flwr.common.retry_invoker import make_simple_grpc_retry_invoker, wrap_stub
from flwr.common.serde import context_from_proto, fab_from_proto, run_from_proto
from flwr.common.typing import Fab, Run
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PushTaskOutputRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_grpc
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)
from flwr.supercore.superexec.dependency_installer import (
    cleanup_app_runtime_environment,
    install_app_dependencies,
)
from flwr.supercore.task_message import JsonObject, JsonValue

AGENT_STARTED_EVENT = "agent.started"
AGENT_COMPLETED_EVENT = "agent.completed"
AGENT_FAILED_EVENT = "agent.failed"

LoadAgentAppFn = Callable[[str, str, str], AgentApp]


class ServerAppIoAgentStub(Protocol):
    """Subset of ServerAppIo RPCs used by the AgentApp executor."""

    def PullTaskInput(self, request: PullTaskInputRequest) -> Any:
        """Pull task input."""

    def PushTaskOutput(self, request: PushTaskOutputRequest) -> Any:
        """Push task output."""


@dataclass(frozen=True)
class AgentAppTaskInput:
    """Decoded AgentApp task input."""

    task_id: int
    context: Context
    run: Run
    fab: Fab


def run_agentapp(  # pylint: disable=R0913,R0917
    serverappio_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower AgentApp process."""
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    channel = create_channel(
        server_address=serverappio_api_address,
        insecure=certificates is None,
        root_certificates=certificates,
        interceptors=[
            RuntimeVersionClientInterceptor(component_name="flwr-agentapp"),
            AppIoTokenClientInterceptor(token),
        ],
    )
    channel.subscribe(on_channel_state_change)

    heartbeat_sender: HeartbeatSender | None = None
    log_uploader = None
    runtime_env_dir: Path | None = None
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        if heartbeat_sender is not None and heartbeat_sender.is_running:
            heartbeat_sender.stop()
        channel.close()
        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)
        cleanup_app_runtime_environment(runtime_env_dir)

    register_signal_handlers(
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        exit_message="Run stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        stub = ServerAppIoStub(channel)
        wrap_stub(stub, make_simple_grpc_retry_invoker())
        heartbeat_sender = HeartbeatSender(make_task_heartbeat_fn_grpc(stub))
        heartbeat_sender.start()

        task_input = _pull_task_input(stub)

        def load_agent_app(fab_id: str, fab_version: str, fab_hash: str) -> AgentApp:
            nonlocal runtime_env_dir
            install_from_fab(task_input.fab.content, skip_prompt=True)
            app_path = get_project_dir(fab_id, fab_version, fab_hash)
            if runtime_dependency_install:
                runtime_env_dir = install_app_dependencies(
                    app_path,
                    launch_id=token,
                    run_id=task_input.run.run_id,
                    index_context={
                        "component": "agentapp",
                        "project_dir": str(app_path),
                        "run_id": task_input.run.run_id,
                        "launch_id": token,
                        "fab_id": fab_id,
                        "fab_version": fab_version,
                        "fab_hash": fab_hash,
                    },
                )

            return get_load_agent_app_fn(
                default_app_ref="",
                app_path=None,
                multi_app=True,
            )(fab_id, fab_version, fab_hash)

        if not _run_agentapp_task(stub, task_input, load_agent_app):
            exit_code = ExitCode.SERVERAPP_EXCEPTION

    except grpc.RpcError as err:
        log(ERROR, "gRPC error occurred: %s", str(err))
        exit_code = ExitCode.CLIENTAPP_COMMUNICATION_ERROR
    except Exception as err:  # pylint: disable=broad-exception-caught
        log(ERROR, "`flwr-agentapp` raised an exception", exc_info=err)
        exit_code = ExitCode.SERVERAPP_EXCEPTION

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
    )


def _pull_task_input(stub: ServerAppIoAgentStub) -> AgentAppTaskInput:
    """Pull and decode AgentApp task input."""
    response = stub.PullTaskInput(PullTaskInputRequest())
    task_id = int(response.task_id)
    if task_id <= 0:
        raise ValueError("AgentApp task input did not include a valid task_id.")
    return AgentAppTaskInput(
        task_id=task_id,
        context=context_from_proto(response.context),
        run=run_from_proto(response.run),
        fab=fab_from_proto(response.fab),
    )


def _run_agentapp_task(
    stub: ServerAppIoAgentStub,
    task_input: AgentAppTaskInput,
    load_agent_app: LoadAgentAppFn,
) -> bool:
    """Run one AgentApp task and push task output."""
    sub_status = SubStatus.FAILED
    details = "AgentApp task failed due to unknown reason."
    session: AgentSession | None = None
    try:
        session = AgentSession.from_context(
            stub=stub,
            task_id=task_input.task_id,
            context=task_input.context,
            run=task_input.run,
        )
        session.emit_event(AGENT_STARTED_EVENT, {})
        agent_app = load_agent_app(
            task_input.run.fab_id,
            task_input.run.fab_version,
            task_input.fab.hash_str,
        )
        response = agent_app(session)
        _persist_agent_response(session, response)
        session.emit_event(AGENT_COMPLETED_EVENT, {})
        sub_status = SubStatus.COMPLETED
        details = ""
        return True
    except Exception as err:  # pylint: disable=broad-exception-caught
        log(ERROR, "`flwr-agentapp` task failed", exc_info=err)
        details = _error_details(err)
        if session is not None:
            _try_emit_failed(session, err)
        return False
    finally:
        stub.PushTaskOutput(
            PushTaskOutputRequest(sub_status=sub_status, details=details)
        )


def _try_emit_failed(session: AgentSession, err: Exception) -> None:
    """Emit an agent.failed event without masking the original task failure."""
    try:
        session.emit_event(AGENT_FAILED_EVENT, {"error": _error_payload(err)})
    except Exception as event_err:  # pylint: disable=broad-exception-caught
        log(ERROR, "Failed to emit agent failure event", exc_info=event_err)


def _persist_agent_response(session: AgentSession, response: JsonObject) -> None:
    """Persist the returned Responses object as one assistant conversation item."""
    session.conversation.add_items([_assistant_item_from_response(session, response)])


def _assistant_item_from_response(
    session: AgentSession, response: JsonObject
) -> JsonObject:
    """Create one assistant conversation item from a Responses object."""
    output_text = _response_output_text(response)
    item: JsonObject = {
        "role": "assistant",
        "content": output_text,
        "response_id": _optional_str(response.get("id")),
        "model": _optional_str(response.get("model")) or session.model.default_model,
    }
    if output_text == "":
        item["response"] = response
    return item


def _response_output_text(response: JsonObject) -> str:
    """Extract output text from a Responses-compatible response object."""
    output_text = _optional_str(response.get("output_text"))
    if output_text is not None:
        return output_text
    output = response.get("output")
    if output is None:
        return ""
    return _extract_text(output)


def _extract_text(value: JsonValue) -> str:
    """Extract text from a Responses-compatible JSON value."""
    parts: list[str] = []
    _collect_text(value, parts)
    return "".join(parts)


def _collect_text(value: JsonValue, parts: list[str]) -> None:
    """Collect text leaves from Responses-compatible output structures."""
    if isinstance(value, str):
        parts.append(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_text(item, parts)
        return
    if isinstance(value, dict):
        for key in ("output_text", "text", "content"):
            child = value.get(key)
            if isinstance(child, (str, list, dict)):
                _collect_text(cast(JsonValue, child), parts)
                return
        output = value.get("output")
        if isinstance(output, (str, list, dict)):
            _collect_text(cast(JsonValue, output), parts)


def _optional_str(value: object) -> str | None:
    """Return value if it is a string."""
    return value if isinstance(value, str) else None


def _error_details(err: Exception) -> str:
    """Return task-output failure details."""
    return f"{type(err).__name__}: {str(err)}"


def _error_payload(err: Exception) -> JsonObject:
    """Return compact event error payload."""
    return {"type": type(err).__name__, "message": str(err)}
