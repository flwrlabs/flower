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
"""Tests for the AgentApp process environment."""

import importlib
import os
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flwr.agentapp import AgentApp
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common.constant import SubStatus
from flwr.proto.message_pb2 import Context as ProtoContext  # pylint: disable=E0611
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
)
from flwr.supercore.exit import ExitCode
from flwr.supercore.superexec.dependency_installer import (
    RuntimeDependencyInstallationError,
)
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType

from .run_agentapp import (
    _run_agentapp_task,
    _set_runtime_environment,
    message_to_prompt,
    pull_prompt,
    run_agentapp,
)

run_agentapp_module = importlib.import_module(
    "flwr.supercore.task_process.agent.run_agentapp"
)


@pytest.fixture(autouse=True)
def task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the task identity required to construct test messages."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 123)
    monkeypatch.setattr(TaskIdentity, "_run_id", 456)
    monkeypatch.setattr(TaskIdentity, "_node_id", 789)


def _payload_message(src_node_id: int, message_type: str = "query") -> Message:
    """Build a Grid message carrying a JSON payload record."""
    message = Message(
        RecordDict(
            {
                AGENT_MESSAGE_CONTENT_RECORD_KEY: ConfigRecord(
                    {AGENT_MESSAGE_TEXT_KEY: "hello world!"}
                )
            }
        ),
        dst_node_id=0,
        message_type=message_type,
    )
    message.metadata.__dict__["_message_id"] = "message-1"
    message.metadata.__dict__["_src_node_id"] = src_node_id
    return message


@pytest.mark.parametrize(("insecure", "scheme"), [(True, "http"), (False, "https")])
def test_set_runtime_environment(
    monkeypatch: pytest.MonkeyPatch, insecure: bool, scheme: str
) -> None:
    """Expose the Runtime Responses base URL and AgentApp task token."""
    monkeypatch.delenv("FLWR_RUNTIME_BASE_URL", raising=False)
    monkeypatch.delenv("FLWR_RUNTIME_API_KEY", raising=False)
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    _set_runtime_environment(
        "runtime.example:9092",
        "task-token",
        insecure=insecure,
        root_certificates_path="/path/to/runtime-ca.pem",
    )

    assert os.environ["FLWR_RUNTIME_BASE_URL"] == (
        f"{scheme}://runtime.example:9092/v1/runtime"
    )
    assert os.environ["FLWR_RUNTIME_API_KEY"] == "task-token"
    assert os.environ["SSL_CERT_FILE"] == "/path/to/runtime-ca.pem"


@pytest.mark.parametrize(
    ("message_type", "msg_src_node_id", "expected"),
    [
        (SYSTEM_MESSAGE_TYPE, 789, "hello world!"),
        (
            "query",
            789,
            '{"message_id":"message-1","src_node_id":"789","payload":"hello world!"}',
        ),
        (
            "query",
            99,
            '{"message_id":"message-1","src_node_id":"99","payload":"hello world!"}',
        ),
    ],
)
def test_message_to_prompt(
    message_type: str, msg_src_node_id: int, expected: str
) -> None:
    """Return system instructions as text and other messages as JSON."""
    assert (
        message_to_prompt(_payload_message(msg_src_node_id, message_type)) == expected
    )


def test_pull_prompt_requires_instruction() -> None:
    """Fail when the run has no initial instruction."""
    grid = Mock()
    grid.pull_messages.return_value = []

    with pytest.raises(RuntimeError, match="exactly one"):
        pull_prompt(grid)
    grid.pull_messages.assert_called_once_with([])


def test_pull_prompt_serializes_instruction() -> None:
    """Pull and serialize the run's initial instruction."""
    grid = Mock()
    grid.pull_messages.return_value = [_payload_message(789, SYSTEM_MESSAGE_TYPE)]

    assert pull_prompt(grid) == "hello world!"


def test_pull_prompt_rejects_multiple_instructions() -> None:
    """Reject ambiguous initial instructions for the singular prompt API."""
    grid = Mock()
    grid.pull_messages.return_value = [
        _payload_message(789),
        _payload_message(789),
    ]

    with pytest.raises(RuntimeError, match="exactly one"):
        pull_prompt(grid)


@pytest.mark.parametrize(
    ("failure", "exit_code", "sub_status", "details"),
    [
        (None, ExitCode.SUCCESS, SubStatus.COMPLETED, ""),
        (
            RuntimeError("app failed"),
            ExitCode.TASK_PROC_EXCEPTION,
            SubStatus.FAILED,
            "AgentApp failed with exception: app failed",
        ),
        (
            ImportError("app missing"),
            ExitCode.COMMON_APP_IMPORT_ERROR,
            SubStatus.FAILED,
            "AgentApp failed with exception: app missing",
        ),
        (
            RuntimeDependencyInstallationError("dependencies failed"),
            ExitCode.COMMON_RUNTIME_DEPENDENCY_INSTALLATION_ERROR,
            SubStatus.FAILED,
            "AgentApp failed with exception: dependencies failed",
        ),
    ],
)
def test_agentapp_lifecycle_runs_and_finalizes_once(  # pylint: disable=too-many-locals,too-many-statements
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    exit_code: int,
    sub_status: SubStatus,
    details: str,
) -> None:
    """The extracted lifecycle should preserve task execution and cleanup."""
    client = Mock()
    client.PullTaskInput.return_value = SimpleNamespace(
        context=object(),
        run=object(),
        fab=object(),
        task_id=17,
        federation_config=object(),
    )
    retry_invoker = Mock(max_tries=10)
    grid = Mock(_runtime_client=client, _retry_invoker=retry_invoker)
    heartbeat = Mock(is_running=True)
    log_uploader = Mock()
    agent_events = Mock()
    context = SimpleNamespace(node_id=99, run_id=42, run_config=None)
    run = SimpleNamespace(
        run_id=42,
        override_config={"setting": "value"},
        federation_id="federation",
        series_id=7,
        fab_id="publisher/app",
        fab_version="1.0.0",
    )
    fab = SimpleNamespace(content=b"fab", hash_str="fab-hash")
    runtime_env_dir = Path("/runtime-env")
    app_path = Path("/installed/app")
    app_main = Mock(side_effect=failure)
    app = AgentApp()
    app.main()(app_main)
    register_signal_handlers = Mock()

    monkeypatch.setattr(run_agentapp_module, "HttpGrid", Mock(return_value=grid))
    monkeypatch.setattr(
        run_agentapp_module, "HeartbeatSender", Mock(return_value=heartbeat)
    )
    monkeypatch.setattr(
        run_agentapp_module, "context_from_proto", Mock(return_value=context)
    )
    monkeypatch.setattr(run_agentapp_module, "run_from_proto", Mock(return_value=run))
    monkeypatch.setattr(run_agentapp_module, "fab_from_proto", Mock(return_value=fab))
    monkeypatch.setattr(
        run_agentapp_module, "get_sha256_hash", Mock(return_value="run-hash")
    )
    monkeypatch.setattr(
        run_agentapp_module, "start_log_uploader", Mock(return_value=log_uploader)
    )
    monkeypatch.setattr(run_agentapp_module, "pull_prompt", Mock(return_value="prompt"))
    monkeypatch.setattr(
        run_agentapp_module, "RuntimeAgentEvents", Mock(return_value=agent_events)
    )
    monkeypatch.setattr(
        run_agentapp_module, "StartRunRequest", Mock(return_value=Mock())
    )
    monkeypatch.setattr(run_agentapp_module, "fab_to_proto", Mock())
    monkeypatch.setattr(run_agentapp_module, "user_config_to_proto", Mock())
    monkeypatch.setattr(run_agentapp_module, "AgentRuntime", Mock())
    monkeypatch.setattr(run_agentapp_module, "RuntimeAgentConnectors", Mock())
    monkeypatch.setattr(run_agentapp_module, "RuntimeAgentGrid", Mock())
    monkeypatch.setattr(run_agentapp_module, "RuntimeAgentSession", Mock())
    monkeypatch.setattr(run_agentapp_module, "install_from_fab", Mock())
    monkeypatch.setattr(
        run_agentapp_module,
        "get_fab_metadata",
        Mock(return_value=("publisher/app", "1.0.0")),
    )
    monkeypatch.setattr(
        run_agentapp_module, "get_project_dir", Mock(return_value=app_path)
    )
    monkeypatch.setattr(
        run_agentapp_module,
        "install_app_dependencies",
        Mock(return_value=runtime_env_dir),
    )
    monkeypatch.setattr(
        run_agentapp_module,
        "get_project_config",
        Mock(
            return_value={
                "tool": {"flwr": {"app": {"components": {"agentapp": "pkg.app:app"}}}}
            }
        ),
    )
    monkeypatch.setattr(
        run_agentapp_module,
        "get_fused_config_from_dir",
        Mock(return_value={"setting": "value"}),
    )
    monkeypatch.setattr(run_agentapp_module, "event", Mock())
    monkeypatch.setattr(run_agentapp_module, "_set_runtime_environment", Mock())
    monkeypatch.setattr(run_agentapp_module, "load_app", Mock(return_value=app))
    monkeypatch.setattr(
        run_agentapp_module,
        "context_to_proto",
        Mock(return_value=ProtoContext(node_id=99)),
    )
    monkeypatch.setattr(run_agentapp_module, "flush_logs", Mock())
    monkeypatch.setattr(run_agentapp_module, "stop_log_uploader", Mock())
    monkeypatch.setattr(run_agentapp_module, "cleanup_app_runtime_environment", Mock())
    monkeypatch.setattr(
        run_agentapp_module, "register_signal_handlers", register_signal_handlers
    )

    log_queue: Queue[str | None] = Queue()
    lifecycle, actual_exit_code = _run_agentapp_task(
        "runtime.example:9092",
        log_queue,
        "task-token",
        False,
        b"root-certificates",
        "/runtime-ca.pem",
        True,
    )
    lifecycle.finalize()
    lifecycle.finalize()

    assert actual_exit_code == exit_code
    assert (TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id) == (
        17,
        42,
        99,
    )
    assert context.run_config == {"setting": "value"}
    output = client.PushTaskOutput.call_args.args[0]
    assert (output.sub_status, output.details) == (sub_status, details)
    assert lifecycle.event_details(exit_code) == {
        "run-id-hash": "run-hash",
        "success": failure is None,
    }
    assert register_signal_handlers.call_args.kwargs == {
        "event_type": EventType.FLWR_AGENTAPP_RUN_LEAVE,
        "exit_message": "Task stopped by user.",
        "exit_handlers": [lifecycle.finalize],
    }
    app_main.assert_called_once()
    client.PushTaskOutput.assert_called_once()
    assert retry_invoker.max_tries == 1
    run_agentapp_module.flush_logs.assert_called_once_with(log_queue)
    run_agentapp_module.stop_log_uploader.assert_called_once_with(
        log_queue, log_uploader
    )
    heartbeat.stop.assert_called_once_with()
    grid.close.assert_called_once_with()
    run_agentapp_module.cleanup_app_runtime_environment.assert_called_once_with(
        runtime_env_dir
    )


def test_run_agentapp_keeps_cold_process_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold entry point should keep validation, monitoring, and Flower exit."""
    lifecycle = Mock()
    lifecycle.event_details.return_value = {
        "run-id-hash": "run-hash",
        "success": False,
    }
    run_task = Mock(return_value=(lifecycle, ExitCode.TASK_PROC_EXCEPTION))
    parent_monitor = Mock()
    validate_certificates = Mock(return_value=b"root-certificates")
    flwr_exit = Mock()
    monkeypatch.setattr(run_agentapp_module, "_run_agentapp_task", run_task)
    monkeypatch.setattr(
        run_agentapp_module, "start_parent_process_monitor", parent_monitor
    )
    monkeypatch.setattr(
        run_agentapp_module,
        "validate_and_resolve_root_certificates",
        validate_certificates,
    )
    monkeypatch.setattr(run_agentapp_module, "flwr_exit", flwr_exit)
    log_queue: Queue[str | None] = Queue()

    run_agentapp(
        "runtime.example:9092",
        log_queue,
        "task-token",
        False,
        certificates_path="/runtime-ca.pem",
        parent_pid=123,
        runtime_dependency_install=False,
    )

    parent_monitor.assert_called_once_with(123)
    validate_certificates.assert_called_once_with("/runtime-ca.pem", False)
    run_task.assert_called_once_with(
        "runtime.example:9092",
        log_queue,
        "task-token",
        False,
        b"root-certificates",
        "/runtime-ca.pem",
        False,
    )
    flwr_exit.assert_called_once_with(
        code=ExitCode.TASK_PROC_EXCEPTION,
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        event_details={"run-id-hash": "run-hash", "success": False},
    )
