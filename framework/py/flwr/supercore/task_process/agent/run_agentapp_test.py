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

# pylint: disable=protected-access

import importlib
import os
import signal
import threading
from queue import Queue
from unittest.mock import Mock

import pytest

from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common.constant import SubStatus
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
)
from flwr.supercore.exit import ExitCode
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType

from .run_agentapp import (
    _AgentAppTaskLifecycle,
    _ignore_graceful_signals,
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


def test_run_agentapp_task_constructs_lifecycle_with_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct and run the lifecycle after installing its signal handler."""
    lifecycle = Mock()
    lifecycle.run.return_value = ExitCode.SUCCESS
    lifecycle_cls = Mock(return_value=lifecycle)
    register_signal_handlers = Mock()
    monkeypatch.setattr(run_agentapp_module, "_AgentAppTaskLifecycle", lifecycle_cls)
    monkeypatch.setattr(
        run_agentapp_module, "register_signal_handlers", register_signal_handlers
    )
    log_queue: Queue[str | None] = Queue()

    assert _run_agentapp_task(
        "runtime.example:9092",
        log_queue,
        "task-token",
        False,
        b"root-certificates",
        "/runtime-ca.pem",
        True,
    ) == (lifecycle, ExitCode.SUCCESS)

    lifecycle_cls.assert_called_once_with(
        runtime_api_address="runtime.example:9092",
        log_queue=log_queue,
        token="task-token",
        insecure=False,
        certificates=b"root-certificates",
        certificates_path="/runtime-ca.pem",
        runtime_dependency_install=True,
    )
    register_signal_handlers.assert_called_once_with(
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        exit_message="Task stopped by user.",
        exit_handlers=[lifecycle.finalize],
    )


def test_agentapp_lifecycle_finalizes_once_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent cleanup should report and release task state only once."""
    client = Mock()
    client.PushTaskOutput.side_effect = RuntimeError("Runtime unavailable")
    retry_invoker = Mock(max_tries=10)
    grid = Mock(_runtime_client=client, _retry_invoker=retry_invoker)
    heartbeat = Mock(is_running=True)
    cleanup_runtime = Mock()
    monkeypatch.setattr(run_agentapp_module, "HttpGrid", Mock(return_value=grid))
    monkeypatch.setattr(
        run_agentapp_module, "cleanup_app_runtime_environment", cleanup_runtime
    )
    lifecycle = _AgentAppTaskLifecycle(
        runtime_api_address="runtime.example:9092",
        log_queue=Queue(),
        token="task-token",
        insecure=False,
        certificates=b"root-certificates",
        certificates_path="/runtime-ca.pem",
        runtime_dependency_install=False,
    )
    lifecycle._heartbeat_sender = heartbeat

    threads = [threading.Thread(target=lifecycle.finalize) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()

    client.PushTaskOutput.assert_called_once()
    output = client.PushTaskOutput.call_args.args[0]
    assert (output.sub_status, output.details) == (
        SubStatus.FAILED,
        "Task failed with unknown error.",
    )
    heartbeat.stop.assert_called_once_with()
    grid.close.assert_called_once_with()
    cleanup_runtime.assert_called_once_with(None)
    assert retry_invoker.max_tries == 1


def test_finalization_ignores_graceful_signals_process_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalization should restore the process-wide graceful signal handlers."""
    previous_handlers = {
        sig: object() for sig in run_agentapp_module.SIGNAL_TO_EXIT_CODE
    }
    handlers = previous_handlers.copy()

    def register(sig: int, handler: object) -> object:
        previous = handlers[sig]
        handlers[sig] = handler
        return previous

    monkeypatch.setattr(signal, "signal", register)

    with _ignore_graceful_signals():
        assert all(handler == signal.SIG_IGN for handler in handlers.values())

    assert handlers == previous_handlers


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
