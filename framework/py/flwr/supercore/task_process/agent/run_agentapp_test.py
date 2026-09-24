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

import hashlib
import importlib
import os
import signal
import threading
from pathlib import Path
from queue import Queue
from unittest.mock import Mock

import pytest

from flwr.agentapp import AgentApp, LoadAgentAppError
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common.constant import SubStatus
from flwr.supercore.constant import (
    AGENT_MESSAGE_CONTENT_RECORD_KEY,
    AGENT_MESSAGE_TEXT_KEY,
    SYSTEM_MESSAGE_TYPE,
)
from flwr.supercore.exit import ExitCode
from flwr.supercore.fab import Fab
from flwr.supercore.run import Run
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.telemetry import EventType

from .run_agentapp import (
    PreloadedAgentApp,
    _AgentAppTaskLifecycle,
    _ignore_graceful_signals,
    _run_agentapp_task,
    _set_runtime_environment,
    message_to_prompt,
    preload_agentapp,
    pull_prompt,
    run_agentapp,
    run_agentapp_once,
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


def test_preload_agentapp_installs_the_verified_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Preload an arbitrary absolute filename from the bytes that were hashed."""
    fab_content = b"trusted-fab-content"
    fab_path = tmp_path / "operator-selected-name"
    fab_path.write_bytes(fab_content)
    fab_hash = hashlib.sha256(fab_content).hexdigest()
    app_path = tmp_path / "installed"
    app = AgentApp()
    install = Mock(return_value=app_path)
    config = {
        "project": {"name": "app", "version": "1.0.0"},
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "publisher",
                    "components": {"agentapp": "pkg.app:app"},
                }
            }
        },
    }
    monkeypatch.setattr(run_agentapp_module, "install_from_fab", install)
    monkeypatch.setattr(
        run_agentapp_module, "get_project_config", Mock(return_value=config)
    )
    monkeypatch.setattr(run_agentapp_module, "load_app", Mock(return_value=app))

    assert preload_agentapp(fab_path, fab_hash) == PreloadedAgentApp(
        app=app,
        app_path=app_path,
        fab_id="publisher/app",
        fab_version="1.0.0",
        fab_hash=fab_hash,
    )
    install.assert_called_once_with(fab_content, skip_prompt=True)
    run_agentapp_module.load_app.assert_called_once_with(
        "pkg.app:app", LoadAgentAppError, str(app_path)
    )


def test_preload_agentapp_rejects_wrong_component_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail startup when the configured component is not an AgentApp."""
    fab_content = b"trusted-fab-content"
    fab_path = tmp_path / "agent.fab"
    fab_path.write_bytes(fab_content)
    app_path = tmp_path / "installed"
    monkeypatch.setattr(
        run_agentapp_module, "install_from_fab", Mock(return_value=app_path)
    )
    monkeypatch.setattr(
        run_agentapp_module,
        "get_project_config",
        Mock(
            return_value={
                "project": {"name": "app", "version": "1.0.0"},
                "tool": {
                    "flwr": {
                        "app": {
                            "publisher": "publisher",
                            "components": {"agentapp": "pkg.app:not_an_app"},
                        }
                    }
                },
            }
        ),
    )
    monkeypatch.setattr(run_agentapp_module, "load_app", Mock(return_value=object()))

    with pytest.raises(LoadAgentAppError, match="is not of type"):
        preload_agentapp(fab_path, hashlib.sha256(fab_content).hexdigest())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fab_hash", "other-hash"),
        ("fab_id", "other/app"),
        ("fab_version", "2.0.0"),
    ],
)
def test_preloaded_agentapp_requires_exact_task_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    """Reject a pulled task unless its hash, id, and version all match."""
    app = AgentApp()
    preloaded = PreloadedAgentApp(
        app=app,
        app_path=tmp_path / "installed",
        fab_id="publisher/app",
        fab_version="1.0.0",
        fab_hash="a" * 64,
    )
    monkeypatch.setattr(run_agentapp_module, "HttpGrid", Mock())
    lifecycle = _AgentAppTaskLifecycle(
        runtime_api_address="runtime.example:9092",
        log_queue=Queue(),
        token="task-token",
        insecure=True,
        certificates=None,
        certificates_path=None,
        runtime_dependency_install=False,
        preloaded=preloaded,
    )
    fab = Fab(
        content=b"task-fab",
        hash_str=preloaded.fab_hash,
        verifications={},
    )
    run = Run.create_empty(0)
    run.fab_id = preloaded.fab_id
    run.fab_version = preloaded.fab_version
    install = Mock()
    load = Mock()
    install_dependencies = Mock()
    monkeypatch.setattr(run_agentapp_module, "install_from_fab", install)
    monkeypatch.setattr(run_agentapp_module, "load_app", load)
    monkeypatch.setattr(
        run_agentapp_module, "install_app_dependencies", install_dependencies
    )

    assert lifecycle._prepare_task_app(fab, run) == (
        preloaded.app_path,
        app,
        None,
    )
    if field == "fab_hash":
        fab.hash_str = value
    else:
        setattr(run, field, value)
    with pytest.raises(RuntimeError, match="does not match"):
        lifecycle._prepare_task_app(fab, run)
    install.assert_not_called()
    load.assert_not_called()
    install_dependencies.assert_not_called()


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
        resident=False,
    ) == (lifecycle, ExitCode.SUCCESS)

    lifecycle_cls.assert_called_once_with(
        runtime_api_address="runtime.example:9092",
        log_queue=log_queue,
        token="task-token",
        insecure=False,
        certificates=b"root-certificates",
        certificates_path="/runtime-ca.pem",
        runtime_dependency_install=True,
        preloaded=None,
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
        preloaded=None,
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


def test_resident_accepts_before_task_input_and_output_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept authority after Runtime setup and signals, before task work."""
    calls: list[str] = []
    lifecycle = Mock()

    def run() -> int:
        calls.append("run")
        return ExitCode.SUCCESS

    def construct(**_: object) -> Mock:
        calls.append("runtime")
        return lifecycle

    lifecycle.run.side_effect = run
    lifecycle_cls = Mock(side_effect=construct)
    monkeypatch.setattr(run_agentapp_module, "_AgentAppTaskLifecycle", lifecycle_cls)
    monkeypatch.setattr(
        run_agentapp_module,
        "_register_resident_signal_handlers",
        lambda _: calls.append("signals"),
    )
    monkeypatch.setattr(
        run_agentapp_module,
        "mirror_output_to_queue",
        lambda _: calls.append("output"),
    )

    returned_lifecycle, exit_code = _run_agentapp_task(
        "runtime.example:9092",
        Queue(),
        "task-token",
        True,
        None,
        None,
        False,
        resident=True,
        preloaded=Mock(),
        on_started=lambda: calls.append("accepted"),
    )

    assert calls == ["runtime", "signals", "accepted", "output", "run"]
    assert (returned_lifecycle, exit_code) == (lifecycle, ExitCode.SUCCESS)


def test_run_agentapp_once_bounds_resident_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resident task should finalize under the worker force-exit timeout."""
    lifecycle = Mock()
    run_task = Mock(return_value=(lifecycle, ExitCode.TASK_PROC_EXCEPTION))
    force_exit_timer = Mock()
    timer_cls = Mock(return_value=force_exit_timer)
    monkeypatch.setattr(run_agentapp_module, "_run_agentapp_task", run_task)
    monkeypatch.setattr(run_agentapp_module.threading, "Timer", timer_cls)
    preloaded = Mock()
    on_started = Mock()

    assert (
        run_agentapp_once(
            "runtime.example:9092",
            "task-token",
            False,
            b"root-certificates",
            on_started,
            preloaded=preloaded,
            certificates_path="/runtime-ca.pem",
        )
        == 1
    )

    args = run_task.call_args.args
    assert args[0] == "runtime.example:9092"
    assert isinstance(args[1], Queue)
    assert args[2:] == (
        "task-token",
        False,
        b"root-certificates",
        "/runtime-ca.pem",
        False,
    )
    assert run_task.call_args.kwargs == {
        "resident": True,
        "preloaded": preloaded,
        "on_started": on_started,
    }
    timer_cls.assert_called_once_with(
        run_agentapp_module.FORCE_EXIT_TIMEOUT_SECONDS,
        os._exit,
        args=(1,),
    )
    assert force_exit_timer.daemon
    force_exit_timer.start.assert_called_once_with()
    lifecycle.complete.assert_called_once_with(ExitCode.TASK_PROC_EXCEPTION)
    force_exit_timer.cancel.assert_called_once_with()


def test_run_agentapp_keeps_cold_process_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold entry point should keep validation, monitoring, and Flower exit."""
    lifecycle = Mock()
    lifecycle.event_details.return_value = {
        "exit_code": ExitCode.TASK_PROC_EXCEPTION,
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
        resident=False,
    )
    flwr_exit.assert_called_once_with(
        code=ExitCode.TASK_PROC_EXCEPTION,
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        event_details={
            "exit_code": ExitCode.TASK_PROC_EXCEPTION,
            "run-id-hash": "run-hash",
            "success": False,
        },
    )
