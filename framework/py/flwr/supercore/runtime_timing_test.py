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
"""Privacy, filtering and forwarding contracts for runtime timing probes."""

import json

# pylint: disable=protected-access
import sys
import threading
import time
from io import StringIO
from logging import DEBUG, INFO
from queue import Queue
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flwr.cli.chat.chat_app import ChatApplication
from flwr.common.constant import SubStatus
from flwr.proto.control_pb2 import StartRunResponse  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import AcquireTaskRequest  # pylint: disable=E0611
from flwr.server.superlink.linkstate import InMemoryLinkState
from flwr.supercore import runtime_timing
from flwr.supercore.constant import NOOP_FEDERATION_ID, TaskType
from flwr.supercore.corestate.corestate import CoreState
from flwr.supercore.dependencies.runtime import get_runtime_state
from flwr.supercore.logger import console_handler, mirror_output_to_queue
from flwr.supercore.object_store.in_memory_object_store import InMemoryObjectStore
from flwr.supercore.routers.runtime.responses import router
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.runtime_timing import RuntimeTiming
from flwr.supercore.servicer.runtime import runtime_handlers
from flwr.supercore.superexec.executor.warm_executor_dispatch import (
    KubernetesWarmExecutorDispatch,
)
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.task_process.agent.session import RuntimeAgentEvents
from flwr.supercore.task_process.model.task import handle_task
from flwr.supercore.task_worker_protocol import relay_task_output
from flwr.supercore.typing import JSONObject
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.control.control_handlers import _stream_run_events


@pytest.fixture(name="records")
def captured_records(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture only timing JSON while enabling the profiling flag and DEBUG."""
    captured: list[dict[str, Any]] = []
    monkeypatch.setenv("FLWR_RUNTIME_TIMING_LOGGING", "1")
    monkeypatch.setattr(console_handler, "level", DEBUG)
    monkeypatch.setattr(
        runtime_timing,
        "log_runtime_timing",
        lambda value: captured.append(
            json.loads(value.removeprefix("runtime_timing "))
        ),
    )
    return captured


@pytest.mark.parametrize(("flag", "level"), [(None, DEBUG), ("0", DEBUG), ("1", INFO)])
def test_disabled_has_no_output_or_clocks(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, level: int
) -> None:
    """Disabled probes do not read clocks or serialize metadata."""
    if flag is None:
        monkeypatch.delenv("FLWR_RUNTIME_TIMING_LOGGING", raising=False)
    else:
        monkeypatch.setenv("FLWR_RUNTIME_TIMING_LOGGING", flag)
    monkeypatch.setattr(console_handler, "level", level)
    clock = Mock(side_effect=AssertionError("disabled probe read a clock"))
    output = Mock()
    monkeypatch.setattr("flwr.supercore.runtime_timing.time.monotonic_ns", clock)
    monkeypatch.setattr(runtime_timing, "log_runtime_timing", output)
    timing = RuntimeTiming(run_id=7, task_id=11)
    with timing.span("agent.user_code"):
        timing.first_event("agent.events", "response.output_text.delta")
    output.assert_not_called()
    clock.assert_not_called()


def test_span_order_correlation_and_failure_redaction(
    records: list[dict[str, Any]],
) -> None:
    """Failed intervals retain correlation without exception text or authority."""
    timing = RuntimeTiming(run_id=7, task_id=22, parent_task_id=11)
    with pytest.raises(ValueError):
        with timing.span("model.provider_post"):
            raise ValueError("secret prompt token response-body")
    assert [record["marker"] for record in records] == [
        "model.provider_post.started",
        "model.provider_post.failed",
    ]
    start, end = records
    assert start["span_id"] == end["span_id"]
    assert start["clock_domain"] == end["clock_domain"]
    assert start["scope_id"] == end["scope_id"]
    assert end["duration_ns"] >= 0
    assert end["monotonic_ns"] >= start["monotonic_ns"]
    assert end["unix_time_ns"] > 0
    assert end["success"] is False
    assert (end["run_id"], end["task_id"], end["parent_task_id"]) == (7, 22, 11)
    assert "secret" not in json.dumps(records)


def test_first_events_are_bounded_and_separate_reasoning(
    records: list[dict[str, Any]],
) -> None:
    """A stream reports first output text independently of earlier reasoning."""
    timing = RuntimeTiming(run_id=7, task_id=11)
    for _ in range(100):
        timing.first_event(
            "client.received", "response.reasoning_summary_text.delta", 1
        )
        timing.first_event("client.received", "response.output_text.delta", 2)
    assert [record["marker"] for record in records] == [
        "client.received.first_event",
        "client.received.first_reasoning",
        "client.received.first_text",
    ]
    assert [record["event_id"] for record in records] == [1, 1, 2]


def test_probe_output_failure_does_not_fail_task(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]
) -> None:
    """Closed stdout or a failing handler must not change task execution."""
    del records
    monkeypatch.setattr(
        runtime_timing, "log_runtime_timing", Mock(side_effect=OSError("closed"))
    )
    with RuntimeTiming(run_id=7).span("agent.user_code"):
        pass


def test_markers_bypass_task_upload_and_survive_worker_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use standard DEBUG logging with the current resident worker output path."""
    monkeypatch.setenv("FLWR_RUNTIME_TIMING_LOGGING", "1")
    monkeypatch.setattr(console_handler, "level", DEBUG)
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", StringIO())
    monkeypatch.setattr(console_handler, "stream", stream)
    queue: Queue[str | None] = Queue()
    mirror_output_to_queue(queue)
    RuntimeTiming(run_id=7, task_id=11).mark("agent.input_ready")
    assert "runtime_timing {" in stream.getvalue()
    assert queue.empty()
    print("normal task output")
    assert queue.get() == "normal task output"
    assert queue.get() == "\n"
    sender = Mock()
    with relay_task_output(sender, "secret-token"):
        RuntimeTiming(run_id=7, task_id=11).mark("agent.preloaded")
    relayed = "".join(call.args[1] for call in sender.write.call_args_list)
    assert "runtime_timing {" in relayed
    assert "secret-token" not in relayed
    assert queue.empty()


def test_forwarded_markers_stay_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Warm output forwarding must not promote DEBUG timing markers to INFO."""
    dispatch = KubernetesWarmExecutorDispatch(
        Mock(), pod_name="pod-1", task_id=11, log_output=True
    )
    with caplog.at_level(DEBUG, logger="flwr"):
        dispatch._log_output_line(
            "stdout", 'DEBUG: runtime_timing {"marker":"agent.preloaded"}'
        )  # pylint: disable=protected-access
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == DEBUG


# pylint: disable-next=too-many-locals,too-many-statements
def test_in_process_model_to_chat_smoke(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]
) -> None:
    """Relay a deterministic provider stream through real in-memory state to chat.

    HTTP Responses routing, acquisition, Model handling, event persistence, Agent
    republishing, Control streaming and client rendering run locally. External
    provider I/O and the worker/client transport adapters are deterministic fakes.
    """
    state = InMemoryLinkState(NoOpFederationManager(), InMemoryObjectStore())
    run_id = state.create_run(
        "smoke", "1", "a" * 64, {}, NOOP_FEDERATION_ID, None, None, TaskType.AGENT_APP
    )
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.primary_task_id is not None
    agent_id = run.primary_task_id
    token = state.claim_task(agent_id)
    assert token and state.activate_task(agent_id)
    monkeypatch.setenv("FLWR_MODEL_API_ENDPOINT", "http://local-provider/v1/responses")
    monkeypatch.delenv("FLWR_MODEL_API_KEY", raising=False)
    provider = Mock(status_code=200, headers={"Content-Type": "text/event-stream"})
    provider.iter_lines.return_value = iter(
        [
            b'data: {"type":"response.created"}',
            b"",
            b'data: {"type":"response.reasoning_summary_text.delta",'
            b'"delta":"private reasoning"}',
            b"",
            b'data: {"type":"response.output_text.delta","delta":"smoke-ready"}',
            b"",
            b'data: {"type":"response.completed",'
            b'"response":{"object":"response","status":"completed",'
            b'"output":[]}}',
            b"",
        ]
    )
    monkeypatch.setattr(
        "flwr.supercore.task_process.model.provider.requests.post",
        Mock(return_value=provider),
    )
    for field, value in (
        ("_run_id", run_id),
        ("_task_id", agent_id),
        ("_node_id", state.get_node_id()),
    ):
        monkeypatch.setattr(TaskIdentity, field, value)
    errors = []

    def execute_model() -> None:
        try:
            deadline = time.monotonic() + 5
            acquired = runtime_handlers.acquire_task(
                AcquireTaskRequest(supported_task_types=[TaskType.MODEL]), state
            )
            while not acquired.HasField("task") and time.monotonic() < deadline:
                time.sleep(0.005)
                acquired = runtime_handlers.acquire_task(
                    AcquireTaskRequest(supported_task_types=[TaskType.MODEL]), state
                )
            assert acquired.HasField("task")
            model = acquired.task
            assert state.activate_task(model.task_id)
            TaskIdentity.task_id = model.task_id
            stub = Mock(spec=RuntimeHttpClient)
            stub.PullTaskMessage.side_effect = (
                lambda request: runtime_handlers.pull_task_message(
                    request, state, model
                )
            )
            stub.PushTaskEvents.side_effect = (
                lambda request: runtime_handlers.push_task_events(request, state, model)
            )
            stub.PushTaskMessage.side_effect = (
                lambda request: runtime_handlers.push_task_message(
                    request, state, model
                )
            )
            stub.RecordTaskUsage.side_effect = (
                lambda request: runtime_handlers.record_task_usage(
                    request, state, model
                )
            )
            handle_task(stub)
            assert state.finish_task(model.task_id, SubStatus.COMPLETED, "")
        except Exception as error:  # pylint: disable=broad-exception-caught
            errors.append(error)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_runtime_state] = lambda: cast(CoreState, state)
    worker = threading.Thread(target=execute_model)
    worker.start()
    with TestClient(app) as client:
        response = client.post(
            "/v1/runtime/responses",
            headers={"Authorization": f"Bearer {token}"},
            json={"model": "local", "input": "private prompt", "stream": True},
        )
    worker.join(timeout=6)
    assert not worker.is_alive()
    assert not errors
    assert response.status_code == 200, response.text
    TaskIdentity.task_id = agent_id
    agent_task = state.get_tasks(task_ids=[agent_id])[0]
    agent_stub = Mock(spec=RuntimeHttpClient)
    agent_stub.PushTaskEvents.side_effect = (
        lambda request: runtime_handlers.push_task_events(request, state, agent_task)
    )
    events = RuntimeAgentEvents(agent_stub)
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.emit(cast(JSONObject, json.loads(line.removeprefix("data: "))))
    events.close()
    assert state.finish_task(agent_id, SubStatus.COMPLETED, "")
    run = state.get_run_info(run_ids=[run_id])[0]
    chat_stub = Mock()
    chat_stub.StartRun.return_value = StartRunResponse(run_id=run_id)
    chat_stub.StreamRunEvents.side_effect = lambda request: _stream_run_events(
        request.run_id, run, None, state, None
    )
    with patch.object(ChatApplication, "_create_application", return_value=Mock()):
        chat = ChatApplication(chat_stub, [Federation(name=NOOP_FEDERATION_ID)])
    chat._run_prompt_sync(
        "private prompt", "@local/smoke", "a" * 64
    )  # pylint: disable=protected-access
    assert any(getattr(block, "body", "") == "smoke-ready" for block in chat.transcript)
    by_marker = {record["marker"]: record for record in records}
    for marker in (
        "runtime.child_created",
        "runtime.task_claimed",
        "model.provider.first_reasoning",
        "model.provider.first_text",
        "responses.yield.first_text",
        "responses.model_reply_received",
        "agent.events_enqueue.first_text",
        "control.yield.first_text",
        "client.received.first_text",
        "client.stream_end",
    ):
        assert marker in by_marker
        assert by_marker[marker]["run_id"] == run_id
    assert by_marker["runtime.child_created"]["parent_task_id"] == agent_id
    assert by_marker["runtime.child_created"]["task_id"] != agent_id
    assert "private prompt" not in json.dumps(records)
    assert "private reasoning" not in json.dumps(records)
    assert "smoke-ready" not in json.dumps(records)
    assert token not in json.dumps(records)


def test_invalid_fab_metadata_is_not_serialized(records: list[dict[str, Any]]) -> None:
    """Do not treat arbitrary task-supplied text as a FAB hash."""
    RuntimeTiming(run_id=7, fab_hash="private credential disguised as hash").mark(
        "runtime.child_created"
    )
    assert records[0]["fab_hash"] is None
    assert "private" not in json.dumps(records)
