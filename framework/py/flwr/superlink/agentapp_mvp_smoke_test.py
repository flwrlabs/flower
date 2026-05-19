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
"""Smoke test for the AgentApp Task-system MVP path."""

# pylint: disable=E0611,too-many-instance-attributes,too-many-locals

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from contextvars import Token
from dataclasses import dataclass, field
from typing import TypeVar, cast
from unittest.mock import Mock

import grpc
import pytest

from flwr.agentapp.builtin import gpt_chat
from flwr.common import Message
from flwr.common.constant import Status, SubStatus
from flwr.common.serde import (
    message_from_proto,
    message_to_proto,
    user_config_to_proto,
)
from flwr.common.typing import AccountInfo
from flwr.proto.appio_pb2 import (
    ClaimTaskRequest,
    PullPendingTasksRequest,
)
from flwr.proto.control_pb2 import (
    DeleteConversationRequest,
    GetConversationRequest,
    ListConversationsRequest,
    StartRunRequest,
    StreamRunEventsResponse,
    StreamRunEventsRequest,
)
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.proto.task_pb2 import Task
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.server.superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION, TaskType
from flwr.supercore.executors.model_provider import ModelProviderResult
from flwr.supercore.executors.run_agentapp import (
    AgentAppTaskInput,
    _pull_task_input,
    _run_agentapp_task,
)
from flwr.supercore.executors.run_model import _run_model_task
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    SuperExecAuthClientInterceptor,
)
from flwr.supercore.interceptors.superexec_auth_interceptor import (
    SERVERAPPIO_SUPEREXEC_METHODS,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.task_message import (
    JsonObject,
    ModelTaskMessage,
    ModelTaskResultMessage,
)
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.control.control_account_auth_interceptor import (
    shared_account_info,
)
from flwr.superlink.servicer.control.control_servicer import ControlServicer

_SUPEREXEC_SECRET = b"agentapp-mvp-smoke-superexec-secret"
_SMOKE_TIMEOUT = 5.0
_POLL_INTERVAL = 0.02

WaitResultT = TypeVar("WaitResultT")


class RpcAbort(Exception):
    """Small exception carrying the code/details passed to ServicerContext.abort."""

    def __init__(self, code: grpc.StatusCode, details: str) -> None:
        super().__init__()
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        """Return the aborted status code."""
        return self._code

    def details(self) -> str:
        """Return the aborted details."""
        return self._details


class FakeControlContext:
    """Minimal Control servicer context used by direct Control RPC calls."""

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        """Abort the direct RPC call."""
        raise RpcAbort(code, details)

    def is_active(self) -> bool:
        """Return whether streaming RPCs should remain active."""
        return True


@dataclass
class AgentAppMvpSmokeHarness:
    """In-process harness for the AgentApp MVP smoke test."""

    monkeypatch: pytest.MonkeyPatch
    objectstore_factory: ObjectStoreFactory = field(default_factory=ObjectStoreFactory)
    state_factory: LinkStateFactory = field(init=False)
    state: LinkState = field(init=False)
    control: ControlServicer = field(init=False)
    server: grpc.Server = field(init=False)
    channel: grpc.Channel = field(init=False)
    superexec_stub: ServerAppIoStub = field(init=False)
    stored_messages: list[Message] = field(default_factory=list)
    _stored_messages_lock: threading.Lock = field(default_factory=threading.Lock)
    _account_info_token: Token[AccountInfo | None] | None = None

    def __post_init__(self) -> None:
        """Create state, servicers, and gRPC channels."""
        self.state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME,
            NoOpFederationManager(),
            self.objectstore_factory,
        )
        self.state = self.state_factory.state()
        authn_plugin = NoOpControlAuthnPlugin(Mock(), False)
        self.control = ControlServicer(
            linkstate_factory=self.state_factory,
            objectstore_factory=self.objectstore_factory,
            authn_plugin=authn_plugin,
        )
        account_info = authn_plugin.validate_tokens_in_metadata([])[1]
        assert account_info is not None
        assert account_info.flwr_aid is not None
        self._account_info_token = shared_account_info.set(account_info)
        self._record_stored_task_messages()

        self.server = run_serverappio_api_grpc(
            "127.0.0.1:0",
            self.state_factory,
            self.objectstore_factory,
            None,
            superexec_auth_secret=_SUPEREXEC_SECRET,
        )
        self.channel = grpc.insecure_channel(self.server.bound_address)
        grpc.channel_ready_future(self.channel).result(timeout=_SMOKE_TIMEOUT)
        superexec_channel = grpc.intercept_channel(
            self.channel,
            SuperExecAuthClientInterceptor(
                master_secret=_SUPEREXEC_SECRET,
                protected_methods=SERVERAPPIO_SUPEREXEC_METHODS,
            ),
        )
        self.superexec_stub = ServerAppIoStub(superexec_channel)

    def close(self) -> None:
        """Close gRPC resources."""
        self.channel.close()
        self.server.stop(None)
        if self._account_info_token is not None:
            shared_account_info.reset(self._account_info_token)

    def start_gpt_chat_run(self) -> tuple[int, str]:
        """Create one gpt-chat agent run through Control StartRun."""
        request = StartRunRequest(federation=NOOP_FEDERATION)
        for key, value in user_config_to_proto(
            {
                "run_type": "agent",
                "agent_ref": "gpt-chat",
                "input_json": json.dumps(
                    [{"role": "user", "content": "hello"}],
                    separators=(",", ":"),
                ),
                "model": "test-model",
            }
        ).items():
            request.override_config[key].CopyFrom(value)

        response = self.control.StartRun(
            request, cast(grpc.ServicerContext, FakeControlContext())
        )
        assert response.run_id > 0
        assert response.conversation_id
        return response.run_id, response.conversation_id

    def task_stub(self, token: str) -> ServerAppIoStub:
        """Create a ServerAppIo stub authenticated with a task token."""
        channel = grpc.intercept_channel(
            self.channel,
            AppIoTokenClientInterceptor(token),
        )
        return ServerAppIoStub(channel)

    def claim_task(self, task_id: int) -> str:
        """Claim a pending task through the SuperExec-authenticated RPC."""
        response = self.superexec_stub.ClaimTask(
            ClaimTaskRequest(task_id=task_id), timeout=_SMOKE_TIMEOUT
        )
        assert response.token
        return cast(str, response.token)

    def pending_tasks(self) -> list[Task]:
        """Return pending tasks through the SuperExec-authenticated RPC."""
        response = self.superexec_stub.PullPendingTasks(
            PullPendingTasksRequest(), timeout=_SMOKE_TIMEOUT
        )
        return list(response.tasks)

    def find_stored_message(
        self,
        *,
        message_type: str,
        src_task_id: int | None = None,
        dst_task_id: int | None = None,
    ) -> Message | None:
        """Return a copy of the first recorded task message matching filters."""
        with self._stored_messages_lock:
            messages = list(self.stored_messages)
        for message in messages:
            metadata = message.metadata
            if metadata.message_type != message_type:
                continue
            if src_task_id is not None and metadata.src_task_id != src_task_id:
                continue
            if dst_task_id is not None and metadata.dst_task_id != dst_task_id:
                continue
            return message_from_proto(message_to_proto(message))
        return None

    def stream_run_events(self, run_id: int) -> list[StreamRunEventsResponse]:
        """Return all persisted run events through Control StreamRunEvents."""
        return list(
            self.control.StreamRunEvents(
                StreamRunEventsRequest(run_id=run_id),
                cast(grpc.ServicerContext, FakeControlContext()),
            )
        )

    def _record_stored_task_messages(self) -> None:
        """Record task messages before state delivery removes them."""
        original_store_task_message = self.state.store_task_message

        def store_task_message(message: Message) -> str | None:
            copied = message_from_proto(message_to_proto(message))
            with self._stored_messages_lock:
                self.stored_messages.append(copied)
            return original_store_task_message(message)

        self.monkeypatch.setattr(self.state, "store_task_message", store_task_message)


def test_gpt_chat_mvp_runs_through_task_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the gpt-chat MVP path from StartRun through conversations."""
    harness = AgentAppMvpSmokeHarness(monkeypatch)
    try:
        run_id, conversation_id = harness.start_gpt_chat_run()
        agent_task_id = _primary_task_id(harness.state, run_id)
        assert _task_by_id(harness.state, agent_task_id).type == TaskType.AGENT_APP
        assert any(task.task_id == agent_task_id for task in harness.pending_tasks())

        agent_token = harness.claim_task(agent_task_id)
        agent_stub = harness.task_stub(agent_token)
        agent_task_input = _pull_task_input(agent_stub)

        agent_result: dict[str, object] = {}
        agent_thread = threading.Thread(
            target=_run_agentapp_thread,
            args=(agent_result, agent_stub, agent_task_input),
            name="agentapp-mvp-smoke",
        )
        agent_thread.start()

        model_task = _wait_for(
            "AgentApp to create a model task",
            lambda: _task_by_type(harness.state, run_id, TaskType.MODEL),
        )
        model_request = _wait_for(
            "AgentApp to push a model request",
            lambda: harness.find_stored_message(
                message_type=ModelTaskMessage.MESSAGE_TYPE,
                src_task_id=agent_task_id,
                dst_task_id=model_task.task_id,
            ),
        )
        model_request_spec = ModelTaskMessage.from_message(model_request)
        assert model_request_spec.payload["model"] == "test-model"
        assert model_request_spec.payload["stream"] is True

        model_token = harness.claim_task(model_task.task_id)
        model_stub = harness.task_stub(model_token)
        _run_model_task(model_stub, _fake_model_provider)

        agent_thread.join(timeout=_SMOKE_TIMEOUT)
        if agent_thread.is_alive():
            raise AssertionError("AgentApp task did not finish within timeout.")
        if err := agent_result.get("exception"):
            raise AssertionError("AgentApp task raised an exception.") from cast(
                BaseException, err
            )
        assert agent_result == {"completed": True}

        model_result = _wait_for(
            "Model task to push a result",
            lambda: harness.find_stored_message(
                message_type=ModelTaskResultMessage.MESSAGE_TYPE,
                src_task_id=model_task.task_id,
                dst_task_id=agent_task_id,
            ),
        )
        model_result_spec = ModelTaskResultMessage.from_message(model_result)
        assert model_result.metadata.reply_to_message_id == (
            model_request.metadata.message_id
        )
        assert model_result_spec.payload["response_id"] == "resp-1"

        agent_task = _task_by_id(harness.state, agent_task_id)
        model_task = _task_by_id(harness.state, model_task.task_id)
        assert agent_task.status.status == Status.FINISHED
        assert agent_task.status.sub_status == SubStatus.COMPLETED
        assert model_task.status.status == Status.FINISHED
        assert model_task.status.sub_status == SubStatus.COMPLETED

        events = harness.stream_run_events(run_id)
        assert [event.sequence_number for event in events] == sorted(
            event.sequence_number for event in events
        )
        event_names = [event.event for event in events]
        for expected_event in [
            "agent.started",
            "model.started",
            "response.output_text.delta",
            "response.completed",
            "model.completed",
            "agent.completed",
        ]:
            assert expected_event in event_names
        delta_events = [
            json.loads(event.data)
            for event in events
            if event.event == "response.output_text.delta"
        ]
        assert delta_events == [
            {"type": "response.output_text.delta", "delta": "Hello from smoke"}
        ]

        _assert_conversation_outputs(harness, conversation_id)
        delete_response = harness.control.DeleteConversation(
            DeleteConversationRequest(conversation_id=conversation_id),
            cast(grpc.ServicerContext, FakeControlContext()),
        )
        assert delete_response.deleted
        with pytest.raises(RpcAbort) as err:
            harness.control.GetConversation(
                GetConversationRequest(conversation_id=conversation_id),
                cast(grpc.ServicerContext, FakeControlContext()),
            )
        assert err.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        harness.close()


def _run_agentapp_thread(
    result: dict[str, object],
    agent_stub: ServerAppIoStub,
    agent_task_input: AgentAppTaskInput,
) -> None:
    """Run the AgentApp executor helper and capture the outcome."""
    try:
        result["completed"] = _run_agentapp_task(
            agent_stub,
            agent_task_input,
            lambda _fab_id, _fab_version, _fab_hash: gpt_chat.app,
        )
    except BaseException as err:  # pylint: disable=broad-exception-caught
        result["exception"] = err


def _fake_model_provider(
    request: JsonObject,
    on_stream_event: Callable[[JsonObject], None] | None,
) -> ModelProviderResult:
    """Return one Responses-compatible provider result without network access."""
    assert request["model"] == "test-model"
    assert request["input"] == [{"role": "user", "content": "hello"}]
    if on_stream_event is not None:
        on_stream_event(
            {"type": "response.output_text.delta", "delta": "Hello from smoke"}
        )
        on_stream_event(
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-1",
                    "output_text": "Hello from the smoke test.",
                    "output": [
                        {
                            "type": "message",
                            "content": "Hello from the smoke test.",
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 5},
                },
            }
        )
    return ModelProviderResult(
        response={
            "id": "resp-1",
            "output_text": "Hello from the smoke test.",
            "output": [{"type": "message", "content": "Hello from the smoke test."}],
            "usage": {"input_tokens": 1, "output_tokens": 5},
        },
        events=[
            {"type": "response.output_text.delta", "delta": "Hello from smoke"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-1",
                    "output_text": "Hello from the smoke test.",
                    "output": [
                        {
                            "type": "message",
                            "content": "Hello from the smoke test.",
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 5},
                },
            },
        ],
    )


def _assert_conversation_outputs(
    harness: AgentAppMvpSmokeHarness,
    conversation_id: str,
) -> None:
    """Assert Control conversation RPCs expose the user and assistant items."""
    list_response = harness.control.ListConversations(
        ListConversationsRequest(limit=10),
        cast(grpc.ServicerContext, FakeControlContext()),
    )
    assert conversation_id in [
        conversation.conversation_id for conversation in list_response.conversations
    ]

    get_response = harness.control.GetConversation(
        GetConversationRequest(conversation_id=conversation_id),
        cast(grpc.ServicerContext, FakeControlContext()),
    )
    assert get_response.conversation.conversation_id == conversation_id
    items = [json.loads(item.item_json) for item in get_response.items]
    assert items[0] == {"role": "user", "content": "hello"}
    assistant_items = [item for item in items if item.get("role") == "assistant"]
    assert assistant_items == [
        {
            "role": "assistant",
            "content": "Hello from the smoke test.",
            "response_id": "resp-1",
            "model": "test-model",
        }
    ]


def _primary_task_id(state: LinkState, run_id: int) -> int:
    """Return the run's primary task ID."""
    run = state.get_run_info(run_ids=[run_id])[0]
    assert run.primary_task_id is not None
    return run.primary_task_id


def _task_by_id(state: LinkState, task_id: int) -> Task:
    """Return a task by ID or fail the smoke test."""
    tasks = state.get_tasks(task_ids=[task_id])
    if not tasks:
        raise AssertionError(f"Task {task_id} not found.")
    return tasks[0]


def _task_by_type(state: LinkState, run_id: int, task_type: TaskType) -> Task | None:
    """Return the first task of a type for a run."""
    for task in state.get_tasks(run_ids=[run_id]):
        if task.type == task_type:
            return task
    return None


def _wait_for(
    description: str,
    predicate: Callable[[], WaitResultT | None],
) -> WaitResultT:
    """Wait for a predicate to return a non-None value."""
    deadline = time.monotonic() + _SMOKE_TIMEOUT
    while time.monotonic() < deadline:
        result = predicate()
        if result is not None:
            return result
        time.sleep(_POLL_INTERVAL)
    raise AssertionError(f"Timed out waiting for {description}.")
