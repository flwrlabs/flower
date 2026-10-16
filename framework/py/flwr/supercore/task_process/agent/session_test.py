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
"""Runtime AgentApp session tests."""

from logging import ERROR
from unittest.mock import Mock, call, patch

import httpx
import pytest

from flwr.agentapp import RUN_SUMMARY_EVENT_TYPES
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
    StartRunRequest,
)
from flwr.proto.runseries_pb2 import CompactedContext  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
    GetCompactedContextsRequest,
    GetCompactedContextsResponse,
    GetRunSeriesEventsRequest,
    GetRunSeriesEventsResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushTaskEventsRequest,
    PutCompactedContextRequest,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.error import ApiErrorCode
from flwr.supercore.json_message.connector_message import ConnectorResponse
from flwr.supercore.retry import make_simple_http_retry_invoker
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.task_process.connector import registry as connector_registry
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.typing import JSONObject

from .session import (
    AgentRuntime,
    RuntimeAgentConnectors,
    RuntimeAgentEvents,
    RuntimeAgentSession,
)


@pytest.fixture(autouse=True)
def task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the task identity used by Agent task messages."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 789)
    monkeypatch.setattr(TaskIdentity, "_run_id", 123)
    monkeypatch.setattr(TaskIdentity, "_node_id", 456)


def test_emit_event_pushes_task_event() -> None:
    """Emit should translate a structured run event to the Runtime API."""
    stub = Mock()
    events = RuntimeAgentEvents(stub)
    event: JSONObject = {
        "type": "response.output_text.delta",
        "content": {"delta": "Hello"},
    }

    events.emit(event)
    content = event["content"]
    assert isinstance(content, dict)
    content["delta"] = "Changed"
    events.close()

    expected_event = TaskEvent(
        event="response.output_text.delta",
        data=('{"type":"response.output_text.delta","content":{"delta":"Hello"}}'),
    )
    stub.PushTaskEvents.assert_called_once_with(
        PushTaskEventsRequest(events=[expected_event])
    )


def test_runtime_agent_session_exposes_prompt() -> None:
    """Expose the initial prompt without additional runtime calls."""
    session = RuntimeAgentSession(
        prompt="Initial prompt",
        connectors=Mock(),
        events=Mock(),
        grid=Mock(),
    )

    assert session.prompt == "Initial prompt"


def test_close_drains_events_before_worker_stops() -> None:
    """Close should publish queued events before stopping the worker."""
    stub = Mock()
    with patch("flwr.supercore.task_process.agent.session.Thread") as thread_cls:
        thread_cls.return_value.is_alive.return_value = False
        events = RuntimeAgentEvents(stub)
        worker_target = thread_cls.call_args.kwargs["target"]
        thread_cls.return_value.join.side_effect = lambda _timeout: worker_target()

        event: JSONObject = {
            "type": "response.output_text.delta",
            "delta": "Hello",
        }
        events.emit(event)
        events.close()

    expected_event = TaskEvent(
        event="response.output_text.delta",
        data='{"type":"response.output_text.delta","delta":"Hello"}',
    )
    stub.PushTaskEvents.assert_called_once_with(
        PushTaskEventsRequest(events=[expected_event])
    )


def test_emit_event_requires_type() -> None:
    """Emit should reject events without a valid type."""
    stub = Mock()
    events = RuntimeAgentEvents(stub)

    with pytest.raises(
        ValueError, match="Run event requires a non-empty string 'type' field"
    ):
        events.emit({"message": "Hello"})
    events.close()

    stub.PushTaskEvents.assert_not_called()


def test_get_trace_forwards_filters_and_decodes_events() -> None:
    """Forward event filters and return parsed event envelopes."""
    stub = Mock()
    stub.GetRunSeriesEvents.return_value = GetRunSeriesEventsResponse(
        events=[
            TaskEvent(
                id=12,
                timestamp="2026-09-08T12:00:00+00:00",
                run_id=34,
                task_id=56,
                event="response.completed",
                data='{"type":"response.completed","response":{"id":"resp_1"}}',
            )
        ]
    )
    events = RuntimeAgentEvents(stub)

    trace = events.get_trace(
        event_types=RUN_SUMMARY_EVENT_TYPES,
        after_event_id=10,
        before_event_id=20,
        last_n=1,
    )
    events.close()

    stub.GetRunSeriesEvents.assert_called_once_with(
        GetRunSeriesEventsRequest(
            event_types=RUN_SUMMARY_EVENT_TYPES,
            after_event_id=10,
            before_event_id=20,
            last_n=1,
        )
    )
    assert trace == [
        {
            "id": 12,
            "timestamp": "2026-09-08T12:00:00+00:00",
            "run_id": 34,
            "task_id": 56,
            "event": "response.completed",
            "data": {"type": "response.completed", "response": {"id": "resp_1"}},
        }
    ]


def test_get_trace_without_filters_requests_all_events() -> None:
    """Keep the default request unfiltered."""
    stub = Mock()
    stub.GetRunSeriesEvents.return_value = GetRunSeriesEventsResponse()
    events = RuntimeAgentEvents(stub)

    assert events.get_trace() == []
    events.close()

    stub.GetRunSeriesEvents.assert_called_once_with(GetRunSeriesEventsRequest())


def test_get_compacted_contexts_emits_returned_ids() -> None:
    """Return summaries and emit their IDs."""
    stub = Mock()
    stub.GetCompactedContexts.return_value = GetCompactedContextsResponse(
        contexts=[
            CompactedContext(
                id=12,
                series_id=42,
                through_event_id=9,
                text="Private summary",
                created_at="2026-10-16T12:00:00+00:00",
            )
        ]
    )
    events = RuntimeAgentEvents(stub)

    with patch.object(events, "emit") as emit:
        contexts = events.get_compacted_contexts(limit=2, before_context_id=20)
    events.close()

    stub.GetCompactedContexts.assert_called_once_with(
        GetCompactedContextsRequest(limit=2, before_context_id=20)
    )
    assert contexts == [
        {
            "id": 12,
            "series_id": 42,
            "through_event_id": 9,
            "text": "Private summary",
            "created_at": "2026-10-16T12:00:00+00:00",
        }
    ]
    emit.assert_called_once_with(
        {"type": "compacted_contexts.get.completed", "context_ids": [12]}
    )


def test_put_compacted_context_emits_start_and_success() -> None:
    """Emit events before and after a successful write."""
    stub = Mock()
    events = RuntimeAgentEvents(stub)
    order: list[JSONObject | str] = []

    def record_emit(event: JSONObject) -> None:
        order.append(event)

    stub.PutCompactedContext.side_effect = lambda _request: order.append("request")

    with patch.object(events, "emit", side_effect=record_emit):
        events.put_compacted_context(through_event_id=9, text="Private summary")
    events.close()

    assert order == [
        {"type": "compacted_context.put.started", "through_event_id": 9},
        "request",
        {"type": "compacted_context.put.succeeded", "through_event_id": 9},
    ]
    stub.PutCompactedContext.assert_called_once_with(
        PutCompactedContextRequest(
            context=CompactedContext(through_event_id=9, text="Private summary")
        )
    )


def test_put_compacted_context_emits_failure_and_reraises() -> None:
    """Emit a safe error code and re-raise the request failure."""
    stub = Mock()
    request = httpx.Request("POST", "https://example.com/put-compacted-context")
    response = httpx.Response(
        409,
        json={
            "code": ApiErrorCode.RUNTIME_COMPACTED_CONTEXT_ALREADY_EXISTS,
            "detail": "Compacted context already exists.",
        },
        request=request,
    )
    error = httpx.HTTPStatusError("Duplicate", request=request, response=response)
    stub.PutCompactedContext.side_effect = error
    events = RuntimeAgentEvents(stub)

    with patch.object(events, "emit") as emit:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            events.put_compacted_context(through_event_id=9, text="Private summary")
    events.close()

    assert caught.value is error
    assert emit.call_args_list == [
        call({"type": "compacted_context.put.started", "through_event_id": 9}),
        call(
            {
                "type": "compacted_context.put.failed",
                "through_event_id": 9,
                "error_type": "HTTPStatusError",
                "error_code": ApiErrorCode.RUNTIME_COMPACTED_CONTEXT_ALREADY_EXISTS,
            }
        ),
    ]


def test_context_event_failure_does_not_change_write_outcome() -> None:
    """A failed event publisher must not turn a successful write into an error."""
    stub = Mock()
    events = RuntimeAgentEvents(stub)

    with patch.object(events, "emit", side_effect=RuntimeError("Publisher failed")):
        events.put_compacted_context(through_event_id=9, text="Summary")
    events.close()

    stub.PutCompactedContext.assert_called_once()


def test_agent_events_and_connector_items_use_same_publisher() -> None:
    """Publish explicit AgentApp events and connector items through one publisher."""
    stub = Mock()
    events = Mock()
    agent_runtime = AgentRuntime(
        stub=stub,
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=events,
    )
    model_event: JSONObject = {
        "type": "response.output_text.delta",
        "delta": "Hello",
    }
    connector_event: JSONObject = {
        "type": "function_call",
        "call_id": "call-1",
        "name": "web_search",
        "arguments": '{"query":"Flower"}',
    }

    events.emit(model_event)
    agent_runtime.push_run_events([connector_event])

    assert events.emit.call_args_list == [
        call(model_event),
        call(connector_event),
    ]


def test_pull_task_messages_filters_by_child_task() -> None:
    """Claim only messages sent by the expected child task."""
    stub = Mock()
    stub.PullTaskMessage.return_value = PullTaskMessageResponse()
    agent_runtime = AgentRuntime(
        stub=stub,
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=Mock(),
    )

    assert agent_runtime._pull_task_messages(456) == []  # pylint: disable=W0212
    stub.PullTaskMessage.assert_called_once_with(
        PullTaskMessageRequest(limit=1, src_task_id=456)
    )


def test_start_automation_tool_exposes_only_input_and_schedule() -> None:
    """Keep the embedded run request out of the model-facing schema."""
    # Prepare
    expected_properties = {"input", "start_at", "fixed_interval", "max_runs"}

    # Execute
    parameters = connector_registry.get_connector_tools(START_AUTOMATION_TOOL_NAME)[0][
        "parameters"
    ]

    # Assert
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == expected_properties
    assert parameters["required"] == ["input", "start_at"]


def test_runtime_connectors_expand_one_connector_into_multiple_tools() -> None:
    """One connector reference can advertise multiple model-facing tools."""
    runtime = Mock()
    connectors = RuntimeAgentConnectors(runtime)
    tools: list[JSONObject] = [
        {"type": "function", "name": "example_search"},
        {"type": "function", "name": "example_read"},
    ]

    runtime.get_connector_tools.return_value = tools
    assert connectors.tools(["example"]) == tools
    runtime.get_connector_tools.assert_called_once_with("example", refresh=False)


def test_call_automation_embeds_input_in_control_request() -> None:
    """Embed model input in the Control request sent to the Runtime API."""
    # Prepare
    stub = Mock()
    stub.StartAutomation.return_value = StartAutomationResponse()
    start_run_request = StartRunRequest(
        app_spec="example/app",
        override_config=user_config_to_proto({"existing": "value"}),
        federation="@account/federation",
        series_id=2,
    )
    agent_runtime = AgentRuntime(
        stub=stub,
        run_id=123,
        task_id=789,
        start_run_request=start_run_request,
        events=Mock(),
    )
    arguments: JSONObject = {
        "input": "Do work",
        "start_at": "2026-07-28T12:00:00Z",
        "fixed_interval": 60,
        "max_runs": 3,
    }

    # Execute
    with patch.object(agent_runtime, "push_run_events") as push_run_events:
        agent_runtime.call_automation_with_events(call_id="call-1", arguments=arguments)

    # Assert
    request = stub.StartAutomation.call_args.args[0]
    assert request == StartAutomationRequest(
        start_at="2026-07-28T12:00:00Z",
        fixed_interval=60,
        max_runs=3,
        start_run_request=StartRunRequest(
            app_spec="example/app",
            override_config=user_config_to_proto({"existing": "value"}),
            federation="@account/federation",
            series_id=2,
            user_prompt="Do work",
        ),
    )
    items = [item.args[0][0] for item in push_run_events.call_args_list]
    assert [item["type"] for item in items] == [
        "function_call",
        "function_call_output",
    ]
    assert items[0]["name"] == START_AUTOMATION_TOOL_NAME


def test_connector_call_emits_standard_items() -> None:
    """Emit standard function call and output items."""
    agent_runtime = AgentRuntime(
        stub=Mock(),
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=Mock(),
    )
    arguments: JSONObject = {"query": "Flower"}

    with (
        patch.object(
            agent_runtime, "create_connector_response", return_value={"results": []}
        ),
        patch.object(agent_runtime, "push_run_events") as push_run_events,
    ):
        output = agent_runtime.call_connector_with_events(
            name="notion_search", call_id="call-1", arguments=arguments
        )

    assert output == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"results":[]}',
    }
    assert push_run_events.call_args_list == [
        call(
            [
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "notion_search",
                    "arguments": '{"query":"Flower"}',
                }
            ]
        ),
        call([output]),
    ]


def test_create_connector_response_resolves_canonical_name() -> None:
    """Resolve normalized names and forward original dynamic names and arguments."""
    stub = Mock()
    stub.CreateTask.return_value = CreateTaskResponse(task_id=456)
    agent_runtime = AgentRuntime(
        stub=stub,
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=Mock(),
    )
    agent_runtime._discovered_tools["example"] = [  # pylint: disable=protected-access
        {"name": "example_Search"}
    ]
    reply = ConnectorResponse(
        dst_task_id=789,
        name="example_Search",
        call_id="call-1",
        output="done",
        error=None,
        reply_to_message_id="request-message-id",
    )

    with (
        patch.dict(
            connector_registry._CONNECTORS_BY_REF,  # pylint: disable=protected-access
            {"example": Mock(ref="example", tools=(), tool_source=Mock())},
        ),
        patch.object(
            agent_runtime, "_send_and_receive", return_value=reply
        ) as send_and_receive,
    ):
        output = agent_runtime.create_connector_response(
            name=" EXAMPLE_search ",
            call_id="call-1",
            arguments={"limit": 3},
        )

    stub.CreateTask.assert_called_once_with(
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref="example")
    )
    request = send_and_receive.call_args.args[0]
    assert request.payload["name"] == "example_Search"
    assert request.payload["arguments"] == {"limit": 3}
    assert output == "done"


def test_create_connector_response_uses_connector_task_for_filesystem() -> None:
    """Filesystem calls should execute through a Connector task."""
    stub = Mock()
    stub.CreateTask.return_value = CreateTaskResponse(task_id=456)
    agent_runtime = AgentRuntime(
        stub=stub,
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=Mock(),
    )
    reply = ConnectorResponse(
        dst_task_id=789,
        name="filesystem_list_directory",
        call_id="call-1",
        output={"entries": []},
        error=None,
        reply_to_message_id="request-message-id",
    )

    with patch.object(agent_runtime, "_send_and_receive", return_value=reply):
        output = agent_runtime.create_connector_response(
            name="filesystem_list_directory",
            call_id="call-1",
            arguments={"path": "/allowed"},
        )

    assert output == {"entries": []}
    stub.CreateTask.assert_called_once_with(
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref="filesystem")
    )


def test_connector_tools_include_agent_handled_automation() -> None:
    """Automation and connector schemas should remain available in request order."""
    runtime = Mock()
    runtime.get_connector_tools.side_effect = (
        lambda name, **_kwargs: connector_registry.get_connector_tools(name)
    )
    connectors = RuntimeAgentConnectors(runtime)

    assert [
        tool["name"]
        for tool in connectors.tools(
            ["web_search", START_AUTOMATION_TOOL_NAME, "notion"]
        )
    ] == [
        "web_search",
        "start_automation",
        "notion_search",
        "notion_get_page",
        "notion_get_page_property",
        "notion_get_database",
        "notion_get_block",
        "notion_get_block_children",
        "notion_query_meeting_notes",
        "notion_list_users",
        "notion_get_user",
        "notion_get_self",
    ]


def test_runtime_discovers_and_caches_tools() -> None:
    """Discover once, return independent copies, and refresh explicitly."""
    runtime = AgentRuntime(
        stub=Mock(),
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=Mock(),
    )
    tools: list[JSONObject] = [{"name": "example_Read"}]
    with (
        patch.dict(
            connector_registry._CONNECTORS_BY_REF,  # pylint: disable=protected-access
            {"example": Mock(ref="example", tools=(), tool_source=Mock())},
        ),
        patch.object(
            runtime, "create_connector_response", return_value=tools
        ) as discover,
    ):
        connectors = RuntimeAgentConnectors(runtime)
        first = connectors.tools(["example"])
        assert first == [{"name": "example_read"}]
        first[0]["name"] = "Changed"
        assert connectors.tools(["example"]) == [{"name": "example_read"}]
        discover.assert_called_once()
        connectors.tools(["example"], refresh=True)
        assert discover.call_count == 2


@pytest.mark.parametrize(
    "status,code,unavailable",
    [
        (403, ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE, True),
        (403, ApiErrorCode.NO_PERMISSIONS, False),
        (500, ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE, False),
    ],
)
def test_discovery_skips_only_connectors_unavailable_to_run(
    status: int, code: int, unavailable: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Optional discovery must not abort an Agent without connector access."""
    response = httpx.Response(
        status,
        json={"code": code, "detail": "Connector is not available to this run."},
        request=httpx.Request("POST", "https://runtime/v1/runtime/create-task"),
    )
    error = httpx.HTTPStatusError(
        "Discovery failed", request=response.request, response=response
    )
    stub = Mock()
    stub.CreateTask.side_effect = error
    runtime = AgentRuntime(
        stub=stub,
        run_id=123,
        task_id=789,
        start_run_request=StartRunRequest(),
        events=Mock(),
    )
    connectors = RuntimeAgentConnectors(runtime)
    if unavailable:
        assert connectors.tools(
            ["github", "web_search"]
        ) == connector_registry.get_connector_tools("web_search")
        assert any(
            record.levelno == ERROR
            and record.getMessage()
            == "Skipping tool discovery for unavailable connector github "
            "in run 123, task 789 (API error 3010)."
            for record in caplog.records
        )
    else:
        with pytest.raises(httpx.HTTPStatusError) as raised:
            connectors.tools(["github"])
        assert raised.value is error
    stub.CreateTask.assert_called_once_with(
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref="github")
    )


def test_discovery_skips_unavailable_connectors_through_http_retry() -> None:
    """Keep the Agent running when real transport rejects optional discovery."""
    requests: list[CreateTaskRequest] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/runtime/create-task"
        requests.append(CreateTaskRequest.FromString(request.content))
        return httpx.Response(
            403,
            json={
                "code": ApiErrorCode.RUNTIME_CONNECTOR_NOT_AVAILABLE,
                "detail": "Connector is not available to this run.",
            },
        )

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as transport_client,
        patch(
            "flwr.supercore.protobuf.client.httpx.Client",
            return_value=transport_client,
        ),
        RuntimeHttpClient(
            "http://runtime.example",
            retry_invoker=make_simple_http_retry_invoker(),
        ) as client,
    ):
        runtime = AgentRuntime(
            stub=client,
            run_id=123,
            task_id=789,
            start_run_request=StartRunRequest(),
            events=Mock(),
        )
        assert RuntimeAgentConnectors(runtime).tools(
            ["github", "web_search", "slack"]
        ) == connector_registry.get_connector_tools("web_search")
        # Calling a connector still reports the authorization failure.
        with pytest.raises(httpx.HTTPStatusError):
            runtime.create_connector_response(
                name="github__discover_tools", call_id="call-1", arguments={}
            )

    assert requests == [
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref=ref)
        for ref in ("github", "slack", "github")
    ]
