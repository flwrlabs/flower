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


from unittest.mock import Mock, call, patch

import pytest

from flwr.agentapp.constants import (
    AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY,
    AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY,
)
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
    StartRunRequest,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
    GetRunSeriesEventsRequest,
    GetRunSeriesEventsResponse,
    PullTaskMessageRequest,
    PullTaskMessageResponse,
    PushTaskEventsRequest,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.task_process.connector.registry import get_builtin_connector_tool
from flwr.supercore.typing import JSONObject

from .session import (
    AgentRuntime,
    RuntimeAgentConnectors,
    RuntimeAgentEvents,
    RuntimeAgentGrid,
)


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


def test_get_trace_gets_current_run_series_events() -> None:
    """Get trace should return parsed event envelopes from the Runtime API."""
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

    trace = events.get_trace()
    events.close()

    stub.GetRunSeriesEvents.assert_called_once_with(GetRunSeriesEventsRequest())
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
    parameters = get_builtin_connector_tool(START_AUTOMATION_TOOL_NAME)["parameters"]

    # Assert
    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == expected_properties
    assert parameters["required"] == ["input", "start_at"]


def test_runtime_connectors_expand_one_connector_into_multiple_tools() -> None:
    """One connector reference can advertise multiple model-facing tools."""
    connectors = RuntimeAgentConnectors(Mock())
    tools: list[JSONObject] = [
        {"type": "function", "name": "example_search"},
        {"type": "function", "name": "example_read"},
    ]

    with patch(
        "flwr.supercore.task_process.agent.session.get_connector_tools",
        return_value=tools,
    ) as get_connector_tools:
        assert connectors.tools(["example"]) == tools

    get_connector_tools.assert_called_once_with("example")


def test_runtime_agent_grid_tools() -> None:
    """Grid tools should sample nodes, send content, and return serialized replies."""
    grid = Mock()
    grid.get_node_ids.return_value = [11, 22]
    grid.push_messages.return_value = ["message-1"]
    reply = Message(
        RecordDict({"result": ConfigRecord({"answer": "done"})}),
        dst_node_id=0,
        message_type="query",
    )
    reply.metadata.__dict__["_reply_to_message_id"] = "message-1"
    grid.pull_messages.return_value = [reply]
    events = Mock()
    agent_grid = RuntimeAgentGrid(grid, events)

    assert [tool["name"] for tool in agent_grid.tools()] == [
        "get_nodes",
        "push_message",
        "pull_messages",
    ]
    get_nodes = agent_grid.call(
        {
            "name": "get_nodes",
            "call_id": "call-1",
            "arguments": '{"sample_size":1}',
        }
    )
    assert get_nodes["output"] in (
        '{"node_ids":["11"],"num_available":2}',
        '{"node_ids":["22"],"num_available":2}',
    )

    pushed = agent_grid.call(
        {
            "name": "push_message",
            "call_id": "call-2",
            "arguments": {
                "dst_node_id": "11",
                "payload": {"prompt": "hi", "values": [[1, 2], [3, 4]]},
            },
        }
    )
    assert pushed["output"] == '{"message_id":"message-1"}'
    grid.create_message.assert_not_called()
    sent = list(grid.push_messages.call_args.args[0])[0]
    assert sent.metadata.dst_node_id == 11
    assert sent.metadata.message_type == "query"
    assert sent.metadata.group_id == ""
    assert (
        sent.content[AGENT_GRID_MESSAGE_PAYLOAD_RECORD_KEY][
            AGENT_GRID_MESSAGE_PAYLOAD_JSON_KEY
        ]
        == '{"prompt":"hi","values":[[1,2],[3,4]]}'
    )

    pulled = agent_grid.call(
        {
            "name": "pull_messages",
            "call_id": "call-3",
            "arguments": {"message_ids": ["message-1"], "timeout": 0},
        }
    )
    pulled_output = pulled["output"]
    assert isinstance(pulled_output, str)
    assert '"replyToMessageId":"message-1"' in pulled_output
    assert '"pending_message_ids":[]' in pulled_output
    assert events.emit.call_count == 6


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
            override_config=user_config_to_proto(
                {"existing": "value", "agent.input": "Do work"}
            ),
            federation="@account/federation",
            series_id=2,
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
    """Task creation should resolve the canonical tool name to its connector."""
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
        name="notion_search",
        call_id="call-1",
        output="done",
        error=None,
        reply_to_message_id="request-message-id",
    )

    with (
        patch(
            "flwr.supercore.task_process.agent.session.get_connector_ref",
            return_value="notion",
        ) as get_connector_ref,
        patch.object(
            agent_runtime, "_send_and_receive", return_value=reply
        ) as send_and_receive,
    ):
        output = agent_runtime.create_connector_response(
            name=" NoTiOn_Search ",
            call_id="call-1",
            arguments={},
        )

    get_connector_ref.assert_called_once_with("notion_search")
    stub.CreateTask.assert_called_once_with(
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref="notion")
    )
    request = send_and_receive.call_args.args[0]
    assert isinstance(request, ConnectorRequest)
    assert request.payload["name"] == "notion_search"
    assert output == "done"
