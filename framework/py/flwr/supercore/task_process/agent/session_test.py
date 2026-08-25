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

from flwr.common.serde import message_to_proto, user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
    StartRunRequest,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
    PullTaskEventsRequest,
    PullTaskEventsResponse,
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

from .session import RuntimeAgentConnectors, RuntimeAgentEvents, RuntimeAgentResponses


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


def test_agent_events_and_connector_events_use_same_publisher() -> None:
    """Publish explicit and built-in AgentApp events through one publisher."""
    stub = Mock()
    events = Mock()
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
        start_run_request=StartRunRequest(),
        events=events,
    )
    model_event: JSONObject = {
        "type": "response.output_text.delta",
        "delta": "Hello",
    }
    connector_event: JSONObject = {"type": "response.tool_call.started"}

    events.emit(model_event)
    responses.push_run_events([connector_event])

    assert events.emit.call_args_list == [
        call(model_event),
        call(connector_event),
    ]


def test_pull_task_messages_filters_by_child_task() -> None:
    """Claim only messages sent by the expected child task."""
    stub = Mock()
    stub.PullTaskMessage.return_value = PullTaskMessageResponse()
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
        start_run_request=StartRunRequest(),
        events=Mock(),
    )

    assert responses._pull_task_messages(456) == []  # pylint: disable=W0212
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
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
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
    with (
        patch.object(responses, "append_and_push_run_events"),
        patch.object(responses, "append_context_items"),
    ):
        responses.call_automation_with_events(call_id="call-1", arguments=arguments)

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


def test_create_connector_response_resolves_canonical_name() -> None:
    """Task creation should resolve the canonical tool name to its connector."""
    stub = Mock()
    stub.CreateTask.return_value = CreateTaskResponse(task_id=456)
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
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
    on_event = Mock()

    with (
        patch(
            "flwr.supercore.task_process.agent.session.get_connector_ref",
            return_value="notion",
        ) as get_connector_ref,
        patch.object(
            responses, "_send_and_receive", return_value=reply
        ) as send_and_receive,
    ):
        output = responses.create_connector_response(
            name=" NoTiOn_Search ",
            call_id="call-1",
            arguments={},
            on_event=on_event,
        )

    get_connector_ref.assert_called_once_with("notion_search")
    stub.CreateTask.assert_called_once_with(
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref="notion")
    )
    request = send_and_receive.call_args.args[0]
    assert isinstance(request, ConnectorRequest)
    assert request.payload["name"] == "notion_search"
    assert send_and_receive.call_args.kwargs == {"on_event": on_event}
    assert output == "done"


def test_send_and_receive_delivers_child_events_before_returning_reply() -> None:
    """Deliver child events, including events found by the final race-closing pull."""
    stub = Mock()
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
        start_run_request=StartRunRequest(),
        events=Mock(),
    )
    request = ConnectorRequest(
        dst_task_id=456,
        name="notion_search",
        call_id="call-1",
        arguments={},
    )
    first_event = {"type": "connector.progress", "progress": 25}
    final_event = {"type": "connector.progress", "progress": 100}
    stub.PullTaskEvents.side_effect = [
        PullTaskEventsResponse(
            events=[
                TaskEvent(
                    id=1,
                    event="connector.progress",
                    data='{"type":"connector.progress","progress":25}',
                )
            ]
        ),
        PullTaskEventsResponse(
            events=[
                TaskEvent(
                    id=2,
                    event="connector.progress",
                    data='{"type":"connector.progress","progress":100}',
                )
            ]
        ),
    ]
    def pull_task_message(_: PullTaskMessageRequest) -> PullTaskMessageResponse:
        reply = ConnectorResponse(
            dst_task_id=789,
            name="notion_search",
            call_id="call-1",
            output="done",
            error=None,
            reply_to_message_id=request.metadata.message_id,
        )
        return PullTaskMessageResponse(messages=[message_to_proto(reply)])

    stub.PullTaskMessage.side_effect = pull_task_message
    received_events: list[JSONObject] = []

    response = responses._send_and_receive(  # pylint: disable=W0212
        request,
        on_event=received_events.append,
    )

    assert response.metadata.reply_to_message_id == request.metadata.message_id
    assert received_events == [first_event, final_event]
    assert stub.PullTaskEvents.call_args_list == [
        call(PullTaskEventsRequest(task_id=456)),
        call(PullTaskEventsRequest(task_id=456, after_task_event_id=1)),
    ]
