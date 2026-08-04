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


from unittest.mock import Mock, patch

from flwr.common.serde import user_config_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
    StartRunRequest,
)
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.task_process.connector.automation import START_AUTOMATION_TOOL_NAME
from flwr.supercore.task_process.connector.registry import get_builtin_connector_tool
from flwr.supercore.typing import JSONObject

from .session import RuntimeAgentResponses


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


def test_call_automation_embeds_input_in_control_request() -> None:
    """Embed model input in the Control request sent to ServerAppIo."""
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
            responses, "_send_and_receive", return_value=reply
        ) as send_and_receive,
    ):
        output = responses.create_connector_response(
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
