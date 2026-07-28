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

from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    CreateTaskRequest,
    CreateTaskResponse,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
)
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.typing import JSONObject

from .session import RuntimeAgentResponses, _make_start_automation_tool


def test_start_automation_tool_uses_control_request() -> None:
    """Expose the complete Control StartAutomation request."""
    parameters = _make_start_automation_tool()["parameters"]

    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert "start_run_request" in properties
    assert "task" not in properties


def test_call_automation_uses_control_request() -> None:
    """Send the complete Control StartAutomation request to ServerAppIo."""
    stub = Mock()
    stub.StartAutomation.return_value = StartAutomationResponse(
        automation_id=1,
        series_id=2,
        next_run_at="2026-07-28T12:00:00Z",
    )
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
    )
    arguments: JSONObject = {
        "start_run_request": {
            "app_spec": "example/app",
            "federation": "@account/federation",
            "series_id": 2,
            "connector_refs": ["calendar"],
            "override_config": {"agent.input": {"string": "Do work"}},
        },
        "fixed_interval": 60,
        "max_runs": 3,
    }

    with (
        patch.object(responses, "append_and_push_run_events"),
        patch.object(responses, "append_context_items"),
    ):
        responses.call_automation_with_events(call_id="call-1", arguments=arguments)

    request = stub.StartAutomation.call_args.args[0]
    assert isinstance(request, StartAutomationRequest)
    assert request.start_run_request.app_spec == "example/app"
    assert request.start_run_request.federation == "@account/federation"
    assert request.start_run_request.series_id == 2
    assert list(request.start_run_request.connector_refs) == ["calendar"]
    assert request.start_run_request.override_config["agent.input"].string == "Do work"
    assert request.fixed_interval == 60
    assert request.max_runs == 3


def test_create_connector_response_canonicalizes_name() -> None:
    """Task creation and its request message should use the canonical name."""
    stub = Mock()
    stub.CreateTask.return_value = CreateTaskResponse(task_id=456)
    responses = RuntimeAgentResponses(
        stub=stub,
        run_id=123,
        task_id=789,
        context=Mock(),
    )
    reply = ConnectorResponse(
        dst_task_id=789,
        name="notion",
        call_id="call-1",
        output="done",
        error=None,
        reply_to_message_id="request-message-id",
    )

    with patch.object(
        responses, "_send_and_receive", return_value=reply
    ) as send_and_receive:
        output = responses.create_connector_response(
            name=" NoTiOn ",
            call_id="call-1",
            arguments={},
        )

    stub.CreateTask.assert_called_once_with(
        CreateTaskRequest(type=TaskType.CONNECTOR, connector_ref="notion")
    )
    request = send_and_receive.call_args.args[0]
    assert isinstance(request, ConnectorRequest)
    assert request.payload["name"] == "notion"
    assert output == "done"
