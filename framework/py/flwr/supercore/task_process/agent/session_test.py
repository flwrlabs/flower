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

from flwr.common.serde import user_config_from_proto, user_config_to_proto
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
from flwr.supercore.typing import JSONObject

from .session import RuntimeAgentResponses, _make_start_automation_tool


def test_start_automation_tool_exposes_only_input_and_schedule() -> None:
    """Keep the embedded run request out of the model-facing schema."""
    parameters = _make_start_automation_tool()["parameters"]

    assert isinstance(parameters, dict)
    properties = parameters["properties"]
    assert isinstance(properties, dict)
    assert "input" in properties
    assert "start_run_request" not in properties
    assert "task" not in properties
    assert parameters["required"] == ["input", "start_at"]


def test_call_automation_embeds_input_in_control_request() -> None:
    """Embed model input in the Control request sent to ServerAppIo."""
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
        start_run_request=StartRunRequest(
            app_spec="example/app",
            override_config=user_config_to_proto({"existing": "value"}),
            federation="@account/federation",
            series_id=2,
        ),
    )
    arguments: JSONObject = {
        "input": "Do work",
        "start_at": "2026-07-28T12:00:00Z",
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
    assert user_config_from_proto(request.start_run_request.override_config) == {
        "existing": "value",
        "agent.input": "Do work",
    }
    assert request.fixed_interval == 60
    assert request.max_runs == 3
    assert request.start_at == "2026-07-28T12:00:00Z"


def test_create_connector_response_canonicalizes_name() -> None:
    """Task creation and its request message should use the canonical name."""
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
