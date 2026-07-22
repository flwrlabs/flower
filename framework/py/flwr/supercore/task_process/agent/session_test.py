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
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)

from .session import RuntimeAgentResponses


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
