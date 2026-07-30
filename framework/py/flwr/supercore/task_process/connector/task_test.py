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
"""Connector task credential-resolution tests."""

import traceback
from unittest.mock import ANY, Mock, patch

import pytest

from flwr.common.serde import message_from_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
)
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)

from . import registry
from .task import handle_task


def _connector_request(name: str) -> ConnectorRequest:
    """Build a connector request with routed task metadata."""
    request = ConnectorRequest(
        dst_task_id=22,
        name=name,
        call_id="call-1",
        arguments={"query": "release notes"},
    )
    request.metadata.__dict__["_run_id"] = 7
    request.metadata.__dict__["_message_id"] = "request-message-id"
    request.metadata.src_task_id = 11
    return request


def _pushed_response(stub: Mock) -> ConnectorResponse:
    """Parse the connector response pushed through a mocked stub."""
    pushed = stub.PushTaskMessage.call_args.args[0].message
    return ConnectorResponse.from_message(message_from_proto(pushed))


def test_handle_task_passes_credentials_to_matching_provider() -> None:
    """Credential-backed providers should receive decoded credentials and config."""
    tool_name = "notion_search"
    stub = Mock()
    stub.GetConnector.return_value = GetConnectorResponse(
        connector_ref="notion",
        credentials_json='{"token":"secret"}',
        config_json='{"workspace":"primary"}',
    )
    provider = Mock(return_value={"pages": 3})

    with (
        patch(
            "flwr.supercore.task_process.connector.task._pull_connector_request",
            return_value=_connector_request(tool_name),
        ),
        patch.dict(
            registry._CREDENTIAL_CONNECTOR_HANDLERS,  # pylint: disable=protected-access
            {tool_name: provider},
            clear=True,
        ),
        patch.dict(
            registry._CREDENTIAL_CONNECTOR_REFS,  # pylint: disable=protected-access
            {tool_name: "notion"},
            clear=True,
        ),
    ):
        handle_task(stub=stub, task_id=22, run_id=7)

    stub.GetConnector.assert_called_once_with(GetConnectorRequest())
    provider.assert_called_once_with(
        query="release notes",
        credentials={"token": "secret"},
        config={"workspace": "primary"},
        usage_recorder=ANY,
    )
    assert _pushed_response(stub).payload == {
        "name": tool_name,
        "call_id": "call-1",
        "output": {"pages": 3},
        "error": None,
    }


def test_handle_task_rejects_credentials_for_different_connector() -> None:
    """Credential-backed providers should receive only their connector's secrets."""
    stub = Mock()
    stub.GetConnector.return_value = GetConnectorResponse(
        connector_ref="notion",
        credentials_json='{"token":"secret"}',
        config_json="{}",
    )
    provider = Mock()

    with (
        patch(
            "flwr.supercore.task_process.connector.task._pull_connector_request",
            return_value=_connector_request("github"),
        ),
        patch.dict(
            registry._CREDENTIAL_CONNECTOR_HANDLERS,  # pylint: disable=protected-access
            {"github": provider},
            clear=True,
        ),
        pytest.raises(
            RuntimeError, match="Credential-backed connector execution failed."
        ),
    ):
        handle_task(stub=stub, task_id=22, run_id=7)

    provider.assert_not_called()


def test_handle_task_does_not_expose_credentials_in_provider_errors() -> None:
    """Credential-backed provider failures should not expose secret values."""
    secret = "TOP-SECRET-TOKEN"
    stub = Mock()
    stub.GetConnector.return_value = GetConnectorResponse(
        connector_ref="notion",
        credentials_json=f'{{"token":"{secret}"}}',
        config_json="{}",
    )
    provider = Mock(side_effect=RuntimeError(f"Provider rejected {secret}"))

    with (
        patch(
            "flwr.supercore.task_process.connector.task._pull_connector_request",
            return_value=_connector_request("notion"),
        ),
        patch.dict(
            registry._CREDENTIAL_CONNECTOR_HANDLERS,  # pylint: disable=protected-access
            {"notion": provider},
            clear=True,
        ),
        pytest.raises(RuntimeError) as error,
    ):
        handle_task(stub=stub, task_id=22, run_id=7)

    response = _pushed_response(stub)
    provider.assert_called_once()
    assert str(error.value) == "Credential-backed connector execution failed."
    assert response.payload["error"] == {
        "code": "connector_error",
        "message": "Connector execution failed.",
    }
    assert secret not in str(error.value)
    assert secret not in str(response.payload)
    assert secret not in "".join(traceback.format_exception(error.value))
    assert error.value.__context__ is None
