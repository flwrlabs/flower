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
from typing import Any
from unittest.mock import Mock, patch

import pytest

from flwr.common.serde import message_from_proto
from flwr.proto.appio_pb2 import GetConnectorRequest, GetConnectorResponse
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)
from flwr.supercore.typing import JSONObject

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
    stub = Mock()
    stub.GetConnector.return_value = GetConnectorResponse(
        connector_ref="notion",
        credentials_json='{"token":"secret"}',
        config_json='{"workspace":"primary"}',
    )
    captured: dict[str, Any] = {}

    def invoke_provider(
        *,
        query: str,
        credentials: JSONObject,
        config: JSONObject,
        usage_recorder: object,
    ) -> JSONObject:
        captured.update(
            query=query,
            credentials=credentials,
            config=config,
            usage_recorder=usage_recorder,
        )
        return {"pages": 3}

    with (
        patch(
            "flwr.supercore.task_process.connector.task._pull_connector_request",
            return_value=_connector_request("notion"),
        ),
        patch.dict(
            registry._CREDENTIAL_CONNECTOR_HANDLERS,  # pylint: disable=protected-access
            {"notion": invoke_provider},
            clear=True,
        ),
    ):
        handle_task(stub=stub, task_id=22, run_id=7)

    stub.GetConnector.assert_called_once_with(
        GetConnectorRequest(connector_ref="notion")
    )
    assert captured["query"] == "release notes"
    assert captured["credentials"] == {"token": "secret"}
    assert captured["config"] == {"workspace": "primary"}
    assert _pushed_response(stub).payload == {
        "name": "notion",
        "call_id": "call-1",
        "output": {"pages": 3},
        "error": None,
    }


def test_handle_task_keeps_builtin_connectors_credential_free() -> None:
    """Built-in providers should keep their existing credential-free path."""
    stub = Mock()

    with (
        patch(
            "flwr.supercore.task_process.connector.task._pull_connector_request",
            return_value=_connector_request("web_search"),
        ),
        patch(
            "flwr.supercore.task_process.connector.task.invoke_connector",
            return_value={"results": []},
        ) as invoke_connector,
    ):
        handle_task(stub=stub, task_id=22, run_id=7)

    stub.GetConnector.assert_not_called()
    invoke_connector.assert_called_once()
    assert invoke_connector.call_args.kwargs["credentials"] is None
    assert invoke_connector.call_args.kwargs["config"] is None


def test_handle_task_does_not_expose_credentials_in_provider_errors() -> None:
    """Credential-backed provider failures should not expose secret values."""
    secret = "TOP-SECRET-TOKEN"
    stub = Mock()
    stub.GetConnector.return_value = GetConnectorResponse(
        connector_ref="notion",
        credentials_json=f'{{"token":"{secret}"}}',
        config_json="{}",
    )

    def failing_provider(**kwargs: Any) -> JSONObject:
        raise RuntimeError(f"Provider rejected {kwargs['credentials']['token']}")

    with (
        patch(
            "flwr.supercore.task_process.connector.task._pull_connector_request",
            return_value=_connector_request("notion"),
        ),
        patch.dict(
            registry._CREDENTIAL_CONNECTOR_HANDLERS,  # pylint: disable=protected-access
            {"notion": failing_provider},
            clear=True,
        ),
        pytest.raises(RuntimeError) as error,
    ):
        handle_task(stub=stub, task_id=22, run_id=7)

    response = _pushed_response(stub)
    assert str(error.value) == "Credential-backed connector execution failed."
    assert response.payload["error"] == {
        "code": "connector_error",
        "message": "Connector execution failed.",
    }
    assert secret not in str(error.value)
    assert secret not in str(response.payload)
    assert secret not in "".join(traceback.format_exception(error.value))
    assert error.value.__context__ is None
