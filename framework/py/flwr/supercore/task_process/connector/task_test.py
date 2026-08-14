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
import unittest
from unittest.mock import ANY, Mock, patch

from flwr.common.serde import message_from_proto
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
)
from flwr.supercore.json_message.connector_message import (
    ConnectorRequest,
    ConnectorResponse,
)

from . import registry
from .definition import ConnectorExecutionContext
from .http import ConnectorApiError
from .task import handle_task


class NotionTestApiError(ConnectorApiError):
    """Secret-safe test error matching the Notion provider."""

    provider = "Notion"


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


class TestHandleTask(unittest.TestCase):
    """Test credential-backed connector task execution."""

    def setUp(self) -> None:
        """Set up the common connector task mocks and registry patches."""
        self.stub = Mock()
        self.stub.GetConnector.return_value = GetConnectorResponse(
            connector_ref="notion",
            credentials_json='{"token":"secret"}',
            config_json="{}",
        )
        self.provider = Mock()
        self.pull_connector_request = self.enterContext(
            patch("flwr.supercore.task_process.connector.task._pull_connector_request")
        )
        self.credential_handlers = (
            registry._CREDENTIAL_CONNECTOR_HANDLERS  # pylint: disable=protected-access
        )
        self.connector_refs = (
            registry._CREDENTIAL_CONNECTOR_REFS  # pylint: disable=protected-access
        )
        self.enterContext(patch.dict(self.credential_handlers, clear=True))
        self.enterContext(patch.dict(self.connector_refs, clear=True))

    def _configure_connector(self, name: str, connector_ref: str | None = None) -> None:
        """Configure the request and registry entry for one connector tool."""
        self.pull_connector_request.return_value = _connector_request(name)
        self.credential_handlers[name] = self.provider
        if connector_ref is not None:
            self.connector_refs[name] = connector_ref

    def test_passes_credentials_to_matching_provider(self) -> None:
        """Credential-backed providers should receive credentials and config."""
        tool_name = "notion_search"
        self._configure_connector(tool_name, connector_ref="notion")
        self.stub.GetConnector.return_value = GetConnectorResponse(
            connector_ref="notion",
            credentials_json='{"token":"secret"}',
            config_json='{"workspace":"primary"}',
        )
        self.provider.return_value = {"pages": 3}

        handle_task(stub=self.stub, task_id=22, run_id=7)

        self.stub.GetConnector.assert_called_once_with(GetConnectorRequest())
        arguments, context = self.provider.call_args.args
        assert arguments == {"query": "release notes"}
        assert context == ConnectorExecutionContext(
            credentials={"token": "secret"},
            config={"workspace": "primary"},
            usage_recorder=ANY,
        )
        assert _pushed_response(self.stub).payload == {
            "name": tool_name,
            "call_id": "call-1",
            "output": {"pages": 3},
            "error": None,
        }

    def test_rejects_credentials_for_different_connector(self) -> None:
        """Providers should receive only their connector's secrets."""
        self._configure_connector("github")

        with self.assertRaisesRegex(
            RuntimeError, "Credential-backed connector execution failed."
        ):
            handle_task(stub=self.stub, task_id=22, run_id=7)

        self.provider.assert_not_called()

    def test_does_not_expose_credentials_in_provider_errors(self) -> None:
        """Credential-backed provider failures should not expose secret values."""
        secret = "TOP-SECRET-TOKEN"
        self._configure_connector("notion")
        self.stub.GetConnector.return_value = GetConnectorResponse(
            connector_ref="notion",
            credentials_json=f'{{"token":"{secret}"}}',
            config_json="{}",
        )
        self.provider.side_effect = RuntimeError(f"Provider rejected {secret}")

        with self.assertRaises(RuntimeError) as error:
            handle_task(stub=self.stub, task_id=22, run_id=7)

        response = _pushed_response(self.stub)
        self.provider.assert_called_once()
        assert str(error.exception) == "Credential-backed connector execution failed."
        assert response.payload["error"] == {
            "code": "connector_error",
            "message": "Connector execution failed.",
        }
        assert secret not in str(error.exception)
        assert secret not in str(response.payload)
        assert secret not in "".join(traceback.format_exception(error.exception))
        assert error.exception.__context__ is None

    def test_exposes_secret_safe_provider_errors(self) -> None:
        """Credential-backed providers should return actionable safe errors."""
        self._configure_connector("notion", connector_ref="notion")
        self.provider.side_effect = NotionTestApiError("validation_error", 400)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Notion API request failed: validation_error \(400\)\.",
        ):
            handle_task(stub=self.stub, task_id=22, run_id=7)

        assert _pushed_response(self.stub).payload["error"] == {
            "code": "connector_error",
            "message": "Notion API request failed: validation_error (400).",
        }
