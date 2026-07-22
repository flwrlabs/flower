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
"""ServerAppIo connector credential authorization tests."""

import unittest
from unittest.mock import Mock, patch

import grpc
from parameterized import parameterized

from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    GetConnectorRequest,
    GetConnectorResponse,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME, NOOP_FEDERATION_ID, TaskType
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.superlink.federation import NoOpFederationManager

from .serverappio_servicer import ServerAppIoServicer


class TestGetConnector(unittest.TestCase):
    """Test the task-token-derived GetConnector authorization boundary."""

    def setUp(self) -> None:
        """Create an in-memory servicer without a network listener."""
        self.objectstore_factory = ObjectStoreFactory()
        self.state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME,
            NoOpFederationManager(),
            self.objectstore_factory,
        )
        self.state = self.state_factory.state()
        self.servicer = ServerAppIoServicer(
            self.state_factory,
            self.objectstore_factory,
        )

    def _create_connector_task(
        self, *, run_owner: str, connector_refs: list[str]
    ) -> object:
        run_id = self.state.create_run(
            "",
            "",
            "",
            {},
            NOOP_FEDERATION_ID,
            None,
            run_owner,
            TaskType.AGENT_APP,
            connector_refs=connector_refs,
        )
        task_id = self.state.create_task(
            TaskType.CONNECTOR,
            run_id,
            connector_ref="notion",
        )
        assert task_id is not None
        return self.state.get_tasks(task_ids=[task_id])[0]

    def test_returns_only_authenticated_task_credentials(self) -> None:
        """GetConnector should resolve the run owner's matching credentials."""
        task = self._create_connector_task(
            run_owner="account-a",
            connector_refs=["notion"],
        )
        self.assertTrue(
            self.state.upsert_connector(
                flwr_aid="account-a",
                connector_ref="notion",
                credentials_json='{"token":"secret"}',
                config_json='{"workspace":"primary"}',
            )
        )

        with patch(
            "flwr.superlink.servicer.serverappio.serverappio_servicer."
            "get_authenticated_task",
            return_value=task,
        ):
            response = self.servicer.GetConnector(
                GetConnectorRequest(connector_ref="notion"),
                Mock(),
            )

        self.assertEqual(
            response,
            GetConnectorResponse(
                connector_ref="notion",
                credentials_json='{"token":"secret"}',
                config_json='{"workspace":"primary"}',
            ),
        )

    @parameterized.expand(  # type: ignore
        [
            ("wrong_task_type", TaskType.AGENT_APP, "notion"),
            ("mismatched_ref", TaskType.CONNECTOR, "github"),
        ]
    )
    def test_rejects_wrong_task_identity(
        self,
        _name: str,
        task_type: str,
        request_ref: str,
    ) -> None:
        """GetConnector should reject non-connector tasks and ref mismatches."""
        context = Mock(spec=grpc.ServicerContext)
        context.abort.side_effect = grpc.RpcError()

        with (
            patch(
                "flwr.superlink.servicer.serverappio.serverappio_servicer."
                "get_authenticated_task",
                return_value=Mock(
                    type=task_type,
                    connector_ref="notion",
                    run_id=123,
                ),
            ),
            self.assertRaises(grpc.RpcError),
        ):
            self.servicer.GetConnector(
                GetConnectorRequest(connector_ref=request_ref),
                context,
            )

        context.abort.assert_called_once_with(
            grpc.StatusCode.PERMISSION_DENIED,
            "Connector credentials are not available to this task.",
        )

    @parameterized.expand(  # type: ignore
        [
            ("unbound", "account-a", []),
            ("other_account", "account-b", ["notion"]),
        ]
    )
    def test_hides_unavailable_credentials(
        self,
        _name: str,
        run_owner: str,
        connector_refs: list[str],
    ) -> None:
        """GetConnector should hide unbound and other-account credentials."""
        self.assertTrue(
            self.state.upsert_connector(
                flwr_aid="account-a",
                connector_ref="notion",
                credentials_json='{"token":"secret"}',
                config_json="{}",
            )
        )
        task = self._create_connector_task(
            run_owner=run_owner,
            connector_refs=connector_refs,
        )
        context = Mock(spec=grpc.ServicerContext)
        context.abort.side_effect = grpc.RpcError()

        with (
            patch(
                "flwr.superlink.servicer.serverappio.serverappio_servicer."
                "get_authenticated_task",
                return_value=task,
            ),
            self.assertRaises(grpc.RpcError),
        ):
            self.servicer.GetConnector(
                GetConnectorRequest(connector_ref="notion"),
                context,
            )

        context.abort.assert_called_once_with(
            grpc.StatusCode.NOT_FOUND,
            "Connector not found.",
        )
