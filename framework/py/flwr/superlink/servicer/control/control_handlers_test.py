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
"""Tests for Control API handler functions."""

import hashlib
import unittest
from unittest.mock import Mock, patch

from parameterized import parameterized

from flwr.agentapp.builtin import try_resolve_builtin_agent_fab
from flwr.common.constant import FAB_MAX_SIZE, NOOP_ACCOUNT_NAME, NOOP_FLWR_AID
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListAutomationsRequest,
    StartAutomationRequest,
    StartRunRequest,
    StopAutomationRequest,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    AutomationStatus,
    TaskType,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.task_process.connector import registry as connector_registry
from flwr.superlink.federation import NoOpFederationManager

from .control_handlers import (
    dispatch_automation,
    list_automations,
    start_automation,
    stop_automation,
)


class _OAuthProvider:
    """Minimal OAuth provider for connector allowlist validation."""

    connector_ref = "slack"
    display_name = "Slack"
    description = "Connect Slack."


class TestAutomationHandlers(unittest.TestCase):
    """Test automation creation and dispatch handlers."""

    def setUp(self) -> None:
        """Create an in-memory LinkState and account."""
        objectstore_factory = Mock()
        objectstore_factory.store.return_value = Mock()
        self.state: LinkState = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME,
            NoOpFederationManager(),
            objectstore_factory,
        ).state()
        self.account = AccountInfo(
            flwr_aid=NOOP_FLWR_AID,
            account_name=NOOP_ACCOUNT_NAME,
        )

    def _create_series(self) -> int:
        """Create and return a run series for an automation."""
        run_id = self.state.create_run(
            None,
            None,
            None,
            {},
            NOOP_FEDERATION_ID,
            None,
            self.account.flwr_aid,
            TaskType.SERVER_APP,
        )
        return self.state.get_run_info(run_ids=[run_id])[0].series_id

    def test_start_automation_rejects_oversized_embedded_fab(self) -> None:
        """Reject an embedded FAB before persisting the automation template."""
        request = StartAutomationRequest(
            start_run_request=StartRunRequest(
                federation=NOOP_FEDERATION_ID,
                series_id=123,
            )
        )
        request.start_run_request.fab.content = b"x" * (FAB_MAX_SIZE + 1)

        with (
            patch.object(self.state, "store_automation") as store_automation,
            self.assertRaises(FlowerError) as error,
        ):
            start_automation(request, self.account, self.state)

        self.assertEqual(error.exception.code, ApiErrorCode.INVALID_AUTOMATION_REQUEST)
        store_automation.assert_not_called()

    def test_start_automation_stores_schedule(self) -> None:
        """Store the normalized automation schedule."""
        start_at = "2026-07-10T04:00:00-05:00"
        response = start_automation(
            StartAutomationRequest(
                start_at=start_at,
                fixed_interval=86400,
                max_runs=3,
                start_run_request=StartRunRequest(
                    federation=NOOP_FEDERATION_ID,
                    series_id=123,
                ),
            ),
            self.account,
            self.state,
        )
        automation = self.state.list_automations(
            federations=[NOOP_FEDERATION_ID],
            order_by="updated_at",
        )[0]

        self.assertEqual(automation.automation_id, response.automation_id)
        self.assertEqual(
            (
                automation.series_id,
                automation.next_run_at,
                automation.fixed_interval,
                automation.remaining_runs,
            ),
            (
                response.series_id,
                "2026-07-10T09:00:00+00:00",
                86400,
                3,
            ),
        )
        self.assertEqual(response.next_run_at, automation.next_run_at)

    def test_start_automation_defaults_to_one_run(self) -> None:
        """Default an automation without recurrence settings to one run."""
        response = start_automation(
            StartAutomationRequest(
                start_run_request=StartRunRequest(
                    federation=NOOP_FEDERATION_ID,
                    series_id=123,
                ),
            ),
            self.account,
            self.state,
        )

        automation = self.state.list_automations(
            automation_ids=[response.automation_id],
            order_by="updated_at",
        )[0]
        self.assertFalse(automation.HasField("fixed_interval"))
        self.assertEqual(automation.remaining_runs, 1)

    @parameterized.expand(  # type: ignore
        [
            (
                "missing_series_id",
                StartAutomationRequest(),
                "The run `series_id` is required to start an automation.",
            ),
            (
                "invalid_start_at",
                StartAutomationRequest(
                    start_at="not-a-timestamp",
                    start_run_request=StartRunRequest(series_id=123),
                ),
                "The automation start_at value must be a valid ISO 8601 "
                "timestamp with a timezone.",
            ),
            (
                "start_at_without_timezone",
                StartAutomationRequest(
                    start_at="2026-07-10T09:00:00",
                    start_run_request=StartRunRequest(series_id=123),
                ),
                "The automation start_at value must be a valid ISO 8601 "
                "timestamp with a timezone.",
            ),
            (
                "zero_max_runs",
                StartAutomationRequest(
                    max_runs=0,
                    start_run_request=StartRunRequest(series_id=123),
                ),
                "`max_runs` must be greater than zero.",
            ),
            (
                "zero_fixed_interval",
                StartAutomationRequest(
                    fixed_interval=0,
                    start_run_request=StartRunRequest(series_id=123),
                ),
                "`fixed_interval` must be greater than zero.",
            ),
            (
                "fixed_interval_exceeds_database_range",
                StartAutomationRequest(
                    fixed_interval=2**63,
                    start_run_request=StartRunRequest(series_id=123),
                ),
                "`fixed_interval` must be less than 2**63.",
            ),
            (
                "multiple_runs_without_interval",
                StartAutomationRequest(
                    max_runs=2,
                    start_run_request=StartRunRequest(series_id=123),
                ),
                "`fixed_interval` is required for automations with multiple runs.",
            ),
        ]
    )
    def test_start_automation_rejects_invalid_request(
        self,
        _name: str,
        request: StartAutomationRequest,
        public_details: str,
    ) -> None:
        """Reject malformed automation requests."""
        with self.assertRaises(FlowerError) as error:
            start_automation(request, self.account, self.state)

        self.assertEqual(error.exception.code, ApiErrorCode.INVALID_AUTOMATION_REQUEST)
        self.assertEqual(error.exception.public_details, public_details)

    def test_list_and_stop_automations(self) -> None:
        """List and stop automations through their handlers."""
        automation = self.state.store_automation(
            federation_id=NOOP_FEDERATION_ID,
            flwr_aid=self.account.flwr_aid,
            start_run_request=StartRunRequest(
                federation=NOOP_FEDERATION_ID,
                series_id=1,
            ),
            series_id=1,
            next_run_at="2026-07-10T09:00:00+00:00",
            max_runs=1,
        )

        list_response = list_automations(
            ListAutomationsRequest(federation=NOOP_FEDERATION_ID),
            self.account,
            self.state,
        )
        stop_automation(
            StopAutomationRequest(automation_id=automation.automation_id),
            self.account,
            self.state,
        )
        stopped = self.state.list_automations(
            federations=[NOOP_FEDERATION_ID],
            statuses=[AutomationStatus.STOPPED],
            order_by="updated_at",
        )

        self.assertEqual(
            [entry.automation_id for entry in list_response.automations],
            [automation.automation_id],
        )
        self.assertEqual(
            [entry.automation_id for entry in stopped],
            [automation.automation_id],
        )

    def test_dispatch_automation_derives_agentapp_task_type(self) -> None:
        """Resolve a built-in app and derive its task type during dispatch."""
        series_id = self._create_series()
        response = start_automation(
            StartAutomationRequest(
                start_run_request=StartRunRequest(
                    app_spec="@flwragent/flwr-agent",
                    federation=NOOP_FEDERATION_ID,
                    series_id=series_id,
                )
            ),
            self.account,
            self.state,
        )

        dispatch_automation(
            self.state,
            response.automation_id,
            previous_next_run_at=response.next_run_at,
            next_run_at=None,
        )

        run_id = self.state.get_run_series(series_ids=[series_id])[0].run_ids[-1]
        run = self.state.get_run_info(run_ids=[run_id])[0]
        builtin_agent_fab = try_resolve_builtin_agent_fab("@flwragent/flwr-agent")
        assert builtin_agent_fab is not None
        self.assertEqual(
            run.fab_hash,
            hashlib.sha256(builtin_agent_fab[0]).hexdigest(),
        )
        self.assertEqual(run.primary_task_type, TaskType.AGENT_APP)
        completed = self.state.list_automations(
            automation_ids=[response.automation_id],
            statuses=[AutomationStatus.COMPLETED],
            order_by="updated_at",
        )
        self.assertEqual(len(completed), 1)

    def test_dispatch_automation_carries_connector_allowlist(self) -> None:
        """Validate and bind connector references when dispatch starts the run."""
        series_id = self._create_series()
        self.state.upsert_connector(
            flwr_aid=self.account.flwr_aid,
            connector_ref="slack",
            credentials_json="{}",
            config_json="{}",
        )
        request = StartAutomationRequest(
            start_run_request=StartRunRequest(
                federation=NOOP_FEDERATION_ID,
                series_id=series_id,
                connector_refs=[" Slack ", "slack"],
            )
        )
        request.start_run_request.fab.content = b"connector automation FAB"
        response = start_automation(request, self.account, self.state)

        with (
            patch.object(
                connector_registry,
                "OAUTH_CONNECTOR_PROVIDERS",
                (_OAuthProvider(),),
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={"tool": {"flwr": {"app": {}}}},
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers."
                "get_metadata_from_config",
                return_value=("flower/demo", "1.0.0"),
            ),
        ):
            dispatch_automation(
                self.state,
                response.automation_id,
                previous_next_run_at=response.next_run_at,
                next_run_at=None,
            )

        run_id = self.state.get_run_series(series_ids=[series_id])[0].run_ids[-1]
        self.assertEqual(
            list(
                self.state.get_run_connector_refs(
                    run_id=run_id,
                )
            ),
            ["slack"],
        )

    def test_dispatch_automation_marks_start_run_failure(self) -> None:
        """Mark the claimed automation failed if StartRun fails."""
        series_id = self._create_series()
        response = start_automation(
            StartAutomationRequest(
                start_run_request=StartRunRequest(
                    app_spec="invalid app spec",
                    federation=NOOP_FEDERATION_ID,
                    series_id=series_id,
                )
            ),
            self.account,
            self.state,
        )

        dispatch_automation(
            self.state,
            response.automation_id,
            previous_next_run_at=response.next_run_at,
            next_run_at=None,
        )

        failed = self.state.list_automations(
            automation_ids=[response.automation_id],
            statuses=[AutomationStatus.FAILED],
            order_by="updated_at",
        )
        self.assertEqual(len(failed), 1)
