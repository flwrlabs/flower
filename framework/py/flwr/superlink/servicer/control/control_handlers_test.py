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

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AddAppRequest,
    AddAppResponse,
    ListAppsRequest,
    ListAutomationsRequest,
    RemoveAppRequest,
    RemoveAppResponse,
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
from flwr.supercore.fab import Fab
from flwr.superlink.federation import NoOpFederationManager

from .control_handlers import (
    add_app,
    list_apps,
    list_automations,
    remove_app,
    start_automation,
    start_run,
    stop_automation,
)


class TestControlHandlers(unittest.TestCase):
    """Test Control API handlers."""

    def setUp(self) -> None:
        """Create an in-memory LinkState and account."""
        self.state: LinkState = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME,
            NoOpFederationManager(),
            Mock(),
        ).state()
        self.account = AccountInfo(
            flwr_aid=NOOP_FLWR_AID,
            account_name=NOOP_ACCOUNT_NAME,
        )

    def test_start_run_reuses_fab_by_hash(self) -> None:
        """Test StartRun reuses a stored FAB by hash."""
        fab_content = b"stored FAB"
        fab_hash = hashlib.sha256(fab_content).hexdigest()
        self.state.store_app(
            fab=Fab(fab_hash, fab_content, {}),
            federation_id=NOOP_FEDERATION_ID,
            app_id="@flwr/demo",
            app_type=TaskType.SERVER_APP,
            added_by=self.account.flwr_aid,
        )

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={"tool": {"flwr": {"app": {}}}},
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers"
                ".get_metadata_from_config",
                return_value=("flwr/demo", "v0.0.1"),
            ),
            patch.object(self.state, "store_app") as mock_store_app,
        ):
            request = StartRunRequest(federation=NOOP_FEDERATION_ID)
            request.app_spec = "@flwr/demo==0.0.1"
            request.fab.hash_str = fab_hash
            response = start_run(request, self.account, self.state, None)

        mock_store_app.assert_not_called()
        run = self.state.get_run_info(run_ids=[response.run_id])[0]
        self.assertEqual(run.fab_hash, fab_hash)
        apps = self.state.list_apps(NOOP_FEDERATION_ID)
        self.assertEqual(
            [(app.app_id, app.fab_hash, app.app_type) for app in apps],
            [("@flwr/demo", fab_hash, TaskType.SERVER_APP)],
        )

    def test_start_run_rejects_unknown_fab_hash(self) -> None:
        """Test StartRun rejects an unknown FAB hash without an app spec."""
        request = StartRunRequest(federation=NOOP_FEDERATION_ID)
        request.fab.hash_str = "unknown"

        with self.assertRaises(FlowerError) as error:
            start_run(request, self.account, self.state, None)

        self.assertEqual(error.exception.code, ApiErrorCode.FAB_DOWNLOAD_FAILURE)

    def test_start_run_does_not_fetch_unknown_stored_app_from_hub(self) -> None:
        """StartRun rejects an unknown stored app without fetching from Hub."""
        request = StartRunRequest(
            app_spec="@flwr/demo",
            federation=NOOP_FEDERATION_ID,
        )
        request.fab.hash_str = "stale-hash"

        with patch(
            "flwr.superlink.servicer.control.control_handlers._get_remote_fab"
        ) as mock_get_remote_fab:
            with self.assertRaises(FlowerError) as error:
                start_run(request, self.account, self.state, None)

        self.assertEqual(error.exception.code, ApiErrorCode.FAB_DOWNLOAD_FAILURE)
        mock_get_remote_fab.assert_not_called()

    def test_list_apps(self) -> None:
        """List apps associated with the requested federation."""
        fab_hash = self.state.store_app(
            fab=Fab("", b"fab", {}),
            federation_id=NOOP_FEDERATION_ID,
            app_id="@flwr/demo",
            app_type=TaskType.SERVER_APP,
            added_by=self.account.flwr_aid,
        )

        response = list_apps(
            ListAppsRequest(federation_id=NOOP_FEDERATION_ID, limit=1),
            self.account,
            self.state,
        )

        self.assertEqual(
            [(app.app_id, app.fab_hash, app.app_type) for app in response.apps],
            [("@flwr/demo", fab_hash, TaskType.SERVER_APP)],
        )

    def test_add_and_remove_app(self) -> None:
        """AddApp stores the latest Hub FAB and RemoveApp removes the app."""
        fab_content = b"hub FAB"
        verification_dict = {"publisher-key": "verified"}
        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers._get_remote_fab",
                return_value=(fab_content, verification_dict, None),
            ) as mock_get_remote_fab,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={
                    "tool": {
                        "flwr": {"app": {"components": {"agentapp": "module:app"}}}
                    }
                },
            ),
        ):
            response = add_app(
                AddAppRequest(
                    federation_id=NOOP_FEDERATION_ID,
                    app_id="@flwr/demo",
                ),
                self.account,
                self.state,
                None,
            )

        self.assertEqual(response, AddAppResponse())
        mock_get_remote_fab.assert_called_once_with(None, "@flwr/demo")
        fab_hash = hashlib.sha256(fab_content).hexdigest()
        apps = self.state.list_apps(NOOP_FEDERATION_ID)
        self.assertEqual(
            [(app.app_id, app.fab_hash, app.app_type) for app in apps],
            [("@flwr/demo", fab_hash, TaskType.AGENT_APP)],
        )
        self.assertEqual(
            self.state.get_app(NOOP_FEDERATION_ID, "@flwr/demo", fab_hash),
            Fab(fab_hash, fab_content, verification_dict),
        )

        remove_response = remove_app(
            RemoveAppRequest(
                federation_id=NOOP_FEDERATION_ID,
                app_id="@flwr/demo",
            ),
            self.account,
            self.state,
        )

        self.assertEqual(remove_response, RemoveAppResponse())
        self.assertEqual(self.state.list_apps(NOOP_FEDERATION_ID), [])

    def test_start_automation_preserves_recurrence_and_normalizes_start_at(
        self,
    ) -> None:
        """Normalize the start time and preserve a recurring interval."""
        # Prepare
        request = StartAutomationRequest(
            start_at="2026-07-10T04:00:00-05:00",
            fixed_interval=60,
            max_runs=3,
            start_run_request=StartRunRequest(
                federation=NOOP_FEDERATION_ID,
                series_id=1,
            ),
        )

        # Execute
        response = start_automation(request, self.account, self.state)

        # Assert
        automation = self.state.list_automations(
            automation_ids=[response.automation_id],
            order_by="updated_at",
        )[0]
        self.assertEqual(
            (
                automation.series_id,
                automation.next_run_at,
                automation.fixed_interval,
                automation.remaining_runs,
            ),
            (response.series_id, "2026-07-10T09:00:00+00:00", 60, 3),
        )

    def test_start_automation_omits_interval_for_one_run(self) -> None:
        """Store and list one-run automations without a recurrence interval."""
        # Prepare
        request = StartAutomationRequest(
            fixed_interval=60,
            max_runs=1,
            start_run_request=StartRunRequest(
                federation=NOOP_FEDERATION_ID,
                series_id=1,
            ),
        )

        # Execute
        response = start_automation(request, self.account, self.state)
        stored_automation = self.state.list_automations(
            automation_ids=[response.automation_id],
            order_by="updated_at",
        )[0]
        listed_automation = list_automations(
            ListAutomationsRequest(federation=NOOP_FEDERATION_ID),
            self.account,
            self.state,
        ).automations[0]

        # Assert
        self.assertFalse(stored_automation.HasField("fixed_interval"))
        self.assertFalse(listed_automation.HasField("fixed_interval"))

    def test_start_automation_rejects_start_at_without_timezone(self) -> None:
        """Reject a start time without timezone information."""
        # Prepare
        request = StartAutomationRequest(
            start_at="2026-07-10T09:00:00",
            start_run_request=StartRunRequest(series_id=1),
        )

        # Execute
        with self.assertRaises(FlowerError) as error:
            start_automation(request, self.account, self.state)

        # Assert
        self.assertEqual(error.exception.code, ApiErrorCode.INVALID_AUTOMATION_REQUEST)
        self.assertEqual(
            error.exception.public_details,
            "The automation start_at value must be a valid ISO 8601 "
            "timestamp with a timezone.",
        )

    def test_list_automations(self) -> None:
        """List automations for a federation."""
        # Prepare
        automation = self.state.store_automation(
            federation_id=NOOP_FEDERATION_ID,
            flwr_aid=self.account.flwr_aid,
            start_run_request=StartRunRequest(series_id=1),
            series_id=1,
            next_run_at="2026-07-10T09:00:00+00:00",
            max_runs=1,
        )

        # Execute
        response = list_automations(
            ListAutomationsRequest(federation=NOOP_FEDERATION_ID),
            self.account,
            self.state,
        )

        # Assert
        self.assertEqual(
            [item.automation_id for item in response.automations],
            [automation.automation_id],
        )

    def test_stop_automation(self) -> None:
        """Stop an active automation."""
        # Prepare
        automation = self.state.store_automation(
            federation_id=NOOP_FEDERATION_ID,
            flwr_aid=self.account.flwr_aid,
            start_run_request=StartRunRequest(series_id=1),
            series_id=1,
            next_run_at="2026-07-10T09:00:00+00:00",
            max_runs=1,
        )

        # Execute
        stop_automation(
            StopAutomationRequest(automation_id=automation.automation_id),
            self.account,
            self.state,
        )

        # Assert
        stopped = self.state.list_automations(
            automation_ids=[automation.automation_id],
            statuses=[AutomationStatus.STOPPED],
            order_by="updated_at",
        )
        self.assertEqual(
            [item.automation_id for item in stopped],
            [automation.automation_id],
        )
