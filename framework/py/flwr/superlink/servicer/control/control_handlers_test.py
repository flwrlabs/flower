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
    ListAutomationsRequest,
    StartAutomationRequest,
    StartRunRequest,
    StopAutomationRequest,
)
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import (
    AUTOMATION_MAX_ACTIVE_PER_USER,
    AUTOMATION_MIN_FIXED_INTERVAL,
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    AutomationStatus,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
from flwr.superlink.federation import NoOpFederationManager

from .control_handlers import (
    list_automations,
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
        self.state.store_fab(Fab(fab_hash, fab_content, {}))

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
        ):
            request = StartRunRequest(federation=NOOP_FEDERATION_ID)
            request.fab.hash_str = fab_hash
            response = start_run(request, self.account, self.state, None)

        run = self.state.get_run_info(run_ids=[response.run_id])[0]
        self.assertEqual(run.fab_hash, fab_hash)

    def test_start_run_rejects_unknown_fab_hash(self) -> None:
        """Test StartRun rejects an unknown FAB hash."""
        request = StartRunRequest()
        request.fab.hash_str = "unknown"

        with self.assertRaises(FlowerError) as error:
            start_run(request, self.account, self.state, None)

        self.assertEqual(error.exception.code, ApiErrorCode.FAB_DOWNLOAD_FAILURE)

    def test_start_automation_normalizes_start_at_to_utc(self) -> None:
        """Normalize the automation start time to UTC."""
        # Prepare
        request = StartAutomationRequest(
            start_at="2026-07-10T04:00:00-05:00",
            fixed_interval=AUTOMATION_MIN_FIXED_INTERVAL,
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
            (
                response.series_id,
                "2026-07-10T09:00:00+00:00",
                AUTOMATION_MIN_FIXED_INTERVAL,
                3,
            ),
        )

    def test_start_automation_rejects_too_short_interval(self) -> None:
        """Reject a recurrence faster than the configured cadence."""
        request = StartAutomationRequest(
            fixed_interval=AUTOMATION_MIN_FIXED_INTERVAL - 1,
            start_run_request=StartRunRequest(series_id=1),
        )

        with self.assertRaises(FlowerError) as error:
            start_automation(request, self.account, self.state)

        self.assertEqual(error.exception.code, ApiErrorCode.INVALID_AUTOMATION_REQUEST)
        self.assertEqual(
            error.exception.public_details,
            f"`fixed_interval` must be at least "
            f"{AUTOMATION_MIN_FIXED_INTERVAL} seconds.",
        )

    def test_start_automation_rejects_user_active_automation_limit(self) -> None:
        """Reject creation after a user reaches the active automation limit."""
        for series_id in range(AUTOMATION_MAX_ACTIVE_PER_USER):
            self.state.store_automation(
                federation_id=NOOP_FEDERATION_ID,
                flwr_aid=self.account.flwr_aid,
                start_run_request=StartRunRequest(series_id=series_id),
                series_id=series_id,
                next_run_at="2026-07-10T09:00:00+00:00",
                max_runs=1,
            )

        with self.assertRaises(FlowerError) as error:
            start_automation(
                StartAutomationRequest(
                    start_run_request=StartRunRequest(series_id=100)
                ),
                self.account,
                self.state,
            )

        self.assertEqual(error.exception.code, ApiErrorCode.INVALID_AUTOMATION_REQUEST)
        self.assertEqual(
            error.exception.public_details,
            f"You can have at most {AUTOMATION_MAX_ACTIVE_PER_USER} "
            "active automations.",
        )

    def test_start_automation_limit_ignores_other_users_and_stopped(self) -> None:
        """Count only the requesting user's active automations."""
        stopped_id = None
        for series_id in range(AUTOMATION_MAX_ACTIVE_PER_USER):
            automation = self.state.store_automation(
                federation_id=NOOP_FEDERATION_ID,
                flwr_aid=self.account.flwr_aid,
                start_run_request=StartRunRequest(series_id=series_id),
                series_id=series_id,
                next_run_at="2026-07-10T09:00:00+00:00",
                max_runs=1,
            )
            stopped_id = automation.automation_id
        assert stopped_id is not None
        self.state.stop_automation(stopped_id)
        for series_id in range(AUTOMATION_MAX_ACTIVE_PER_USER):
            self.state.store_automation(
                federation_id=NOOP_FEDERATION_ID,
                flwr_aid="other-user",
                start_run_request=StartRunRequest(series_id=series_id + 100),
                series_id=series_id + 100,
                next_run_at="2026-07-10T09:00:00+00:00",
                max_runs=1,
            )

        response = start_automation(
            StartAutomationRequest(start_run_request=StartRunRequest(series_id=200)),
            self.account,
            self.state,
        )

        self.assertGreater(response.automation_id, 0)

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
