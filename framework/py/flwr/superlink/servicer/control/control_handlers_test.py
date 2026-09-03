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
from typing import Any, cast
from unittest.mock import Mock, call, patch

from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    NOOP_ACCOUNT_NAME,
    NOOP_FLWR_AID,
    REFRESH_TOKEN_KEY,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AddAppRequest,
    AddAppResponse,
    AppInfo,
    ListAppsRequest,
    ListAppsResponse,
    ListAutomationsRequest,
    ListRunSeriesEventsRequest,
    RefreshAuthTokensRequest,
    RemoveAppRequest,
    RemoveAppResponse,
    StartAutomationRequest,
    StartRunRequest,
    StopAutomationRequest,
)
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import (
    FLOWER_AGENT_APP_ID,
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    AutomationStatus,
    TaskType,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
from flwr.superlink.auth_plugin import ControlAuthnPlugin
from flwr.superlink.extensions import RESULT_DELIVERY_CHANNEL_CHAT
from flwr.superlink.federation import NoOpFederationManager

from .control_handlers import (
    add_app,
    list_apps,
    list_automations,
    list_run_series_events,
    refresh_auth_tokens,
    remove_app,
    start_automation,
    start_run,
    stop_automation,
)


class TestControlHandlers(unittest.TestCase):  # pylint: disable=R0904
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

    def _create_dummy_run(self) -> int:
        """Create a run owned by the test account."""
        return self.state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            NOOP_FEDERATION_ID,
            None,
            self.account.flwr_aid,
            TaskType.SERVER_APP,
        )

    def _create_dummy_run_series(
        self, series_id: int, run_ids: list[int] | None = None
    ) -> None:
        """Create a run series in the in-memory state."""
        cast(Any, self.state).run_series_store[series_id] = RunSeries(
            series_id=series_id,
            federation=NOOP_FEDERATION_ID,
            description=f"series {series_id}",
            created_at="2026-05-29T00:00:00+00:00",
            updated_at="2026-05-30T00:00:00+00:00",
            run_ids=run_ids or [],
        )

    def test_list_run_series_events_returns_only_primary_task_events(self) -> None:
        """Return primary-task events from every run in the series."""
        run_ids = [self._create_dummy_run() for _ in range(2)]
        primary_task_ids = [
            cast(int, self.state.get_run_info(run_ids=[run_id])[0].primary_task_id)
            for run_id in run_ids
        ]
        child_task_id = self.state.create_task(
            task_type=TaskType.MODEL, run_id=run_ids[0]
        )
        assert child_task_id is not None
        self._create_dummy_run_series(10, run_ids)
        self.assertTrue(
            self.state.store_task_events(
                [
                    TaskEvent(
                        run_id=run_ids[0],
                        task_id=primary_task_ids[0],
                        event="response.created",
                        data='{"type":"response.created"}',
                    ),
                    TaskEvent(
                        run_id=run_ids[0],
                        task_id=child_task_id,
                        event="response.output_text.delta",
                        data='{"type":"response.output_text.delta","delta":"child"}',
                    ),
                    TaskEvent(
                        run_id=run_ids[1],
                        task_id=primary_task_ids[1],
                        event="response.completed",
                        data='{"type":"response.completed"}',
                    ),
                ]
            )
        )

        with patch(
            "flwr.superlink.servicer.control.control_handlers"
            ".extensions.notify_result_delivered"
        ) as notify_result_delivered:
            response = list_run_series_events(
                ListRunSeriesEventsRequest(series_id=10), self.account, self.state
            )

        self.assertEqual([event.task_id for event in response.events], primary_task_ids)
        runs = self.state.get_run_info(run_ids=run_ids)
        notify_result_delivered.assert_has_calls(
            [
                call(run, self.account.flwr_aid, RESULT_DELIVERY_CHANNEL_CHAT)
                for run in runs
            ],
            any_order=True,
        )
        self.assertEqual(notify_result_delivered.call_count, len(runs))

    def test_list_run_series_events_hides_unauthorized_series(self) -> None:
        """Reject event history access outside the caller's federations."""
        self._create_dummy_run_series(10)

        with (
            patch.object(
                self.state.federation_manager, "has_member", return_value=False
            ),
            self.assertRaises(FlowerError) as error,
        ):
            list_run_series_events(
                ListRunSeriesEventsRequest(series_id=10), self.account, self.state
            )

        self.assertEqual(error.exception.code, ApiErrorCode.RUN_SERIES_ID_NOT_FOUND)

    def test_refresh_auth_tokens_returns_rotated_tokens(self) -> None:
        """Return both tokens produced by the authentication plugin."""
        authn_plugin = Mock(spec=ControlAuthnPlugin)
        authn_plugin.refresh_tokens.return_value = (
            [
                (ACCESS_TOKEN_KEY, "new-access-token"),
                (REFRESH_TOKEN_KEY, "new-refresh-token"),
            ],
            self.account,
        )

        response = refresh_auth_tokens(
            RefreshAuthTokensRequest(refresh_token="old-refresh-token"),
            authn_plugin,
        )

        self.assertEqual(response.access_token, "new-access-token")
        self.assertEqual(response.refresh_token, "new-refresh-token")
        authn_plugin.refresh_tokens.assert_called_once_with(
            [(REFRESH_TOKEN_KEY, "old-refresh-token")]
        )

    def test_refresh_auth_tokens_rejects_missing_token(self) -> None:
        """Reject an empty refresh token before invoking the plugin."""
        authn_plugin = Mock(spec=ControlAuthnPlugin)

        with self.assertRaises(FlowerError) as exc_context:
            refresh_auth_tokens(RefreshAuthTokensRequest(), authn_plugin)

        self.assertEqual(
            exc_context.exception.code,
            ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
        )
        authn_plugin.refresh_tokens.assert_not_called()

    def test_refresh_auth_tokens_rejects_invalid_plugin_results(self) -> None:
        """Reject incomplete or malformed tokens and missing account information."""
        valid_tokens: list[tuple[str, str | bytes]] = [
            (ACCESS_TOKEN_KEY, "new-access-token"),
            (REFRESH_TOKEN_KEY, "new-refresh-token"),
        ]
        invalid_results: list[
            tuple[list[tuple[str, str | bytes]] | None, AccountInfo | None]
        ] = [
            (None, None),
            (valid_tokens, None),
            ([(ACCESS_TOKEN_KEY, "new-access-token")], self.account),
            (
                [
                    (ACCESS_TOKEN_KEY, "first-access-token"),
                    (ACCESS_TOKEN_KEY, "second-access-token"),
                    (REFRESH_TOKEN_KEY, "new-refresh-token"),
                ],
                self.account,
            ),
            (
                [
                    (ACCESS_TOKEN_KEY, b"new-access-token"),
                    (REFRESH_TOKEN_KEY, "new-refresh-token"),
                ],
                self.account,
            ),
        ]

        for tokens, account in invalid_results:
            with self.subTest(tokens=tokens, account=account):
                authn_plugin = Mock(spec=ControlAuthnPlugin)
                authn_plugin.refresh_tokens.return_value = (tokens, account)

                with self.assertRaises(FlowerError) as exc_context:
                    refresh_auth_tokens(
                        RefreshAuthTokensRequest(refresh_token="secret-refresh-token"),
                        authn_plugin,
                    )

                self.assertEqual(
                    exc_context.exception.code,
                    ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
                )
                self.assertNotIn("secret-refresh-token", str(exc_context.exception))

    def test_refresh_auth_tokens_requires_authentication_plugin(self) -> None:
        """Return the established error when authentication is unavailable."""
        with self.assertRaises(FlowerError) as exc_context:
            refresh_auth_tokens(RefreshAuthTokensRequest(refresh_token="token"), None)

        self.assertEqual(exc_context.exception.code, ApiErrorCode.NO_ACCOUNT_AUTH)

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

    def test_start_run_persists_agent_input_event(self) -> None:
        """Persist agent input as a primary-task message item."""
        request = StartRunRequest(federation=NOOP_FEDERATION_ID)
        request.fab.content = b"AgentApp FAB"
        request.override_config["agent.input"].string = "Hello"

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={
                    "tool": {"flwr": {"app": {"config": {"agent": {"input": ""}}}}}
                },
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers"
                ".get_metadata_from_config",
                return_value=("flwr/agent", "v0.0.1"),
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers._get_app_type",
                return_value=TaskType.AGENT_APP,
            ),
        ):
            response = start_run(request, self.account, self.state, None)

        run = self.state.get_run_info(run_ids=[response.run_id])[0]
        event = self.state.get_task_events(run_ids=[response.run_id])[0]
        self.assertEqual(
            (event.task_id, event.event, event.data),
            (
                run.primary_task_id,
                "message",
                '{"type":"message","role":"user","content":"Hello"}',
            ),
        )

    def test_start_run_notifies_extension_after_persisting_run(self) -> None:
        """Notify the optional extension with the persisted run snapshot."""
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
            patch(
                "flwr.superlink.servicer.control.control_handlers"
                ".extensions.notify_run_started"
            ) as notify_run_started,
        ):
            request = StartRunRequest(federation=NOOP_FEDERATION_ID)
            request.app_spec = "@flwr/demo==0.0.1"
            request.fab.hash_str = fab_hash
            response = start_run(
                request,
                self.account,
                self.state,
                None,
                source="web_ui",
            )

        run = self.state.get_run_info(run_ids=[response.run_id])[0]
        notify_run_started.assert_called_once()
        notified_run, source = notify_run_started.call_args.args
        self.assertEqual(notified_run.run_id, run.run_id)
        self.assertEqual(source, "web_ui")

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
            ListAppsRequest(federation_id=NOOP_FEDERATION_ID),
            self.account,
            self.state,
        )

        self.assertEqual(
            [(app.app_id, app.fab_hash, app.app_type) for app in response.apps],
            [
                ("@flwr/demo", fab_hash, TaskType.SERVER_APP),
                (FLOWER_AGENT_APP_ID, "", TaskType.AGENT_APP),
            ],
        )

    def test_list_apps_does_not_duplicate_stored_flower_agent(self) -> None:
        """List apps uses the stored Flower Agent entry when available."""
        fab_hash = self.state.store_app(
            fab=Fab("", b"fab", {}),
            federation_id=NOOP_FEDERATION_ID,
            app_id=FLOWER_AGENT_APP_ID,
            app_type=TaskType.AGENT_APP,
            added_by=self.account.flwr_aid,
        )

        response = list_apps(
            ListAppsRequest(federation_id=NOOP_FEDERATION_ID),
            self.account,
            self.state,
        )

        self.assertEqual(
            [(app.app_id, app.fab_hash, app.app_type) for app in response.apps],
            [(FLOWER_AGENT_APP_ID, fab_hash, TaskType.AGENT_APP)],
        )

    def test_list_apps_preserves_hub_flag_over_wire(self) -> None:
        """ListApps preserves Hub provenance through protobuf serialization."""
        self.state.store_app(
            fab=Fab("", b"hub fab", {}),
            federation_id=NOOP_FEDERATION_ID,
            app_id="@flwr/demo",
            app_type=TaskType.AGENT_APP,
            added_by=self.account.flwr_aid,
            is_hub_app=True,
        )

        response = list_apps(
            ListAppsRequest(federation_id=NOOP_FEDERATION_ID),
            self.account,
            self.state,
        )
        round_tripped = ListAppsResponse.FromString(response.SerializeToString())

        self.assertEqual(round_tripped.apps[0].fab_hash, "")
        self.assertTrue(round_tripped.apps[0].is_hub_app)

    def test_list_apps_preserves_unknown_hub_origin_over_wire(self) -> None:
        """ListApps leaves unknown legacy provenance absent over the wire."""
        app = AppInfo(
            app_id="@flwr/demo",
            fab_hash="legacy-hash",
            app_type=TaskType.AGENT_APP,
        )
        with patch.object(self.state, "list_apps", return_value=[app]):
            response = list_apps(
                ListAppsRequest(federation_id=NOOP_FEDERATION_ID),
                self.account,
                self.state,
            )

        round_tripped = ListAppsResponse.FromString(response.SerializeToString())

        self.assertFalse(round_tripped.apps[0].HasField("is_hub_app"))

    def test_add_and_remove_hub_app_metadata(self) -> None:
        """AddApp stores Hub metadata without retaining the downloaded FAB."""
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
            [("@flwr/demo", "", TaskType.AGENT_APP)],
        )
        self.assertTrue(apps[0].is_hub_app)
        self.assertIsNone(self.state.get_fab(fab_hash))
        self.assertIsNone(
            self.state.get_app(NOOP_FEDERATION_ID, "@flwr/demo", fab_hash)
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

    def test_start_automation_stores_hub_app_without_fab(self) -> None:
        """Store Hub automations by app ID so dispatch fetches the latest FAB."""
        self.state.store_app(
            fab=None,
            federation_id=NOOP_FEDERATION_ID,
            app_id="@flwr/agent",
            app_type=TaskType.AGENT_APP,
            added_by=self.account.flwr_aid,
            is_hub_app=True,
        )
        fab_content = b"current Hub FAB"
        request = StartAutomationRequest(
            start_run_request=StartRunRequest(
                federation=NOOP_FEDERATION_ID,
                series_id=1,
            )
        )
        request.start_run_request.fab.hash_str = hashlib.sha256(fab_content).hexdigest()
        request.start_run_request.fab.content = fab_content

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={"tool": {"flwr": {"app": {}}}},
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers"
                ".get_metadata_from_config",
                return_value=("flwr/agent", "1.0.0"),
            ),
        ):
            response = start_automation(request, self.account, self.state)

        claimed = self.state.claim_automation(
            response.automation_id,
            previous_next_run_at=response.next_run_at,
            next_run_at=None,
        )
        self.assertIsNotNone(claimed)
        stored_request, _ = cast(tuple[StartRunRequest, str], claimed)
        self.assertEqual(stored_request.app_spec, "@flwr/agent")
        self.assertFalse(stored_request.HasField("fab"))

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
