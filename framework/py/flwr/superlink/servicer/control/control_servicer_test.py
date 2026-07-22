# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Test the Control API servicer."""

# pylint: disable=too-many-lines

import hashlib
import json
import os
import time
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import grpc
from parameterized import parameterized

from flwr.agentapp.builtin import try_resolve_builtin_agent_fab
from flwr.app import ConfigRecord, Context, RecordDict
from flwr.common.constant import NOOP_ACCOUNT_NAME, SUPERLINK_NODE_ID, Status, SubStatus
from flwr.common.serde import user_config_to_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AcceptInvitationResponse,
    AddNodeToFederationRequest,
    AddNodeToFederationResponse,
    ArchiveFederationRequest,
    ArchiveFederationResponse,
    BeginConnectorOAuthRequest,
    CompleteConnectorOAuthRequest,
    CreateFederationRequest,
    CreateInvitationRequest,
    CreateInvitationResponse,
    DisconnectConnectorRequest,
    GetRunSeriesRequest,
    ListConnectorsRequest,
    ListFederationsRequest,
    ListFederationsResponse,
    ListInvitationsRequest,
    ListInvitationsResponse,
    ListNodesRequest,
    ListNodesResponse,
    ListRunSeriesRequest,
    ListRunsRequest,
    RegisterNodeRequest,
    RejectInvitationRequest,
    RejectInvitationResponse,
    RemoveAccountFromFederationRequest,
    RemoveAccountFromFederationResponse,
    RemoveNodeFromFederationRequest,
    RemoveNodeFromFederationResponse,
    RevokeInvitationRequest,
    RevokeInvitationResponse,
    ShowFederationRequest,
    ShowFederationResponse,
    StartRunRequest,
    StopRunRequest,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
    UnregisterNodeRequest,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Account, Member  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.constant import (
    DEFAULT_FEDERATION_SIMULATION,
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    ActionType,
    RunTime,
    TaskType,
)
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, EntitlementError, FlowerError
from flwr.supercore.primitives.asymmetric import generate_key_pairs, public_key_to_bytes
from flwr.supercore.run import Run, RunStatus
from flwr.supercore.task_process.connector import registry as connector_registry
from flwr.supercore.typing import (
    AcceptInvitationContext,
    CreateFederationContext,
    CreateInvitationContext,
    JSONObject,
    RegisterSupernodeContext,
    StartRunContext,
)
from flwr.superlink.auth_plugin import NoOpControlAuthnPlugin
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.control.control_account_auth_interceptor import (
    shared_account_info,
)

from .control_handlers import (
    _derive_run_series_description,
    _format_verification,
    _validate_federation_and_node_in_request,
    _validate_federation_membership_in_request,
)
from .control_servicer import ControlServicer


class _OAuthProvider:
    """Minimal OAuth provider for Control servicer tests."""

    connector_ref = "slack"
    display_name = "Slack"
    description = "Connect Slack."

    def __init__(self, fail_exchange: bool = False) -> None:
        self.fail_exchange = fail_exchange
        self.authorization_state: str | None = None
        self.exchanged_codes: list[str] = []

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Return the test callback URI."""
        return requested_redirect_uri.rstrip("/") + "/oauth/callback"

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Capture state and return a deterministic authorization URL."""
        _ = redirect_uri, pkce_challenge
        self.authorization_state = state
        return f"https://oauth.example/authorize?state={state}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> tuple[JSONObject, JSONObject]:
        """Return test credentials or simulate an exchange failure."""
        _ = redirect_uri, pkce_verifier
        self.exchanged_codes.append(code)
        if self.fail_exchange:
            raise RuntimeError(f"Provider rejected sensitive code {code}")
        return {"access_token": "access-secret"}, {"workspace": "flower"}


class TestControlServicer(unittest.TestCase):  # pylint: disable=R0904
    """Test the Control API servicer."""

    def test_derive_run_series_description(self) -> None:
        """Test normalizing, clipping, and omitting agent input."""
        self.assertEqual(
            _derive_run_series_description(
                {"agent.input": "  Hello\n  from\tthe agent  "}
            ),
            "Hello from the agent",
        )
        self.assertEqual(
            _derive_run_series_description({"agent.input": "a" * 81}),
            f"{'a' * 79}…",
        )
        self.assertEqual(_derive_run_series_description({"agent.input": 42}), "")
        self.assertEqual(_derive_run_series_description({"agent.input": " \n\t "}), "")
        self.assertEqual(_derive_run_series_description({}), "")

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.store = Mock()
        objectstore_factory = Mock(store=Mock(return_value=self.store))
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
            ),
            objectstore_factory=objectstore_factory,
            authn_plugin=(authn_plugin := NoOpControlAuthnPlugin(Mock(), False)),
        )
        account_info = authn_plugin.validate_tokens_in_metadata([])[1]
        assert account_info is not None
        self.account_info = account_info
        self.aid: str = account_info.flwr_aid
        shared_account_info.set(account_info)
        self.state = self.servicer.linkstate_factory.state()

    def _create_dummy_run(self, flwr_aid: str | None) -> int:
        return self.state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            NOOP_FEDERATION_ID,
            None,
            flwr_aid,
            TaskType.SERVER_APP,
        )

    def _create_dummy_run_series(
        self,
        series_id: int,
        *,
        federation_id: str = NOOP_FEDERATION_ID,
        updated_at: str = "2026-05-30T00:00:00+00:00",
        run_ids: list[int] | None = None,
    ) -> None:
        cast(Any, self.state).run_series_store[series_id] = RunSeries(
            series_id=series_id,
            federation=federation_id,
            description=f"series {series_id}",
            created_at="2026-05-29T00:00:00+00:00",
            updated_at=updated_at,
            run_ids=run_ids or [],
        )

    def _begin_connector_oauth(self) -> tuple[str, str]:
        """Begin OAuth and return the persisted session ID and state."""
        response = self.servicer.BeginConnectorOAuth(
            BeginConnectorOAuthRequest(
                connector_ref=" Slack ", redirect_uri="https://client.example/"
            ),
            Mock(),
        )
        session = self.state.get_connector_oauth_session(
            oauth_session_id=response.oauth_session_id,
            flwr_aid=self.aid,
        )
        assert session is not None
        self.assertEqual(response.connector_ref, "slack")
        self.assertEqual(session.redirect_uri, "https://client.example/oauth/callback")
        self.assertTrue(session.pkce_verifier)
        return session.oauth_session_id, session.state

    def test_connector_oauth_flow(self) -> None:
        """Begin and complete a single-use OAuth flow."""
        provider = _OAuthProvider()
        with patch.object(connector_registry, "OAUTH_CONNECTOR_PROVIDERS", (provider,)):
            oauth_session_id, oauth_state = self._begin_connector_oauth()
            request = CompleteConnectorOAuthRequest(
                oauth_session_id=oauth_session_id,
                code=" authorization-code ",
                state=oauth_state,
            )

            response = self.servicer.CompleteConnectorOAuth(request, Mock())

            self.assertEqual(response.connector_ref, "slack")
            connector = self.state.get_connector(
                flwr_aid=self.aid, connector_ref="slack"
            )
            assert connector is not None
            self.assertEqual(
                json.loads(connector.credentials_json),
                {"access_token": "access-secret"},
            )
            self.assertEqual(provider.exchanged_codes, ["authorization-code"])

            with self.assertRaises(FlowerError) as exc_info:
                self.servicer.CompleteConnectorOAuth(request, Mock())
            self.assertEqual(
                exc_info.exception.code, ApiErrorCode.INVALID_CONNECTOR_REQUEST
            )

    def test_list_and_disconnect_connectors_are_account_scoped(self) -> None:
        """List and disconnect only the authenticated account's connector."""
        provider = _OAuthProvider()
        for flwr_aid in (self.aid, "other-account"):
            self.assertTrue(
                self.state.upsert_connector(
                    flwr_aid=flwr_aid,
                    connector_ref="slack",
                    credentials_json="{}",
                    config_json="{}",
                )
            )

        with patch.object(connector_registry, "OAUTH_CONNECTOR_PROVIDERS", (provider,)):
            response = self.servicer.ListConnectors(ListConnectorsRequest(), Mock())
            self.assertEqual(len(response.connectors), 1)
            self.assertTrue(response.connectors[0].connected)

            self.servicer.DisconnectConnector(
                DisconnectConnectorRequest(connector_ref=" Slack "), Mock()
            )

        self.assertIsNone(
            self.state.get_connector(flwr_aid=self.aid, connector_ref="slack")
        )
        self.assertIsNotNone(
            self.state.get_connector(flwr_aid="other-account", connector_ref="slack")
        )

    def test_connector_oauth_rejects_invalid_or_expired_session(self) -> None:
        """Reject invalid state and expired OAuth sessions before exchange."""
        provider = _OAuthProvider()
        with patch.object(connector_registry, "OAUTH_CONNECTOR_PROVIDERS", (provider,)):
            oauth_session_id, _ = self._begin_connector_oauth()
            with self.assertRaises(FlowerError) as invalid_state:
                self.servicer.CompleteConnectorOAuth(
                    CompleteConnectorOAuthRequest(
                        oauth_session_id=oauth_session_id,
                        code="authorization-code",
                        state="wrong-state",
                    ),
                    Mock(),
                )

            expired = self.state.create_connector_oauth_session(
                oauth_session_id="expired-session",
                flwr_aid=self.aid,
                connector_ref="slack",
                state="expected-state",
                redirect_uri="https://client.example/oauth/callback",
                pkce_verifier=None,
                expires_at=now() - timedelta(seconds=1),
            )
            assert expired is not None
            with self.assertRaises(FlowerError) as expired_session:
                self.servicer.CompleteConnectorOAuth(
                    CompleteConnectorOAuthRequest(
                        oauth_session_id=expired.oauth_session_id,
                        code="authorization-code",
                        state=expired.state,
                    ),
                    Mock(),
                )

        self.assertEqual(
            invalid_state.exception.code, ApiErrorCode.INVALID_CONNECTOR_REQUEST
        )
        self.assertEqual(
            expired_session.exception.code, ApiErrorCode.INVALID_CONNECTOR_REQUEST
        )
        self.assertEqual(provider.exchanged_codes, [])

    def test_connector_oauth_provider_failure_is_sanitized(self) -> None:
        """Hide authorization codes from provider failure errors."""
        provider = _OAuthProvider(fail_exchange=True)
        sensitive_code = "sensitive-authorization-code"
        with patch.object(connector_registry, "OAUTH_CONNECTOR_PROVIDERS", (provider,)):
            oauth_session_id, oauth_state = self._begin_connector_oauth()
            with self.assertRaises(FlowerError) as exc_info:
                self.servicer.CompleteConnectorOAuth(
                    CompleteConnectorOAuthRequest(
                        oauth_session_id=oauth_session_id,
                        code=sensitive_code,
                        state=oauth_state,
                    ),
                    Mock(),
                )

        self.assertEqual(exc_info.exception.code, ApiErrorCode.CONNECTOR_FAILURE)
        self.assertNotIn(sensitive_code, exc_info.exception.message)

    def test_start_run(self) -> None:
        """Test StartRun method of ControlServicer."""
        # Prepare
        fab_content = b"test FAB content 123456"
        fab_hash = hashlib.sha256(fab_content).hexdigest()
        fab_id = b"mock FAB ID"
        fab_version = b"mock FAB version"
        request = StartRunRequest()
        request.fab.hash_str = fab_hash
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID

        # Execute
        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as _,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
        ):
            mock_get_metadata_from_config.return_value = (fab_id, fab_version)
            response = self.servicer.StartRun(request, Mock())
        runs = self.state.get_run_info(run_ids=[response.run_id])
        run_info = runs[0] if runs else None

        # Assert
        assert run_info is not None
        self.assertEqual(run_info.fab_hash, fab_hash)
        self.assertEqual(run_info.fab_id, fab_id)
        self.assertEqual(run_info.fab_version, fab_version)
        self.assertEqual(run_info.primary_task_type, TaskType.SERVER_APP)
        self.assertFalse(response.HasField("note"))
        self.assertTrue(response.HasField("series_id"))
        self.assertGreater(response.series_id, 0)
        self.assertEqual(run_info.series_id, response.series_id)
        self.assertEqual(response.federation, NOOP_FEDERATION_ID)
        run_context = self.state.get_run_series_context(response.series_id)
        assert run_context is not None
        self.assertEqual(run_context.run_id, response.run_id)
        self.assertEqual(run_context.series_id, response.series_id)

    def test_start_run_validates_and_binds_oauth_connectors(self) -> None:
        """StartRun should bind canonical connected OAuth connector refs."""
        provider = _OAuthProvider()
        self.state.upsert_connector(
            flwr_aid=self.aid,
            connector_ref="slack",
            credentials_json="{}",
            config_json="{}",
        )
        request = StartRunRequest(
            federation=NOOP_FEDERATION_ID,
            connector_refs=[" Slack ", "slack"],
        )
        request.fab.content = b"test FAB content with connector refs"

        with (
            patch.object(
                connector_registry,
                "OAUTH_CONNECTOR_PROVIDERS",
                (provider,),
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={"tool": {"flwr": {"app": {}}}},
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers."
                "get_metadata_from_config",
                return_value=("flwr/demo", "1.0.0"),
            ),
        ):
            response = self.servicer.StartRun(request, Mock())

        self.assertEqual(
            list(self.state.get_run_connector_refs(run_id=response.run_id)),
            ["slack"],
        )

    @parameterized.expand(  # type: ignore
        [
            ("unknown", "unknown", ApiErrorCode.CONNECTOR_NOT_FOUND),
            ("empty", "  ", ApiErrorCode.INVALID_CONNECTOR_REQUEST),
            ("other_account", "slack", ApiErrorCode.CONNECTOR_NOT_FOUND),
        ]
    )
    def test_start_run_rejects_unavailable_oauth_connector(
        self,
        _name: str,
        connector_ref: str,
        expected_code: ApiErrorCode,
    ) -> None:
        """StartRun should reject invalid, unknown, and other-account refs."""
        provider = _OAuthProvider()
        if connector_ref == "slack":
            self.state.upsert_connector(
                flwr_aid="other-account",
                connector_ref="slack",
                credentials_json="{}",
                config_json="{}",
            )
        request = StartRunRequest(connector_refs=[connector_ref])

        with (
            patch.object(
                connector_registry,
                "OAUTH_CONNECTOR_PROVIDERS",
                (provider,),
            ),
            self.assertRaises(FlowerError) as error,
        ):
            self.servicer.StartRun(request, Mock())

        self.assertEqual(error.exception.code, expected_code)
        self.assertEqual(list(self.state.get_run_info()), [])

    def test_start_run_defaults_to_account_simulation_federation(self) -> None:
        """Test StartRun uses the account default simulation federation."""
        self.account_info.account_name = "test_account"
        expected_federation_id = f"@test_account/{DEFAULT_FEDERATION_SIMULATION}"
        federation_manager = Mock(exists=Mock(side_effect=RuntimeError))
        self.servicer.linkstate_factory.federation_manager = federation_manager
        self.servicer.linkstate_factory.state_instance = None

        with self.assertRaises(RuntimeError):
            self.servicer.StartRun(StartRunRequest(), Mock())

        federation_manager.exists.assert_called_once_with(expected_federation_id)

    def test_start_run_uses_existing_series_id(self) -> None:
        """Test StartRun links the run to an existing run series."""
        fab_content = b"test FAB content with series ID"
        initial_run_id = self._create_dummy_run(self.aid)
        series_id = self.state.get_run_info(run_ids=[initial_run_id])[0].series_id
        shared_state = RecordDict({"shared": ConfigRecord({"value": "kept"})})
        initial_context = Context(
            run_id=initial_run_id,
            node_id=SUPERLINK_NODE_ID,
            node_config={"stale": "node-config"},
            state=shared_state,
            run_config={"existing": "context"},
            series_id=series_id,
        )
        self.state.set_run_series_context(series_id, initial_context)
        request = StartRunRequest(series_id=series_id, federation=NOOP_FEDERATION_ID)
        request.fab.hash_str = hashlib.sha256(fab_content).hexdigest()
        request.fab.content = fab_content

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
        ):
            mock_get_fab_config.return_value = {"tool": {"flwr": {"app": {}}}}
            mock_get_metadata_from_config.return_value = ("flwr/demo", "v1.0.0")
            response = self.servicer.StartRun(request, Mock())

        run = self.state.get_run_info(run_ids=[response.run_id])[0]
        run_context = self.state.get_run_series_context(series_id)

        self.assertEqual(response.series_id, series_id)
        self.assertEqual(run.series_id, series_id)
        assert run_context is not None
        self.assertIsNot(run_context, initial_context)
        self.assertEqual(run_context.run_id, response.run_id)
        self.assertEqual(run_context.node_id, SUPERLINK_NODE_ID)
        self.assertEqual(run_context.node_config, {})
        self.assertIs(run_context.state, shared_state)
        self.assertEqual(run_context.run_config, {})
        self.assertEqual(run_context.series_id, series_id)
        self.assertEqual(initial_context.run_id, initial_run_id)

    @parameterized.expand(
        [
            (None, TaskType.SERVER_APP, TaskType.SERVER_APP),
            (SimulationConfig(), TaskType.SIMULATION, TaskType.SIMULATION),
        ]
    )  # type: ignore
    def test_start_run_creates_task_with_matching_type(
        self,
        sim_cfg: SimulationConfig | None,
        expected_primary_task_type: TaskType,
        expected_task_type: TaskType,
    ) -> None:
        """Test StartRun creates an initial task matching the resolved task type."""
        fab_content = b"test FAB content task type"
        request = StartRunRequest()
        request.fab.hash_str = hashlib.sha256(fab_content).hexdigest()
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
            patch.object(
                self.state.federation_manager,
                "get_simulation_config",
                return_value=sim_cfg,
            ),
        ):
            mock_get_fab_config.return_value = {
                "tool": {"flwr": {"app": {"config": {"train": {"lr": 0.1}}}}}
            }
            mock_get_metadata_from_config.return_value = ("flwr/demo", "v1.0.0")
            response = self.servicer.StartRun(request, Mock())

        runs = self.state.get_run_info(run_ids=[response.run_id])
        tasks = self.state.get_tasks()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].primary_task_type, expected_primary_task_type)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].run_id, response.run_id)
        self.assertEqual(tasks[0].type, expected_task_type)

    def test_start_run_creates_agentapp_run_from_local_fab(self) -> None:
        """Test StartRun creates an AgentApp run for a submitted AgentApp FAB."""
        fab_content = b"test AgentApp FAB content"
        request = StartRunRequest()
        request.fab.hash_str = hashlib.sha256(fab_content).hexdigest()
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID
        agent_input = "  Hello\n  from\tthe agent  "
        for key, value in user_config_to_proto({"agent.input": agent_input}).items():
            request.override_config[key].CopyFrom(value)

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
            patch.object(
                self.state.federation_manager,
                "get_simulation_config",
                return_value=SimulationConfig(),
            ),
        ):
            mock_get_fab_config.return_value = {
                "tool": {
                    "flwr": {
                        "app": {
                            "config": {"agent": {"input": "Default input"}},
                            "components": {"agentapp": "agent:app"},
                        }
                    }
                }
            }
            mock_get_metadata_from_config.return_value = ("flwr/agent", "0.1.0")
            response = self.servicer.StartRun(request, Mock())

        runs = self.state.get_run_info(run_ids=[response.run_id])
        tasks = self.state.get_tasks()
        series = self.state.get_run_series(series_ids=[response.series_id])

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].fab_id, "flwr/agent")
        self.assertEqual(runs[0].fab_version, "0.1.0")
        self.assertEqual(runs[0].primary_task_type, TaskType.AGENT_APP)
        self.assertEqual(runs[0].override_config["agent.input"], agent_input)
        self.assertEqual(series[0].description, "Hello from the agent")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].run_id, response.run_id)
        self.assertEqual(tasks[0].type, TaskType.AGENT_APP)
        self.assertEqual(tasks[0].fab_hash, runs[0].fab_hash)

    def test_start_run_creates_builtin_agentapp_run_from_app_spec(self) -> None:
        """Test StartRun creates an AgentApp run for the built-in flwr agent."""
        request = StartRunRequest(
            app_spec="@flwragent/flwr-agent",
            federation=NOOP_FEDERATION_ID,
        )
        for key, value in user_config_to_proto({"agent.input": "Hello"}).items():
            request.override_config[key].CopyFrom(value)
        builtin_agent_fab = try_resolve_builtin_agent_fab("@flwragent/flwr-agent")
        assert builtin_agent_fab is not None
        fab_file, _ = builtin_agent_fab
        expected_fab_hash = hashlib.sha256(fab_file).hexdigest()

        response = self.servicer.StartRun(request, Mock())

        runs = self.state.get_run_info(run_ids=[response.run_id])
        tasks = self.state.get_tasks()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].fab_id, "flwrlabs/flwr-agent")
        self.assertEqual(runs[0].fab_version, "0.1.0")
        self.assertEqual(runs[0].fab_hash, expected_fab_hash)
        self.assertEqual(runs[0].primary_task_type, TaskType.AGENT_APP)
        self.assertEqual(runs[0].override_config["agent.input"], "Hello")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].run_id, response.run_id)
        self.assertEqual(tasks[0].type, TaskType.AGENT_APP)
        self.assertEqual(tasks[0].fab_hash, runs[0].fab_hash)

    def test_start_run_raises_if_create_run_fails(self) -> None:
        """Test StartRun raises if the initial task cannot be created."""
        fab_content = b"test FAB content task failure"
        request = StartRunRequest()
        request.fab.hash_str = hashlib.sha256(fab_content).hexdigest()
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID
        context = Mock()

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
            patch.object(self.state, "create_run", return_value=0),
            self.assertRaises(FlowerError) as cm,
        ):
            mock_get_fab_config.return_value = {
                "tool": {"flwr": {"app": {"config": {"train": {"lr": 0.1}}}}}
            }
            mock_get_metadata_from_config.return_value = ("flwr/demo", "v1.0.0")
            self.servicer.StartRun(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.FAILED_TO_CREATE_RUN)

    def test_start_run_returns_note_for_remote_app(self) -> None:
        """Test StartRun includes the Hub compatibility note for remote apps."""
        request = StartRunRequest(
            app_spec="@anne-dev/simple-legacy-127",
            federation=NOOP_FEDERATION_ID,
        )

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers._get_remote_fab",
                return_value=(
                    b"test FAB content 123456",
                    {"valid_license": ""},
                    "Using app version 0.1.0 because the latest published version "
                    "requires a newer Flower version.",
                ),
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
        ):
            mock_get_fab_config.return_value = {"tool": {"flwr": {"app": {}}}}
            mock_get_metadata_from_config.return_value = ("flwr/demo", "0.1.0")
            response = self.servicer.StartRun(request, Mock())

        assert response.HasField("note")
        assert response.note

    def test_start_run_accepts_valid_nested_override_keys(self) -> None:
        """Test StartRun accepts valid dotted override keys from nested FAB config."""
        # Prepare
        fab_content = b"test FAB content 654321"
        fab_hash = hashlib.sha256(fab_content).hexdigest()
        request = StartRunRequest()
        request.fab.hash_str = fab_hash
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID
        for key, value in user_config_to_proto(
            {"train.lr": 0.01, "train.epochs": 3}
        ).items():
            request.override_config[key].CopyFrom(value)

        # Execute
        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
        ):
            mock_get_fab_config.return_value = {
                "tool": {
                    "flwr": {"app": {"config": {"train": {"lr": 0.1, "epochs": 1}}}}
                }
            }
            mock_get_metadata_from_config.return_value = ("flwr/demo", "v1.0.0")
            response = self.servicer.StartRun(request, Mock())
        runs = self.state.get_run_info(run_ids=[response.run_id])
        run_info = runs[0] if runs else None

        # Assert
        assert run_info is not None
        self.assertEqual(run_info.override_config["train.lr"], 0.01)
        self.assertEqual(run_info.override_config["train.epochs"], 3)

    def test_start_run_rejects_unknown_override_keys(self) -> None:
        """Test StartRun rejects override keys not present in FAB config."""
        # Prepare
        fab_content = b"test FAB content 123456"
        fab_hash = hashlib.sha256(fab_content).hexdigest()
        request = StartRunRequest()
        request.fab.hash_str = fab_hash
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID
        for key, value in user_config_to_proto({"unknown.key": 10}).items():
            request.override_config[key].CopyFrom(value)
        context = Mock()

        # Execute/Assert
        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
            self.assertRaises(FlowerError) as cm,
        ):
            mock_get_fab_config.return_value = {
                "tool": {"flwr": {"app": {"config": {"train": {"lr": 0.1}}}}}
            }
            mock_get_metadata_from_config.return_value = ("flwr/demo", "v1.0.0")
            self.servicer.StartRun(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.INVALID_RUN_CONFIG)

    def test_start_run_denied_when_not_entitled(self) -> None:
        """Test StartRun raises when federation manager denies execution."""
        request = StartRunRequest()
        request.fab.hash_str = hashlib.sha256(b"test FAB content").hexdigest()
        request.fab.content = b"test FAB content"
        request.federation = NOOP_FEDERATION_ID

        context = Mock()

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch.object(
                self.state.federation_manager,
                "can_execute",
                side_effect=EntitlementError(
                    "Start run denied for this account.",
                    public_details="Start run not permitted.",
                    entitlement_code=101,
                ),
            ),
            self.assertRaises(EntitlementError) as cm,
        ):
            mock_get_fab_config.return_value = {
                "tool": {"flwr": {"app": {"config": {"train": {"lr": 0.1}}}}}
            }
            self.servicer.StartRun(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.ENTITLEMENT_ERROR)
        self.assertEqual(cm.exception.public_details, "Start run not permitted.")
        self.assertEqual(cm.exception.entitlement_code, 101)

    @parameterized.expand(
        [
            (RunTime.DEPLOYMENT, False),
            (RunTime.SIMULATION, True),
        ]
    )  # type: ignore
    def test_start_run_calls_can_execute_with_expected_args(
        self, expected_runtime: RunTime, simulation: bool
    ) -> None:
        """Test StartRun calls can_execute with correct runtime in StartRunContext."""
        fab_content = b"test FAB content 777"
        request = StartRunRequest()
        request.fab.hash_str = hashlib.sha256(fab_content).hexdigest()
        request.fab.content = fab_content
        request.federation = NOOP_FEDERATION_ID

        sim_cfg = SimulationConfig() if simulation else None

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config"
            ) as mock_get_fab_config,
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_metadata_from_config"
            ) as mock_get_metadata_from_config,
            patch.object(
                self.state.federation_manager,
                "can_execute",
                return_value=None,
            ) as mock_can_execute,
            patch.object(
                self.state.federation_manager,
                "get_simulation_config",
                return_value=sim_cfg,
            ),
        ):
            mock_get_fab_config.return_value = {
                "tool": {"flwr": {"app": {"config": {"train": {"lr": 0.1}}}}}
            }
            mock_get_metadata_from_config.return_value = ("flwr/demo", "v1.0.0")
            _ = self.servicer.StartRun(request, Mock())

        mock_can_execute.assert_called_once_with(
            self.aid,
            ActionType.START_RUN,
            StartRunContext(federation_id=NOOP_FEDERATION_ID, runtime=expected_runtime),
        )

    @parameterized.expand([(None,), (1,), (2,), (3,), (9,)])  # type: ignore
    def test_list_runs(self, limit: int | None) -> None:
        """Test List method of ControlServicer with --runs option."""
        # Prepare
        run_ids: list[int] = []
        for _ in range(3):
            run_ids.append(self._create_dummy_run(self.aid))
            time.sleep(1e-6)  # Ensure different timestamps for sorting

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(limit=limit), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        if limit is None:
            limit = 999
        self.assertAlmostEqual(retrieved_timestamp, now().timestamp(), delta=1e-1)
        self.assertEqual(set(response.run_dict.keys()), set(run_ids[-limit:]))
        self.assertTrue(
            all(
                run.account_name == self.account_info.account_name
                for run in response.run_dict.values()
            )
        )

    def test_list_run_id(self) -> None:
        """Test List method of ControlServicer with --run-id option."""
        # Prepare
        for _ in range(3):
            run_id = self._create_dummy_run(self.aid)

        # Execute
        response = self.servicer.ListRuns(ListRunsRequest(run_id=run_id), Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertAlmostEqual(retrieved_timestamp, now().timestamp(), delta=1e-1)
        self.assertEqual(set(response.run_dict.keys()), {run_id})
        self.assertEqual(
            response.run_dict[run_id].account_name, self.account_info.account_name
        )

    def test_list_run_series_filters_by_federation(self) -> None:
        """Test ListRunSeries filters by an explicit federation."""
        # Prepare
        self._create_dummy_run_series(1, federation_id=NOOP_FEDERATION_ID)
        self._create_dummy_run_series(2, federation_id="@me/other")

        # Execute
        response = self.servicer.ListRunSeries(
            ListRunSeriesRequest(federation_id=NOOP_FEDERATION_ID), Mock()
        )

        # Assert
        self.assertEqual([entry.series_id for entry in response.entries], [1])

    def test_get_run_series_returns_context(self) -> None:
        """Test GetRunSeries returns series metadata and shared Context."""
        # Prepare
        series_id = 10
        run_id = self._create_dummy_run(self.aid)
        self._create_dummy_run_series(series_id, run_ids=[run_id])
        shared_context = Context(
            run_id=0,
            node_id=SUPERLINK_NODE_ID,
            node_config={},
            state=RecordDict(),
            run_config={},
            series_id=series_id,
        )
        self.state.set_run_series_context(series_id, shared_context)

        # Execute
        response = self.servicer.GetRunSeries(
            GetRunSeriesRequest(series_id=series_id), Mock()
        )

        # Assert
        self.assertEqual(response.series.series_id, series_id)
        self.assertEqual(response.series.last_run_status.status, Status.PENDING)
        self.assertTrue(response.HasField("context"))
        self.assertEqual(response.context.series_id, series_id)

    def test_get_run_series_raises_for_unknown_series(self) -> None:
        """Test GetRunSeries raises for unknown RunSeries IDs."""
        # Prepare
        context = Mock()

        # Execute/Assert
        with self.assertRaises(FlowerError) as cm:
            self.servicer.GetRunSeries(GetRunSeriesRequest(series_id=999), context)

        self.assertEqual(cm.exception.code, ApiErrorCode.RUN_SERIES_ID_NOT_FOUND)

    def test_stop_run(self) -> None:
        """Test StopRun method of ControlServicer."""
        # Prepare
        run_id = self._create_dummy_run(self.aid)
        expected_run_status = RunStatus(Status.FINISHED, SubStatus.STOPPED, "")

        # Execute
        response = self.servicer.StopRun(StopRunRequest(run_id=run_id), Mock())
        runs = self.state.get_run_info(run_ids=[run_id])
        run_state = runs[0] if runs else None

        # Assert
        self.assertTrue(response.success)
        self.assertIsNotNone(run_state)
        if run_state is not None:
            self.assertEqual(run_state.status, expected_run_status)
        self.store.delete_objects_in_run.assert_called_once_with(run_id)

    @parameterized.expand(
        [
            (
                None,
                False,
                public_key_to_bytes(generate_key_pairs()[1]),
            ),  # PASSES, true EC keys used once
            (
                ApiErrorCode.PUBLIC_KEY_ALREADY_IN_USE,
                True,
                public_key_to_bytes(generate_key_pairs()[1]),
            ),  # FAILS, true EC keys but already in use
            (
                ApiErrorCode.PUBLIC_KEY_NOT_VALID,
                False,
                os.urandom(32),
            ),  # FAILS, fake EC keys
        ]
    )  # type: ignore
    def test_create_node_cli(
        self, expected_code: ApiErrorCode | None, pre_register_key: bool, pub_key: bytes
    ) -> None:
        """Test CreateNodeCli method of ControlServicer."""
        # Prepare
        if pre_register_key:
            self.state.create_node(
                owner_aid="fake_aid",
                owner_name="fake_name",
                public_key=pub_key,
                heartbeat_interval=10,
            )

        # Execute
        req = RegisterNodeRequest(public_key=pub_key)
        ctx = Mock()
        if expected_code is not None:
            with self.assertRaises(FlowerError) as cm:
                self.servicer.RegisterNode(req, ctx)
            self.assertEqual(cm.exception.code, expected_code)
        else:
            response = self.servicer.RegisterNode(req, ctx)
            assert response.node_id

    def test_register_node_denied_when_not_entitled(self) -> None:
        """Test RegisterNode raises when federation manager denies execution."""
        req = RegisterNodeRequest(
            public_key=public_key_to_bytes(generate_key_pairs()[1])
        )
        ctx = Mock()

        with (
            patch.object(self.state, "create_node") as mock_create_node,
            patch.object(
                self.state.federation_manager,
                "can_execute",
                side_effect=EntitlementError(
                    "Register node denied for this account.",
                    public_details="Register node not permitted.",
                    entitlement_code=102,
                ),
            ),
            self.assertRaises(EntitlementError) as cm,
        ):
            self.servicer.RegisterNode(req, ctx)

        self.assertEqual(cm.exception.code, ApiErrorCode.ENTITLEMENT_ERROR)
        self.assertEqual(cm.exception.public_details, "Register node not permitted.")
        self.assertEqual(cm.exception.entitlement_code, 102)
        mock_create_node.assert_not_called()

    def test_register_node_calls_can_execute_with_expected_args(self) -> None:
        """Test RegisterNode calls can_execute with register action."""
        req = RegisterNodeRequest(
            public_key=public_key_to_bytes(generate_key_pairs()[1])
        )

        with patch.object(
            self.state.federation_manager,
            "can_execute",
            return_value=None,
        ) as mock_can_execute:
            _ = self.servicer.RegisterNode(req, Mock())

        mock_can_execute.assert_called_once_with(
            self.aid,
            ActionType.REGISTER_SUPERNODE,
            RegisterSupernodeContext(),
        )

    @parameterized.expand(
        [
            (True,),  # PASSES, uses registered node ID
            (False),  # FAILS, uses unregistered node ID
        ]
    )  # type: ignore
    def test_delete_node_cli(self, real_node_id: bool) -> None:
        """Test DeleteNodeCli method of ControlServicer."""
        # Prepare
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        node_id = self.state.create_node(
            owner_aid=self.aid,
            owner_name="fake_name",
            public_key=pub_key,
            heartbeat_interval=10,
        )

        # Execute
        req = UnregisterNodeRequest(node_id=node_id if real_node_id else node_id + 1)
        ctx = Mock()
        if not real_node_id:
            with self.assertRaises(FlowerError) as cm:
                self.servicer.UnregisterNode(req, ctx)
            self.assertEqual(cm.exception.code, ApiErrorCode.NODE_NOT_FOUND)
        else:
            self.servicer.UnregisterNode(req, ctx)

    def test_create_delete_create_node_cli(self) -> None:
        """Test CreateNodeCli and DeleteNodeCli method of ControlServicer."""
        # Prepare
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        node_id = self.state.create_node(
            owner_aid=self.aid,
            owner_name="fake_name",
            public_key=pub_key,
            heartbeat_interval=10,
        )

        # Execute
        # Unregister node
        self.servicer.UnregisterNode(UnregisterNodeRequest(node_id=node_id), Mock())

        # Try to add node with same public key again
        with self.assertRaises(FlowerError) as cm:
            self.servicer.RegisterNode(RegisterNodeRequest(public_key=pub_key), Mock())
        self.assertEqual(cm.exception.code, ApiErrorCode.PUBLIC_KEY_ALREADY_IN_USE)

    @parameterized.expand(
        [
            ("fake_aid", True),  # One NodeId is retrieved
            ("another_fake_aid", False),  # Zero NodeId are retrieved
        ]
    )  # type: ignore
    def test_list_nodes_cli(self, flwr_aid_retrieving: str, expected: bool) -> None:
        """Test ListNodesCli method of ControlServicer."""
        # Prepare
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        node_id = self.state.create_node(
            owner_aid="fake_aid",
            owner_name="fake_name",
            public_key=pub_key,
            heartbeat_interval=10,
        )

        # Execute
        with patch(
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(flwr_aid=flwr_aid_retrieving),
        ):
            res: ListNodesResponse = self.servicer.ListNodes(ListNodesRequest(), Mock())

        # Assert
        if expected:
            self.assertEqual(len(res.nodes_info), 1)
            self.assertEqual(res.nodes_info[0].node_id, node_id)
            self.assertEqual(res.nodes_info[0].owner_aid, "fake_aid")
            self.assertEqual(res.nodes_info[0].public_key, pub_key)
        else:
            self.assertEqual(len(res.nodes_info), 0)

    def test_show_federation(self) -> None:
        """Test ShowFederation method of ControlServicer."""
        # Prepare
        request = ShowFederationRequest(federation_name=NOOP_FEDERATION_ID)

        # Execute
        response: ShowFederationResponse = self.servicer.ShowFederation(request, Mock())
        retrieved_timestamp = datetime.fromisoformat(response.now).timestamp()

        # Assert
        self.assertAlmostEqual(retrieved_timestamp, now().timestamp(), delta=1e-1)
        self.assertEqual(response.federation.name, NOOP_FEDERATION_ID)
        self.assertFalse(response.federation.simulation)

    def test_list_federations_includes_simulation_flag(self) -> None:
        """Test ListFederations surfaces the federation simulation flag."""
        objectstore_factory = Mock(store=Mock(return_value=self.store))
        servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME,
                NoOpFederationManager(simulation=True),
                objectstore_factory,
            ),
            objectstore_factory=objectstore_factory,
            authn_plugin=NoOpControlAuthnPlugin(Mock(), False),
        )

        response: ListFederationsResponse = servicer.ListFederations(
            ListFederationsRequest(), Mock()
        )

        self.assertEqual(len(response.federations), 1)
        self.assertTrue(response.federations[0].simulation)

    def test_create_federation_success(self) -> None:
        """Test CreateFederation succeeds when federation_manager.create_federation
        works."""
        # Prepare
        name = "test-federation"
        description = "A test federation"
        expected_fed_id = f"@{NOOP_ACCOUNT_NAME}/{name}"
        request = CreateFederationRequest(
            federation_name=name,
            description=description,
            simulation=True,
        )
        mock_members = [
            Member(account=Account(id=self.aid), role="owner"),
        ]
        mock_federation = SimpleNamespace(
            id=expected_fed_id,
            description=description,
            members=mock_members,
            simulation=True,
        )
        manager_calls = Mock()

        # Execute
        with (
            patch.object(
                self.state.federation_manager,
                "can_execute",
                return_value=None,
            ) as mock_can_execute,
            patch.object(
                self.state.federation_manager,
                "ensure_default_federations_exist",
                return_value=None,
            ) as mock_ensure_default_federations_exist,
            patch.object(
                self.state.federation_manager,
                "create_federation",
                return_value=mock_federation,
            ) as mock_create,
        ):
            manager_calls.attach_mock(mock_can_execute, "can_execute")
            manager_calls.attach_mock(mock_create, "create_federation")
            response = self.servicer.CreateFederation(request, Mock())

        # Assert
        mock_can_execute.assert_called_once_with(
            self.aid,
            ActionType.CREATE_FEDERATION,
            CreateFederationContext(
                federation_id=expected_fed_id,
                runtime=RunTime.SIMULATION,
                visibility="private",
            ),
        )
        mock_ensure_default_federations_exist.assert_called_once_with(
            flwr_aid=self.aid,
        )
        mock_create.assert_called_once_with(
            federation_id=expected_fed_id,
            description=description,
            flwr_aid=self.aid,
            simulation=True,
        )
        self.assertEqual(response.federation.name, expected_fed_id)
        self.assertEqual(response.federation.description, description)
        self.assertEqual(len(response.federation.members), 1)
        self.assertEqual(response.federation.members[0].account.id, self.aid)
        self.assertEqual(response.federation.members[0].role, "owner")
        self.assertTrue(response.federation.simulation)

    def test_create_federation_fails_on_manager_error(self) -> None:
        """Test CreateFederation raises when federation_manager.create_federation
        raises."""
        # Prepare
        name = "test-federation"
        description = "A test federation"
        request = CreateFederationRequest(
            federation_name=name,
            description=description,
        )
        mock_context = Mock()

        # Execute & Assert
        with self.assertRaises(FlowerError):
            self.servicer.CreateFederation(request, mock_context)

    def test_create_federation_denied_when_not_entitled(self) -> None:
        """Test CreateFederation raises when federation manager denies execution."""
        request = CreateFederationRequest(
            federation_name="test-federation",
            description="A test federation",
            simulation=False,
        )
        context = Mock()

        with (
            patch.object(
                self.state.federation_manager,
                "can_execute",
                side_effect=EntitlementError(
                    "Create federation denied for this account.",
                    public_details="Create federation not permitted.",
                    entitlement_code=103,
                ),
            ),
            self.assertRaises(EntitlementError) as cm,
        ):
            self.servicer.CreateFederation(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.ENTITLEMENT_ERROR)
        self.assertEqual(
            cm.exception.public_details, "Create federation not permitted."
        )
        self.assertEqual(cm.exception.entitlement_code, 103)

    def test_create_federation_raises_on_invalid_name(self) -> None:
        """Test CreateFederation raises when federation name is invalid."""
        request = CreateFederationRequest(
            federation_name="Invalid Federation Name!",
            description="A test federation with invalid name",
            simulation=False,
        )
        context = Mock()

        with self.assertRaises(FlowerError) as cm:
            self.servicer.CreateFederation(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.INVALID_FEDERATION_NAME)

    def test_archive_federation_success(self) -> None:
        """Test ArchiveFederation succeeds when federation_manager.archive_federation
        works."""
        # Prepare
        request = ArchiveFederationRequest(federation_name="@me/fed")

        # Execute
        with patch.object(
            self.state.federation_manager,
            "archive_federation",
            return_value=None,
        ) as mock_archive:
            response = self.servicer.ArchiveFederation(request, Mock())

        # Assert
        mock_archive.assert_called_once_with(
            flwr_aid=self.aid,
            federation_id="@me/fed",
        )
        self.assertIsNotNone(response)

    def test_archive_federation_fails_on_manager_error(self) -> None:
        """Test ArchiveFederation raises when federation_manager.archive_federation
        raises."""
        # Prepare
        federation_id = "@me/fed"
        request = ArchiveFederationRequest(federation_name=federation_id)
        mock_context = Mock()

        # Execute & Assert
        with self.assertRaises(FlowerError):
            self.servicer.ArchiveFederation(request, mock_context)

    def test_archive_federation_stops_active_runs(self) -> None:
        """Test ArchiveFederation stops unfinished runs in the federation."""
        request = ArchiveFederationRequest(federation_name="@me/fed")
        # Create an unfinished run in the federation and give it a live token,
        # matching the state that StopRun would normally have to clean up.
        run_id = self.state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            "@me/fed",
            None,
            self.aid,
            TaskType.SERVER_APP,
        )

        with patch.object(
            self.state.federation_manager,
            "archive_federation",
            return_value=None,
        ):
            response = self.servicer.ArchiveFederation(request, Mock())

        # Archiving should reuse the same stop-run cleanup path as StopRun.
        run = self.state.get_run_info(run_ids=[run_id])[0]
        self.assertEqual(run.status, RunStatus(Status.FINISHED, SubStatus.STOPPED, ""))
        self.store.delete_objects_in_run.assert_called_once_with(run_id)
        self.assertIsInstance(response, ArchiveFederationResponse)

    def test_remove_account_from_federation_success(self) -> None:
        """Test RemoveAccountFromFederation succeeds when manager call works."""
        request = RemoveAccountFromFederationRequest(
            federation_name="@me/fed",
            account_name="target-account",
        )
        target_flwr_aid = "target-aid"

        with patch.object(
            self.state.federation_manager,
            "remove_account",
            return_value=target_flwr_aid,
        ) as mock_remove_account:
            response = self.servicer.RemoveAccountFromFederation(request, Mock())

        mock_remove_account.assert_called_once_with(
            flwr_aid=self.aid,
            federation_id="@me/fed",
            target_account_name="target-account",
        )
        self.assertIsInstance(response, RemoveAccountFromFederationResponse)

    def test_remove_account_from_federation_stops_removed_account_runs(self) -> None:
        """Test removing an account stops that account's unfinished federation runs."""
        request = RemoveAccountFromFederationRequest(
            federation_name="@me/fed",
            account_name="target-account",
        )
        target_flwr_aid = "target-aid"
        run_id = self.state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            "@me/fed",
            None,
            target_flwr_aid,
            TaskType.SERVER_APP,
        )

        with patch.object(
            self.state.federation_manager,
            "remove_account",
            return_value=target_flwr_aid,
        ):
            response = self.servicer.RemoveAccountFromFederation(request, Mock())

        run = self.state.get_run_info(run_ids=[run_id])[0]
        self.assertEqual(run.status, RunStatus(Status.FINISHED, SubStatus.STOPPED, ""))
        self.store.delete_objects_in_run.assert_called_once_with(run_id)
        self.assertIsInstance(response, RemoveAccountFromFederationResponse)


class TestControlServicerInvitationRPCs(unittest.TestCase):
    """Unit tests for invitation RPC success paths."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.flwr_aid = "test-flwr-aid"
        self.account_name = "test-account"
        self.state = Mock()
        self.state.federation_manager = Mock()
        self.linkstate_factory = Mock()
        self.linkstate_factory.state.return_value = self.state
        self.servicer = ControlServicer(
            linkstate_factory=self.linkstate_factory,
            objectstore_factory=Mock(),
            authn_plugin=Mock(),
        )
        self.get_current_account_info_patcher = patch(
            "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
            return_value=SimpleNamespace(
                flwr_aid=self.flwr_aid, account_name=self.account_name
            ),
        )
        self.get_current_account_info_patcher.start()
        self.addCleanup(self.get_current_account_info_patcher.stop)

    def test_create_invitation_success(self) -> None:
        """Test CreateInvitation success path."""
        request = CreateInvitationRequest(
            invitee_account_name="invitee-aid",
            federation_name="@me/fed",
        )
        context = Mock()
        self.state.federation_manager.can_execute.return_value = None
        self.state.federation_manager.get_simulation_config.return_value = None

        response = self.servicer.CreateInvitation(request, context)

        self.state.federation_manager.ensure_default_federations_exist.assert_called_once_with(
            flwr_aid=self.flwr_aid
        )
        self.state.federation_manager.can_execute.assert_called_once_with(
            flwr_aid=self.flwr_aid,
            action=ActionType.CREATE_INVITATION,
            context=CreateInvitationContext(
                federation_id="@me/fed",
                invitee_account_name="invitee-aid",
                runtime=RunTime.DEPLOYMENT,
            ),
        )
        self.state.federation_manager.create_invitation.assert_called_once_with(
            flwr_aid=self.flwr_aid,
            federation_id="@me/fed",
            invitee_account_name="invitee-aid",
        )
        self.assertIsInstance(response, CreateInvitationResponse)

    def test_create_invitation_denied_when_not_permitted(self) -> None:
        """Test CreateInvitation raises when can_execute returns False."""
        request = CreateInvitationRequest(
            invitee_account_name="invitee-aid",
            federation_name="@me/fed",
        )
        context = Mock()
        self.state.federation_manager.can_execute.side_effect = EntitlementError(
            "Create invitation denied for this account.",
            public_details="Create invitation not permitted.",
            entitlement_code=104,
        )

        with self.assertRaises(EntitlementError) as cm:
            self.servicer.CreateInvitation(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.ENTITLEMENT_ERROR)
        self.assertEqual(
            cm.exception.public_details, "Create invitation not permitted."
        )
        self.assertEqual(cm.exception.entitlement_code, 104)
        self.state.federation_manager.create_invitation.assert_not_called()

    def test_list_invitations_success(self) -> None:
        """Test ListInvitations success path."""
        request = ListInvitationsRequest()
        context = Mock()
        self.state.federation_manager.list_invitations.return_value = ([], [])

        response = self.servicer.ListInvitations(request, context)

        self.state.federation_manager.list_invitations.assert_called_once_with(
            self.flwr_aid
        )
        self.assertIsInstance(response, ListInvitationsResponse)
        self.assertEqual(len(response.created_invitations), 0)
        self.assertEqual(len(response.received_invitations), 0)

    def test_accept_invitation_success(self) -> None:
        """Test AcceptInvitation success path."""
        request = AcceptInvitationRequest(federation_name="@me/fed")
        context = Mock()
        self.state.federation_manager.can_execute.return_value = None
        self.state.federation_manager.get_simulation_config.return_value = None

        response = self.servicer.AcceptInvitation(request, context)

        self.state.federation_manager.can_execute.assert_called_once_with(
            flwr_aid=self.flwr_aid,
            action=ActionType.ACCEPT_INVITATION,
            context=AcceptInvitationContext(
                federation_id="@me/fed",
                runtime=RunTime.DEPLOYMENT,
            ),
        )
        self.state.federation_manager.accept_invitation.assert_called_once_with(
            flwr_aid=self.flwr_aid,
            federation_id="@me/fed",
        )
        self.assertIsInstance(response, AcceptInvitationResponse)

    def test_accept_invitation_denied_when_not_permitted(self) -> None:
        """Test AcceptInvitation raises when can_execute returns False."""
        request = AcceptInvitationRequest(federation_name="@me/fed")
        context = Mock()
        self.state.federation_manager.can_execute.side_effect = EntitlementError(
            "Accept invitation denied for this account.",
            public_details="Accept invitation not permitted.",
            entitlement_code=105,
        )

        with self.assertRaises(EntitlementError) as cm:
            self.servicer.AcceptInvitation(request, context)

        self.assertEqual(cm.exception.code, ApiErrorCode.ENTITLEMENT_ERROR)
        self.assertEqual(
            cm.exception.public_details, "Accept invitation not permitted."
        )
        self.assertEqual(cm.exception.entitlement_code, 105)
        self.state.federation_manager.accept_invitation.assert_not_called()

    def test_reject_invitation_success(self) -> None:
        """Test RejectInvitation success path."""
        request = RejectInvitationRequest(federation_name="@me/fed")
        context = Mock()

        response = self.servicer.RejectInvitation(request, context)

        self.state.federation_manager.reject_invitation.assert_called_once_with(
            flwr_aid=self.flwr_aid,
            federation_id="@me/fed",
        )
        self.assertIsInstance(response, RejectInvitationResponse)

    def test_revoke_invitation_success(self) -> None:
        """Test RevokeInvitation success path."""
        request = RevokeInvitationRequest(
            invitee_account_name="invitee-aid",
            federation_name="@me/fed",
        )
        context = Mock()

        response = self.servicer.RevokeInvitation(request, context)

        self.state.federation_manager.revoke_invitation.assert_called_once_with(
            flwr_aid=self.flwr_aid,
            federation_id="@me/fed",
            invitee_account_name="invitee-aid",
        )
        self.assertIsInstance(response, RevokeInvitationResponse)


class TestControlServicerAuth(unittest.TestCase):
    """Test ControlServicer methods with authentication plugin and flwr_aid checking."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), Mock()
            ),
            objectstore_factory=Mock(),
            authn_plugin=Mock(),
        )
        self.state = self.servicer.linkstate_factory.state()

    def _create_dummy_run(self, flwr_aid: str | None) -> int:
        return self.state.create_run(
            "flwr/demo",
            "v0.0.1",
            "hash123",
            {},
            NOOP_FEDERATION_ID,
            None,
            flwr_aid,
            TaskType.SERVER_APP,
        )

    def make_context(self) -> MagicMock:
        """Create a mock context."""
        ctx = MagicMock(spec=grpc.ServicerContext)
        ctx.is_active.return_value = False
        return ctx

    def test_streamlogs_auth_unsuccessful_when_not_federation_member(self) -> None:
        """Test StreamLogs raises when requester is not a federation member."""
        run_id = self._create_dummy_run("run-owner")
        request = StreamLogsRequest(run_id=run_id, after_timestamp=0)
        ctx = self.make_context()

        with (
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(flwr_aid="user-123"),
            ),
            patch.object(
                self.state.federation_manager, "has_member", return_value=False
            ),
        ):
            gen = self.servicer.StreamLogs(request, ctx)
            with self.assertRaises(FlowerError) as cm:
                next(gen)
            self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_FOUND)

    def test_streamlogs_auth_successful(self) -> None:
        """Test StreamLogs succeeds for a federation member."""
        run_id = 789
        request = StreamLogsRequest(run_id=run_id, after_timestamp=0)
        ctx = self.make_context()
        ctx.is_active.return_value = True
        mock_get_run_info = Mock()
        mock_run = Mock(
            federation_id=NOOP_FEDERATION_ID,
            primary_task_id=456,
            status=RunStatus(Status.FINISHED, SubStatus.COMPLETED, ""),
        )
        mock_get_run_info.return_value = [mock_run]
        mock_get_task_log = Mock(return_value=("log1", 1.0))

        # Execute & Assert
        with (
            patch.object(self.state, "get_task_log", new=mock_get_task_log),
            patch.object(self.state, "get_run_info", new=mock_get_run_info),
            patch.object(
                self.state.federation_manager, "has_member", return_value=True
            ),
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(flwr_aid="user-123"),
            ),
        ):
            gen = self.servicer.StreamLogs(request, ctx)
            msgs = list(gen)
            mock_get_run_info.assert_called_with(run_ids=[run_id])
            mock_get_task_log.assert_called_once_with(456, 1e-06)
            self.assertEqual(len(msgs), 1)
            self.assertIsInstance(msgs[0], StreamLogsResponse)
            self.assertEqual(msgs[0].log_output, "log1")
            self.assertEqual(msgs[0].latest_timestamp, 1.0)

    def test_streamrunevents_yields_events(self) -> None:
        """Test StreamRunEvents streams task events for an accessible run."""
        # Prepare
        run_id = 789
        request = StreamRunEventsRequest(run_id=run_id, after_task_event_id=4)
        ctx = self.make_context()
        ctx.is_active.return_value = True
        mock_run = Mock(
            federation_id=NOOP_FEDERATION_ID,
            status=RunStatus(Status.FINISHED, SubStatus.COMPLETED, ""),
        )
        event_1 = TaskEvent(
            id=5,
            run_id=run_id,
            task_id=123,
            event="response.output_text.delta",
            data='{"delta":"Hel"}',
        )
        event_2 = TaskEvent(
            id=6,
            run_id=run_id,
            task_id=123,
            event="response.completed",
            data='{"type":"response.completed"}',
        )
        mock_get_task_events = Mock(return_value=[event_1, event_2])

        # Execute
        with (
            patch.object(self.state, "get_run_info", return_value=[mock_run]),
            patch.object(self.state, "get_task_events", new=mock_get_task_events),
            patch.object(
                self.state.federation_manager, "has_member", return_value=True
            ),
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(flwr_aid="user-123"),
            ),
        ):
            msgs = list(self.servicer.StreamRunEvents(request, ctx))

        # Assert
        mock_get_task_events.assert_called_once_with(
            run_id=run_id, after_task_event_id=4
        )
        self.assertEqual(len(msgs), 2)
        self.assertIsInstance(msgs[0], StreamRunEventsResponse)
        self.assertEqual(msgs[0].task_event.id, 5)
        self.assertEqual(msgs[0].task_event.task_id, 123)
        self.assertEqual(msgs[0].task_event.event, "response.output_text.delta")
        self.assertEqual(msgs[0].task_event.data, '{"delta":"Hel"}')
        self.assertEqual(msgs[1].task_event.id, 6)
        self.assertEqual(msgs[1].task_event.event, "response.completed")

    def test_stoprun_auth_unsuccessful_when_not_federation_member(self) -> None:
        """Test StopRun raises when requester is not a federation member."""
        run_id = self._create_dummy_run("run-owner")
        request = StopRunRequest(run_id=run_id)
        ctx = self.make_context()

        with (
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(flwr_aid="user-123"),
            ),
            patch.object(
                self.state.federation_manager, "has_member", return_value=False
            ),
        ):
            with self.assertRaises(FlowerError) as cm:
                self.servicer.StopRun(request, ctx)
            self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_FOUND)

    def test_stoprun_auth_successful(self) -> None:
        """Test StopRun succeeds for a federation member."""
        run_id = self._create_dummy_run("run-owner")
        request = StopRunRequest(run_id=run_id)
        ctx = self.make_context()

        with (
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(flwr_aid="user-123"),
            ),
            patch.object(
                self.state.federation_manager, "has_member", return_value=True
            ),
        ):
            response = self.servicer.StopRun(request, ctx)
            self.assertTrue(response.success)
            runs = self.state.get_run_info(run_ids=[run_id])
            run = runs[0] if runs else None
            self.assertEqual(cast(Run, run).status.status, Status.FINISHED)
            self.assertEqual(cast(Run, run).status.sub_status, SubStatus.STOPPED)

    def test_listruns_auth_unsuccessful_when_not_federation_member(self) -> None:
        """Test ListRuns raises when requester is not a federation member."""
        run_id = self._create_dummy_run("run-owner")
        request = ListRunsRequest(run_id=run_id)
        ctx = self.make_context()

        with (
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(
                    flwr_aid="user-123", account_name="test-account"
                ),
            ),
            patch.object(
                self.state.federation_manager, "has_member", return_value=False
            ),
        ):
            with self.assertRaises(FlowerError) as cm:
                self.servicer.ListRuns(request, ctx)
            self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_FOUND)

    def test_listruns_auth_run_success(self) -> None:
        """Test ListRuns succeeds for a federation member."""
        run_id = self._create_dummy_run("run-owner")
        request = ListRunsRequest(run_id=run_id)
        ctx = self.make_context()

        with (
            patch(
                "flwr.superlink.servicer.control.control_servicer.get_current_account_info",
                return_value=SimpleNamespace(
                    flwr_aid="user-123", account_name="test-account"
                ),
            ),
            patch.object(
                self.state.federation_manager, "has_member", return_value=True
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers.resolve_account_ids",
                return_value={"run-owner": "owner-account"},
            ),
        ):
            response = self.servicer.ListRuns(request, ctx)
            self.assertEqual(set(response.run_dict.keys()), {run_id})
            self.assertEqual(response.run_dict[run_id].account_name, "owner-account")


class TestValidateFederationAndNodesInRequest(unittest.TestCase):
    """Tests for federation and node validation helpers."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        objectstore_factory = Mock(store=Mock(return_value=Mock()))
        self.servicer = ControlServicer(
            linkstate_factory=LinkStateFactory(
                FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
            ),
            objectstore_factory=objectstore_factory,
            authn_plugin=(authn_plugin := NoOpControlAuthnPlugin(Mock(), False)),
        )
        account_info = authn_plugin.validate_tokens_in_metadata([])[1]
        assert account_info is not None
        self.aid: str = account_info.flwr_aid
        shared_account_info.set(account_info)
        self.state = self.servicer.linkstate_factory.state()

    def _make_context(self) -> MagicMock:
        """Create a mock gRPC context."""
        return MagicMock(spec=grpc.ServicerContext)

    def _create_owned_node(self, owner_aid: str) -> int:
        """Create a node owned by the given flwr_aid."""
        pub_key = public_key_to_bytes(generate_key_pairs()[1])
        return self.state.create_node(
            owner_aid=owner_aid,
            owner_name="test_owner",
            public_key=pub_key,
            heartbeat_interval=10,
        )

    # --- _validate_federation_membership_in_request tests ---

    def test_validate_membership_raises_when_federation_not_specified(self) -> None:
        """Test raises FlowerError when federation name is empty."""
        with self.assertRaises(FlowerError) as cm:
            _validate_federation_membership_in_request(self.state, self.aid, "")
        self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_SPECIFIED)

    def test_validate_membership_raises_when_federation_not_found(self) -> None:
        """Test raises when federation does not exist."""
        with self.assertRaises(FlowerError) as cm:
            _validate_federation_membership_in_request(
                self.state, self.aid, "@me/missing"
            )
        self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_FOUND)

    def test_validate_membership_raises_when_not_a_member(self) -> None:
        """Test raises when flwr_aid is not a member of the federation."""
        with self.assertRaises(FlowerError) as cm:
            _validate_federation_membership_in_request(
                self.state, "wrong-aid", NOOP_FEDERATION_ID
            )
        self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_FOUND)

    # --- _validate_federation_and_node_in_request tests ---

    def test_validate_raises_when_federation_not_specified(self) -> None:
        """Test raises FlowerError when federation name is empty."""
        with self.assertRaises(FlowerError) as cm:
            _validate_federation_and_node_in_request(self.state, self.aid, "", 1)
        self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_SPECIFIED)

    def test_validate_raises_when_node_not_owned(self) -> None:
        """Test raises when a node is not owned by the requester."""
        # Create a node owned by someone else
        node_id = self._create_owned_node("other-aid")
        with self.assertRaises(FlowerError) as cm:
            _validate_federation_and_node_in_request(
                self.state, self.aid, NOOP_FEDERATION_ID, node_id
            )
        self.assertEqual(cm.exception.code, ApiErrorCode.NODE_NOT_FOUND_OR_NOT_OWNER)

    def test_validate_raises_when_node_does_not_exist(self) -> None:
        """Test raises when a node ID does not exist."""
        with self.assertRaises(FlowerError) as cm:
            _validate_federation_and_node_in_request(
                self.state, self.aid, NOOP_FEDERATION_ID, 999999
            )
        self.assertEqual(cm.exception.code, ApiErrorCode.NODE_NOT_FOUND_OR_NOT_OWNER)

    # --- AddNodeToFederation / RemoveNodeFromFederation integration tests ---

    def test_add_node_to_federation_success(self) -> None:
        """Test AddNodeToFederation succeeds with valid inputs."""
        node_id = self._create_owned_node(self.aid)
        request = AddNodeToFederationRequest(
            federation_name=NOOP_FEDERATION_ID, node_id=node_id
        )
        ctx = self._make_context()

        with patch.object(
            self.state.federation_manager,
            "add_supernode",
            return_value=None,
        ) as mock_add:
            response = self.servicer.AddNodeToFederation(request, ctx)

        mock_add.assert_called_once_with(
            flwr_aid=self.aid,
            federation_id=NOOP_FEDERATION_ID,
            node_id=node_id,
        )
        self.assertIsInstance(response, AddNodeToFederationResponse)
        ctx.abort.assert_not_called()

    def test_add_node_to_federation_raises_no_federation(self) -> None:
        """Test AddNodeToFederation raises when no federation is specified."""
        request = AddNodeToFederationRequest(federation_name="", node_id=1)
        ctx = self._make_context()
        with self.assertRaises(FlowerError) as cm:
            self.servicer.AddNodeToFederation(request, ctx)
        self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_SPECIFIED)

    def test_remove_node_from_federation_success(self) -> None:
        """Test RemoveNodeFromFederation succeeds with valid inputs."""
        node_id = self._create_owned_node(self.aid)
        request = RemoveNodeFromFederationRequest(
            federation_name=NOOP_FEDERATION_ID, node_id=node_id
        )
        ctx = self._make_context()

        with patch.object(
            self.state.federation_manager,
            "remove_supernode",
            return_value=None,
        ) as mock_remove:
            response = self.servicer.RemoveNodeFromFederation(request, ctx)

        mock_remove.assert_called_once_with(
            flwr_aid=self.aid,
            federation_id=NOOP_FEDERATION_ID,
            node_id=node_id,
        )
        self.assertIsInstance(response, RemoveNodeFromFederationResponse)
        ctx.abort.assert_not_called()

    def test_remove_node_from_federation_raises_no_federation(self) -> None:
        """Test RemoveNodeFromFederation raises when no federation is specified."""
        request = RemoveNodeFromFederationRequest(federation_name="", node_id=1)
        ctx = self._make_context()
        with self.assertRaises(FlowerError) as cm:
            self.servicer.RemoveNodeFromFederation(request, ctx)
        self.assertEqual(cm.exception.code, ApiErrorCode.FEDERATION_NOT_SPECIFIED)


def test_format_verification_compact() -> None:
    """One test covering both 'with entries' and 'None' input."""
    # Case 1: verifications list present
    verifications: list[dict[str, str]] = [
        {"public_key_id": "key1", "sig": "abc", "algo": "ed25519"},
        {"public_key_id": "key2", "sig": "def", "algo": "ed25519"},
    ]
    out: dict[str, str] = _format_verification(verifications)

    # Should mark valid
    assert out["valid_license"] == "Valid"
    # public_key_id -> JSON of remaining fields
    v1: dict[str, str] = json.loads(out["key1"])
    v2: dict[str, str] = json.loads(out["key2"])
    assert v1 == {"sig": "abc", "algo": "ed25519"}
    assert v2 == {"sig": "def", "algo": "ed25519"}
