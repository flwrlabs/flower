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
"""Tests for Control OAuth connector handlers and servicer wiring."""


import json
from collections.abc import Sequence
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    BeginConnectorOAuthRequest,
    CompleteConnectorOAuthRequest,
    DisconnectConnectorRequest,
    ListConnectorsRequest,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.typing import JSONObject
from flwr.superlink.federation.noop_federation_manager import NoOpFederationManager
from flwr.superlink.servicer.control.control_account_auth_interceptor import (
    shared_account_info,
)
from flwr.superlink.servicer.control.control_servicer import ControlServicer

from .provider import ConnectorDefinition, ConnectorOAuthProvider, ConnectorOAuthResult

ACCOUNT_A = AccountInfo(flwr_aid="account-a", account_name="Account A")
ACCOUNT_B = AccountInfo(flwr_aid="account-b", account_name="Account B")


class FakeOAuthProvider:
    """Controllable OAuth provider used by connector handler tests."""

    def __init__(
        self,
        connector_ref: str = "slack",
        *,
        supports_pkce: bool = True,
    ) -> None:
        self.definition = ConnectorDefinition(
            connector_ref=connector_ref,
            display_name=connector_ref.title(),
            description=f"Connect {connector_ref.title()}.",
            supports_pkce=supports_pkce,
        )
        self.invalid_redirect = False
        self.fail_exchange = False
        self.authorization_state: str | None = None
        self.authorization_pkce_challenge: str | None = None
        self.exchange_calls = 0

    def resolve_redirect_uri(self, requested_redirect_uri: str) -> str:
        """Validate and normalize one test redirect URI."""
        if self.invalid_redirect:
            raise ValueError("Redirect URI rejected")
        return requested_redirect_uri.rstrip("/") + "/oauth/callback"

    def build_authorization_url(
        self,
        *,
        redirect_uri: str,
        state: str,
        pkce_challenge: str | None,
    ) -> str:
        """Capture authorization inputs and return a deterministic URL."""
        self.authorization_state = state
        self.authorization_pkce_challenge = pkce_challenge
        return f"https://oauth.example/authorize?state={state}"

    def exchange_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        pkce_verifier: str | None,
    ) -> ConnectorOAuthResult:
        """Return test credentials or simulate a provider failure."""
        self.exchange_calls += 1
        if self.fail_exchange:
            raise RuntimeError(f"Provider rejected sensitive code {code}")
        credentials: JSONObject = {
            "access_token": "access-secret",
            "refresh_token": "refresh-secret",
        }
        config: JSONObject = {"workspace": "flower"}
        return ConnectorOAuthResult(credentials=credentials, config=config)


class TestControlConnectorOAuth:
    """Exercise connector OAuth behavior through the Control servicer."""

    def setup_method(self) -> None:
        """Create an in-memory Control servicer and authenticated account."""
        self.objectstore_factory = Mock()
        self.objectstore_factory.store.return_value = Mock()
        self.linkstate_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME,
            NoOpFederationManager(),
            self.objectstore_factory,
        )
        self.provider = FakeOAuthProvider()
        self.servicer = self._make_servicer([self.provider])
        self.state = self.linkstate_factory.state()
        self.account_token = shared_account_info.set(ACCOUNT_A)

    def teardown_method(self) -> None:
        """Restore the authentication context after each test."""
        shared_account_info.reset(self.account_token)

    def _make_servicer(
        self, providers: Sequence[ConnectorOAuthProvider]
    ) -> ControlServicer:
        """Create a Control servicer sharing this test's LinkState."""
        return ControlServicer(
            linkstate_factory=self.linkstate_factory,
            objectstore_factory=self.objectstore_factory,
            authn_plugin=Mock(),
            connector_oauth_providers=providers,
        )

    def _begin_oauth(self) -> tuple[str, str]:
        """Begin OAuth and return the session ID and persisted state."""
        response = self.servicer.BeginConnectorOAuth(
            BeginConnectorOAuthRequest(
                connector_ref=" Slack ",
                redirect_uri="https://client.example",
            ),
            Mock(),
        )
        session = self.state.get_connector_oauth_session(
            oauth_session_id=response.oauth_session_id,
            flwr_aid=ACCOUNT_A.flwr_aid,
        )
        assert session is not None
        return response.oauth_session_id, session.state

    def test_list_connectors_is_sorted_and_account_scoped(self) -> None:
        """List registered providers with status for only the current account."""
        github = FakeOAuthProvider("GitHub", supports_pkce=False)
        servicer = self._make_servicer([self.provider, github])
        assert self.state.upsert_connector(
            flwr_aid=ACCOUNT_A.flwr_aid,
            connector_ref="slack",
            credentials_json="{}",
            config_json="{}",
        )
        assert self.state.upsert_connector(
            flwr_aid=ACCOUNT_B.flwr_aid,
            connector_ref="github",
            credentials_json="{}",
            config_json="{}",
        )

        response = servicer.ListConnectors(ListConnectorsRequest(), Mock())

        assert [connector.connector_ref for connector in response.connectors] == [
            "github",
            "slack",
        ]
        assert [connector.connected for connector in response.connectors] == [
            False,
            True,
        ]

    def test_begin_oauth_persists_normalized_session_and_pkce(self) -> None:
        """Begin OAuth with random state, an expiry, and provider-requested PKCE."""
        before = now()
        response = self.servicer.BeginConnectorOAuth(
            BeginConnectorOAuthRequest(
                connector_ref=" Slack ",
                redirect_uri="https://client.example/",
            ),
            Mock(),
        )
        after = now()

        session = self.state.get_connector_oauth_session(
            oauth_session_id=response.oauth_session_id,
            flwr_aid=ACCOUNT_A.flwr_aid,
        )
        assert session is not None
        assert response.connector_ref == "slack"
        assert response.authorization_url.startswith("https://oauth.example/")
        assert response.expires_at == session.expires_at
        assert session.redirect_uri == "https://client.example/oauth/callback"
        assert session.state == self.provider.authorization_state
        assert session.pkce_verifier
        assert self.provider.authorization_pkce_challenge
        assert before + timedelta(minutes=9) < _parse_iso(response.expires_at)
        assert _parse_iso(response.expires_at) <= after + timedelta(minutes=10)

    def test_begin_oauth_rejects_provider_redirect_validation(self) -> None:
        """Map a provider redirect rejection to an invalid request error."""
        self.provider.invalid_redirect = True

        with pytest.raises(FlowerError) as exc_info:
            self.servicer.BeginConnectorOAuth(
                BeginConnectorOAuthRequest(
                    connector_ref="slack",
                    redirect_uri="https://invalid.example",
                ),
                Mock(),
            )

        assert exc_info.value.code == ApiErrorCode.INVALID_CONNECTOR_REQUEST

    def test_complete_oauth_persists_credentials_and_is_single_use(self) -> None:
        """Complete OAuth once and store credentials under the authenticated account."""
        oauth_session_id, oauth_state = self._begin_oauth()
        request = CompleteConnectorOAuthRequest(
            oauth_session_id=oauth_session_id,
            code="authorization-code",
            state=oauth_state,
        )

        response = self.servicer.CompleteConnectorOAuth(request, Mock())

        assert response.connector_ref == "slack"
        connector = self.state.get_connector(
            flwr_aid=ACCOUNT_A.flwr_aid, connector_ref="slack"
        )
        assert connector is not None
        assert json.loads(connector.credentials_json) == {
            "access_token": "access-secret",
            "refresh_token": "refresh-secret",
        }
        assert json.loads(connector.config_json) == {"workspace": "flower"}
        assert self.provider.exchange_calls == 1

        with pytest.raises(FlowerError) as exc_info:
            self.servicer.CompleteConnectorOAuth(request, Mock())
        assert exc_info.value.code == ApiErrorCode.CONNECTOR_OAUTH_SESSION_INVALID
        assert self.provider.exchange_calls == 1

    def test_invalid_state_does_not_consume_oauth_session(self) -> None:
        """Reject a mismatched state while leaving the pending session usable."""
        oauth_session_id, oauth_state = self._begin_oauth()

        with pytest.raises(FlowerError) as exc_info:
            self.servicer.CompleteConnectorOAuth(
                CompleteConnectorOAuthRequest(
                    oauth_session_id=oauth_session_id,
                    code="authorization-code",
                    state="wrong-state",
                ),
                Mock(),
            )
        assert exc_info.value.code == ApiErrorCode.CONNECTOR_OAUTH_SESSION_INVALID

        response = self.servicer.CompleteConnectorOAuth(
            CompleteConnectorOAuthRequest(
                oauth_session_id=oauth_session_id,
                code="authorization-code",
                state=oauth_state,
            ),
            Mock(),
        )
        assert response.connector_ref == "slack"

    def test_expired_oauth_session_is_rejected_before_exchange(self) -> None:
        """Reject expired sessions without calling the provider."""
        session = self.state.create_connector_oauth_session(
            oauth_session_id="expired-session",
            flwr_aid=ACCOUNT_A.flwr_aid,
            connector_ref="slack",
            state="expected-state",
            redirect_uri="https://client.example/oauth/callback",
            pkce_verifier=None,
            expires_at=now() - timedelta(seconds=1),
        )
        assert session is not None

        with pytest.raises(FlowerError) as exc_info:
            self.servicer.CompleteConnectorOAuth(
                CompleteConnectorOAuthRequest(
                    oauth_session_id=session.oauth_session_id,
                    code="authorization-code",
                    state=session.state,
                ),
                Mock(),
            )

        assert exc_info.value.code == ApiErrorCode.CONNECTOR_OAUTH_SESSION_INVALID
        assert self.provider.exchange_calls == 0

    def test_oauth_sessions_and_connections_are_isolated_by_account(self) -> None:
        """Hide another account's OAuth session and connector credentials."""
        oauth_session_id, oauth_state = self._begin_oauth()
        assert self.state.upsert_connector(
            flwr_aid=ACCOUNT_A.flwr_aid,
            connector_ref="slack",
            credentials_json="{}",
            config_json="{}",
        )
        shared_account_info.set(ACCOUNT_B)

        with pytest.raises(FlowerError) as complete_error:
            self.servicer.CompleteConnectorOAuth(
                CompleteConnectorOAuthRequest(
                    oauth_session_id=oauth_session_id,
                    code="authorization-code",
                    state=oauth_state,
                ),
                Mock(),
            )
        assert (
            complete_error.value.code == ApiErrorCode.CONNECTOR_OAUTH_SESSION_NOT_FOUND
        )

        with pytest.raises(FlowerError) as disconnect_error:
            self.servicer.DisconnectConnector(
                DisconnectConnectorRequest(connector_ref="slack"), Mock()
            )
        assert disconnect_error.value.code == ApiErrorCode.CONNECTOR_NOT_FOUND
        assert self.state.get_connector(
            flwr_aid=ACCOUNT_A.flwr_aid, connector_ref="slack"
        )

    def test_provider_failure_is_sanitized_and_consumes_session(self) -> None:
        """Do not leak an OAuth code and prevent retries after provider exchange."""
        oauth_session_id, oauth_state = self._begin_oauth()
        self.provider.fail_exchange = True
        sensitive_code = "sensitive-authorization-code"
        request = CompleteConnectorOAuthRequest(
            oauth_session_id=oauth_session_id,
            code=sensitive_code,
            state=oauth_state,
        )

        with pytest.raises(FlowerError) as exc_info:
            self.servicer.CompleteConnectorOAuth(request, Mock())

        assert exc_info.value.code == ApiErrorCode.CONNECTOR_OAUTH_PROVIDER_FAILURE
        assert sensitive_code not in exc_info.value.message
        assert (
            self.state.get_connector(flwr_aid=ACCOUNT_A.flwr_aid, connector_ref="slack")
            is None
        )

        with pytest.raises(FlowerError) as retry_error:
            self.servicer.CompleteConnectorOAuth(request, Mock())
        assert retry_error.value.code == ApiErrorCode.CONNECTOR_OAUTH_SESSION_INVALID
        assert self.provider.exchange_calls == 1

    def test_persistence_failure_after_exchange_is_translated(self) -> None:
        """Translate a connector credential write failure without returning secrets."""
        oauth_session_id, oauth_state = self._begin_oauth()

        with (
            patch.object(self.state, "upsert_connector", return_value=False),
            pytest.raises(FlowerError) as exc_info,
        ):
            self.servicer.CompleteConnectorOAuth(
                CompleteConnectorOAuthRequest(
                    oauth_session_id=oauth_session_id,
                    code="authorization-code",
                    state=oauth_state,
                ),
                Mock(),
            )

        assert exc_info.value.code == ApiErrorCode.CONNECTOR_PERSISTENCE_FAILURE
        assert "access-secret" not in exc_info.value.message

    def test_disconnect_only_deletes_current_account_connection(self) -> None:
        """Disconnect the normalized ref without touching another account."""
        for account in (ACCOUNT_A, ACCOUNT_B):
            assert self.state.upsert_connector(
                flwr_aid=account.flwr_aid,
                connector_ref="slack",
                credentials_json="{}",
                config_json="{}",
            )

        self.servicer.DisconnectConnector(
            DisconnectConnectorRequest(connector_ref=" Slack "), Mock()
        )

        assert (
            self.state.get_connector(flwr_aid=ACCOUNT_A.flwr_aid, connector_ref="slack")
            is None
        )
        assert (
            self.state.get_connector(flwr_aid=ACCOUNT_B.flwr_aid, connector_ref="slack")
            is not None
        )
        with pytest.raises(FlowerError) as exc_info:
            self.servicer.DisconnectConnector(
                DisconnectConnectorRequest(connector_ref="slack"), Mock()
            )
        assert exc_info.value.code == ApiErrorCode.CONNECTOR_NOT_FOUND

    @pytest.mark.parametrize(
        ("oauth_request", "expected_code"),
        [
            (
                BeginConnectorOAuthRequest(
                    connector_ref="", redirect_uri="https://client.example"
                ),
                ApiErrorCode.INVALID_CONNECTOR_REQUEST,
            ),
            (
                BeginConnectorOAuthRequest(connector_ref="unknown", redirect_uri="x"),
                ApiErrorCode.CONNECTOR_NOT_FOUND,
            ),
        ],
    )
    def test_begin_oauth_validates_request(
        self, oauth_request: BeginConnectorOAuthRequest, expected_code: ApiErrorCode
    ) -> None:
        """Reject missing and unknown connector references."""
        with pytest.raises(FlowerError) as exc_info:
            self.servicer.BeginConnectorOAuth(oauth_request, Mock())
        assert exc_info.value.code == expected_code

    @pytest.mark.parametrize(
        "oauth_request",
        [
            CompleteConnectorOAuthRequest(
                oauth_session_id="session", code="", state="state"
            ),
            CompleteConnectorOAuthRequest(
                oauth_session_id="session", code="code", state=""
            ),
        ],
    )
    def test_complete_oauth_validates_required_fields(
        self, oauth_request: CompleteConnectorOAuthRequest
    ) -> None:
        """Reject missing authorization codes and OAuth state values."""
        with pytest.raises(FlowerError) as exc_info:
            self.servicer.CompleteConnectorOAuth(oauth_request, Mock())
        assert exc_info.value.code == ApiErrorCode.INVALID_CONNECTOR_REQUEST

    def test_disconnect_requires_connector_ref(self) -> None:
        """Reject disconnect requests without a connector reference."""
        with pytest.raises(FlowerError) as exc_info:
            self.servicer.DisconnectConnector(DisconnectConnectorRequest(), Mock())
        assert exc_info.value.code == ApiErrorCode.INVALID_CONNECTOR_REQUEST

    def test_duplicate_provider_references_are_rejected(self) -> None:
        """Reject ambiguous providers after connector-ref normalization."""
        duplicate = FakeOAuthProvider(" SLACK ")
        with pytest.raises(ValueError, match="Duplicate connector OAuth provider"):
            self._make_servicer([self.provider, duplicate])


def _parse_iso(value: str) -> datetime:
    """Parse one ISO timestamp for test assertions."""
    return datetime.fromisoformat(value)
