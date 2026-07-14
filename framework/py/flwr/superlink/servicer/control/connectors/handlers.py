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
"""Business handlers for Control OAuth connector RPCs."""


import base64
import hashlib
import secrets
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from logging import INFO
from typing import NoReturn, TypeVar

from flwr.common.logger import log
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    BeginConnectorOAuthRequest,
    BeginConnectorOAuthResponse,
    CompleteConnectorOAuthRequest,
    CompleteConnectorOAuthResponse,
    Connector,
    DisconnectConnectorRequest,
    DisconnectConnectorResponse,
    ListConnectorsRequest,
    ListConnectorsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.utils import strict_json_dumps

from .provider import ConnectorOAuthProvider, normalize_connector_ref

OAUTH_SESSION_TTL = timedelta(minutes=10)
ProviderMap = Mapping[str, ConnectorOAuthProvider]
T = TypeVar("T")


def list_connectors(
    request: ListConnectorsRequest,
    account: AccountInfo,
    state: LinkState,
    providers: ProviderMap,
) -> ListConnectorsResponse:
    """List user-connectable OAuth providers and account connection status."""
    log(INFO, "ControlServicer.ListConnectors")
    _ = request

    connectors: list[Connector] = []
    for connector_ref, provider in sorted(providers.items()):
        connectors.append(
            Connector(
                connector_ref=connector_ref,
                display_name=provider.definition.display_name,
                description=provider.definition.description,
                connected=_is_connector_connected(
                    state, account.flwr_aid, connector_ref
                ),
            )
        )
    return ListConnectorsResponse(connectors=connectors)


def disconnect_connector(
    request: DisconnectConnectorRequest,
    account: AccountInfo,
    state: LinkState,
    providers: ProviderMap,
) -> DisconnectConnectorResponse:
    """Delete one account-scoped connector connection."""
    log(INFO, "ControlServicer.DisconnectConnector")
    connector_ref = _request_connector_ref(request.connector_ref)
    _get_provider(connector_ref, providers)

    deleted = _call_state(
        lambda: state.delete_connector(
            flwr_aid=account.flwr_aid, connector_ref=connector_ref
        ),
        "disconnect connector",
    )
    if not deleted:
        raise FlowerError(
            ApiErrorCode.CONNECTOR_NOT_FOUND,
            f"Connector '{connector_ref}' is not connected for this account.",
        )
    return DisconnectConnectorResponse()


def begin_connector_oauth(
    request: BeginConnectorOAuthRequest,
    account: AccountInfo,
    state: LinkState,
    providers: ProviderMap,
) -> BeginConnectorOAuthResponse:
    """Create a short-lived account-scoped OAuth session."""
    log(INFO, "ControlServicer.BeginConnectorOAuth")
    connector_ref = _request_connector_ref(request.connector_ref)
    if not request.redirect_uri.strip():
        _raise_invalid_request("redirect_uri is required")
    provider = _get_provider(connector_ref, providers)
    redirect_uri = _resolve_redirect_uri(
        provider, connector_ref, request.redirect_uri.strip()
    )

    oauth_session_id = secrets.token_urlsafe(32)
    oauth_state = secrets.token_urlsafe(32)
    pkce_verifier, pkce_challenge = _create_pkce_pair(provider.definition.supports_pkce)
    expires_at = now() + OAUTH_SESSION_TTL
    authorization_url = _call_provider(
        lambda: provider.build_authorization_url(
            redirect_uri=redirect_uri,
            state=oauth_state,
            pkce_challenge=pkce_challenge,
        ),
        connector_ref,
        "build authorization URL",
    )
    if not authorization_url:
        _raise_provider_failure(
            connector_ref, "build authorization URL", "empty response"
        )

    session = _call_state(
        lambda: state.create_connector_oauth_session(
            oauth_session_id=oauth_session_id,
            flwr_aid=account.flwr_aid,
            connector_ref=connector_ref,
            state=oauth_state,
            redirect_uri=redirect_uri,
            pkce_verifier=pkce_verifier,
            expires_at=expires_at,
        ),
        "create OAuth session",
    )
    if session is None:
        _raise_persistence_failure("OAuth session could not be created")

    return BeginConnectorOAuthResponse(
        oauth_session_id=session.oauth_session_id,
        authorization_url=authorization_url,
        connector_ref=session.connector_ref,
        expires_at=session.expires_at,
    )


def complete_connector_oauth(
    request: CompleteConnectorOAuthRequest,
    account: AccountInfo,
    state: LinkState,
    providers: ProviderMap,
) -> CompleteConnectorOAuthResponse:
    """Exchange an OAuth code and persist one account-scoped connection."""
    log(INFO, "ControlServicer.CompleteConnectorOAuth")
    oauth_session_id = request.oauth_session_id.strip()
    if not oauth_session_id:
        _raise_invalid_request("oauth_session_id is required")
    if not request.code.strip():
        _raise_invalid_request("code is required")
    if not request.state:
        _raise_invalid_request("state is required")

    session = _call_state(
        lambda: state.get_connector_oauth_session(
            oauth_session_id=oauth_session_id, flwr_aid=account.flwr_aid
        ),
        "get OAuth session",
    )
    if session is None:
        raise FlowerError(
            ApiErrorCode.CONNECTOR_OAUTH_SESSION_NOT_FOUND,
            "Connector OAuth session was not found for this account.",
        )

    expires_at = _parse_session_expiry(session.expires_at)
    if (
        session.completed_at is not None
        or expires_at <= now()
        or not secrets.compare_digest(
            request.state.encode("utf-8"), session.state.encode("utf-8")
        )
    ):
        _raise_invalid_oauth_session(session.oauth_session_id)

    connector_ref = normalize_connector_ref(session.connector_ref)
    provider = _get_provider(connector_ref, providers)

    claimed = _call_state(
        lambda: state.complete_connector_oauth_session(
            oauth_session_id=session.oauth_session_id,
            flwr_aid=account.flwr_aid,
        ),
        "complete OAuth session",
    )
    if not claimed:
        _raise_invalid_oauth_session(session.oauth_session_id)

    result = _call_provider(
        lambda: provider.exchange_code(
            code=request.code,
            redirect_uri=session.redirect_uri,
            pkce_verifier=session.pkce_verifier,
        ),
        connector_ref,
        "exchange authorization code",
    )
    try:
        credentials_json = strict_json_dumps(result.credentials, compact=True)
        config_json = strict_json_dumps(result.config, compact=True)
    except (TypeError, ValueError) as err:
        _raise_provider_failure(
            connector_ref,
            "serialize exchanged credentials",
            type(err).__name__,
        )

    stored = _call_state(
        lambda: state.upsert_connector(
            flwr_aid=account.flwr_aid,
            connector_ref=connector_ref,
            credentials_json=credentials_json,
            config_json=config_json,
        ),
        "store connector credentials",
    )
    if not stored:
        _raise_persistence_failure("Connector credentials could not be stored")
    return CompleteConnectorOAuthResponse(connector_ref=connector_ref)


def _request_connector_ref(connector_ref: str) -> str:
    """Normalize a request connector reference and require a non-empty value."""
    normalized = normalize_connector_ref(connector_ref)
    if not normalized:
        _raise_invalid_request("connector_ref is required")
    return normalized


def _is_connector_connected(
    state: LinkState, flwr_aid: str, connector_ref: str
) -> bool:
    """Return whether an account has stored credentials for a connector."""
    return (
        _call_state(
            lambda: state.get_connector(flwr_aid=flwr_aid, connector_ref=connector_ref),
            "list connectors",
        )
        is not None
    )


def _get_provider(connector_ref: str, providers: ProviderMap) -> ConnectorOAuthProvider:
    """Return the OAuth provider matching a canonical connector reference."""
    provider = providers.get(connector_ref)
    if provider is None:
        raise FlowerError(
            ApiErrorCode.CONNECTOR_NOT_FOUND,
            f"OAuth provider for connector '{connector_ref}' was not found.",
        )
    provider_ref = normalize_connector_ref(provider.definition.connector_ref)
    if provider_ref != connector_ref:
        _raise_persistence_failure("OAuth provider identity does not match its key")
    return provider


def _resolve_redirect_uri(
    provider: ConnectorOAuthProvider,
    connector_ref: str,
    requested_redirect_uri: str,
) -> str:
    """Resolve a provider redirect URI and translate validation failures."""
    try:
        redirect_uri = provider.resolve_redirect_uri(requested_redirect_uri)
    except ValueError:
        _raise_invalid_request("redirect_uri is not allowed for this connector")
    except Exception as err:  # Provider boundary
        _raise_provider_failure(
            connector_ref, "resolve redirect URI", type(err).__name__
        )
    if not redirect_uri:
        _raise_provider_failure(connector_ref, "resolve redirect URI", "empty response")
    return redirect_uri


def _create_pkce_pair(enabled: bool) -> tuple[str | None, str | None]:
    """Return a PKCE verifier and S256 challenge when requested by a provider."""
    if not enabled:
        return None, None
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _parse_session_expiry(value: str) -> datetime:
    """Parse a persisted timezone-aware OAuth session expiry."""
    try:
        expires_at = datetime.fromisoformat(value)
    except ValueError:
        _raise_persistence_failure("OAuth session expiry is invalid")
    if expires_at.utcoffset() is None:
        _raise_persistence_failure("OAuth session expiry is timezone-naive")
    return expires_at


def _call_provider(operation: Callable[[], T], connector_ref: str, action: str) -> T:
    """Call provider code without exposing provider exception details."""
    try:
        return operation()
    except Exception as err:  # Provider boundary
        _raise_provider_failure(connector_ref, action, type(err).__name__)


def _call_state(operation: Callable[[], T], action: str) -> T:
    """Call connector state code and translate persistence failures."""
    try:
        return operation()
    except Exception as err:  # Persistence boundary
        raise FlowerError(
            ApiErrorCode.CONNECTOR_PERSISTENCE_FAILURE,
            f"Failed to {action} ({type(err).__name__}).",
        ) from None


def _raise_invalid_request(reason: str) -> NoReturn:
    """Raise a sanitized invalid connector request error."""
    raise FlowerError(
        ApiErrorCode.INVALID_CONNECTOR_REQUEST,
        f"Invalid connector request: {reason}.",
    )


def _raise_invalid_oauth_session(oauth_session_id: str) -> NoReturn:
    """Raise a sanitized invalid OAuth session error."""
    raise FlowerError(
        ApiErrorCode.CONNECTOR_OAUTH_SESSION_INVALID,
        f"Connector OAuth session '{oauth_session_id}' is invalid or no longer "
        "pending.",
    )


def _raise_provider_failure(
    connector_ref: str, action: str, error_type: str
) -> NoReturn:
    """Raise a provider failure without including provider exception details."""
    raise FlowerError(
        ApiErrorCode.CONNECTOR_OAUTH_PROVIDER_FAILURE,
        f"Connector '{connector_ref}' failed to {action} ({error_type}).",
    ) from None


def _raise_persistence_failure(reason: str) -> NoReturn:
    """Raise a sanitized connector persistence error."""
    raise FlowerError(
        ApiErrorCode.CONNECTOR_PERSISTENCE_FAILURE,
        f"Connector persistence failure: {reason}.",
    )
