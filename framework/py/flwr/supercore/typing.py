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
"""Flower SuperCore type definitions."""


from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from flwr.supercore.constant import RunTime

JSONScalar: TypeAlias = bool | float | int | str | None
JSONArray: TypeAlias = Sequence["JSONValue"]
JSONValue: TypeAlias = JSONScalar | dict[str, "JSONValue"] | JSONArray
JSONObject: TypeAlias = dict[str, JSONValue]


@dataclass(frozen=True)
class ConnectorRecord:
    """Persisted connector configuration and credentials."""

    flwr_aid: str
    connector_ref: str
    credentials_json: str
    config_json: str


@dataclass(frozen=True)
class ConnectorOAuthSessionRecord:  # pylint: disable=too-many-instance-attributes
    """Persisted OAuth session for connector authorization."""

    oauth_session_id: str
    flwr_aid: str
    connector_ref: str
    state: str
    redirect_uri: str
    pkce_verifier: str | None
    created_at: str
    expires_at: str
    completed_at: str | None


@dataclass(frozen=True)
class ActionContext:
    """Base context for authorization checks in ``can_execute``."""


@dataclass(frozen=True)
class RegisterSupernodeContext(ActionContext):
    """Context for the `ActionType.REGISTER_SUPERNODE` action."""


@dataclass(frozen=True)
class StartRunContext(ActionContext):
    """Context for the `ActionType.START_RUN` action.

    Attributes
    ----------
    federation_id : str
        Target federation ID.
    runtime : RunTime
        The runtime relevant to the action.
    """

    federation_id: str
    runtime: RunTime


@dataclass(frozen=True)
class CreateFederationContext(ActionContext):
    """Context for the `ActionType.CREATE_FEDERATION` action.

    Attributes
    ----------
    federation_id : str
        Target federation ID.
    runtime : RunTime
        The runtime relevant to the action.
    visibility: str
        The visibility level of the federation to be created.
    """

    federation_id: str
    runtime: RunTime
    visibility: str


@dataclass(frozen=True)
class CreateInvitationContext(ActionContext):
    """Context for the `ActionType.CREATE_INVITATION` action.

    Attributes
    ----------
    federation_id : str
        Target federation ID.
    invitee_account_name : str
        Account name of the invitee.
    runtime : RunTime
        The runtime relevant to the action.
    """

    federation_id: str
    invitee_account_name: str
    runtime: RunTime


@dataclass(frozen=True)
class AcceptInvitationContext(ActionContext):
    """Context for the `ActionType.ACCEPT_INVITATION` action.

    Attributes
    ----------
    federation_id : str
        Target federation ID.
    runtime : RunTime
        The runtime relevant to the action.
    """

    federation_id: str
    runtime: RunTime
