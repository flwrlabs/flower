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


from dataclasses import dataclass

from flwr.supercore.constant import RunTime


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
    federation_name : str
        Target federation name.
    runtime : RunTime
        The runtime relevant to the action.
    """

    federation_name: str
    runtime: RunTime


@dataclass(frozen=True)
class CreateFederationContext(ActionContext):
    """Context for the `ActionType.CREATE_FEDERATION` action.

    Attributes
    ----------
    federation_name : str
        Target federation name.
    runtime : RunTime
        The runtime relevant to the action.
    visibility: str
        The visibility level of the federation to be created.
    """

    federation_name: str
    runtime: RunTime
    visibility: str


@dataclass(frozen=True)
class CreateInvitationContext(ActionContext):
    """Context for the `ActionType.CREATE_INVITATION` action.

    Attributes
    ----------
    federation_name : str
        Target federation name.
    invitee_account_name : str
        Account name of the invitee.
    runtime : RunTime
        The runtime relevant to the action.
    """

    federation_name: str
    invitee_account_name: str
    runtime: RunTime


@dataclass(frozen=True)
class AcceptInvitationContext(ActionContext):
    """Context for the `ActionType.ACCEPT_INVITATION` action.

    Attributes
    ----------
    federation_name : str
        Target federation name.
    runtime : RunTime
        The runtime relevant to the action.
    """

    federation_name: str
    runtime: RunTime
