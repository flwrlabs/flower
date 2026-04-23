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


@dataclass(frozen=True)
class Task:  # pylint: disable=too-many-instance-attributes
    """Task details.

    Attributes
    ----------
    task_id : int
        Unique identifier of the task.
    type : str
        Task type.
    run_id : int
        Identifier of the run this task belongs to.
    status : str
        Current lifecycle status of the task.
    fab_hash : str | None
        Optional FAB hash associated with the task.
    model_ref : str | None
        Optional reference to the model used by the task.
    connector_ref : str | None
        Optional reference to the connector used by the task.
    token : str
        Token associated with the task.
    pending_at : str | None
        Timestamp recorded when the task entered the pending state.
    starting_at : str | None
        Timestamp recorded when the task started.
    running_at : str | None
        Timestamp recorded when the task entered the running state.
    finished_at : str | None
        Timestamp recorded when the task finished.
    """

    task_id: int
    type: str
    run_id: int
    status: str
    fab_hash: str | None
    model_ref: str | None
    connector_ref: str | None
    token: str
    pending_at: str | None
    starting_at: str | None
    running_at: str | None
    finished_at: str | None
