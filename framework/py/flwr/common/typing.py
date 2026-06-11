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
"""Flower type definitions."""


from dataclasses import dataclass

from flwr.app.user_config import UserConfig
from flwr.compat.common import typing as _compat_typing
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Member  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.constant import RunType

# Compatibility shims to avoid breaking `from flwr.commmon.typing import [...]``
ClientMessage = _compat_typing.ClientMessage
Code = _compat_typing.Code
Config = _compat_typing.Config
DisconnectRes = _compat_typing.DisconnectRes
EvaluateIns = _compat_typing.EvaluateIns
EvaluateRes = _compat_typing.EvaluateRes
FitIns = _compat_typing.FitIns
FitRes = _compat_typing.FitRes
GetParametersIns = _compat_typing.GetParametersIns
GetParametersRes = _compat_typing.GetParametersRes
GetPropertiesIns = _compat_typing.GetPropertiesIns
GetPropertiesRes = _compat_typing.GetPropertiesRes
Metrics = _compat_typing.Metrics
MetricsAggregationFn = _compat_typing.MetricsAggregationFn
NDArray = _compat_typing.NDArray
NDArrayFloat = _compat_typing.NDArrayFloat
NDArrayInt = _compat_typing.NDArrayInt
NDArrays = _compat_typing.NDArrays
Parameters = _compat_typing.Parameters
Properties = _compat_typing.Properties
ReconnectIns = _compat_typing.ReconnectIns
Scalar = _compat_typing.Scalar
ServerMessage = _compat_typing.ServerMessage
Status = _compat_typing.Status
Value = _compat_typing.Value


@dataclass
class RunStatus:
    """Run status information."""

    status: str
    sub_status: str
    details: str


@dataclass
class Run:  # pylint: disable=too-many-instance-attributes
    """Run details."""

    run_id: int
    fab_id: str
    fab_version: str
    fab_hash: str
    override_config: UserConfig
    pending_at: str
    starting_at: str
    running_at: str
    finished_at: str
    status: RunStatus
    flwr_aid: str
    federation: str
    primary_task_id: int | None
    bytes_sent: int
    bytes_recv: int
    clientapp_runtime: float
    run_type: str = ""
    series_id: int = 0

    @classmethod
    def create_empty(cls, run_id: int) -> "Run":
        """Return an empty Run instance."""
        return cls(
            run_id=run_id,
            fab_id="",
            fab_version="",
            fab_hash="",
            override_config={},
            pending_at="",
            starting_at="",
            running_at="",
            finished_at="",
            status=RunStatus(status="", sub_status="", details=""),
            flwr_aid="",
            federation="",
            primary_task_id=None,
            bytes_sent=0,
            bytes_recv=0,
            clientapp_runtime=0.0,
            run_type=RunType.SERVER_APP,
            series_id=0,
        )


@dataclass
class Fab:
    """Fab file representation."""

    hash_str: str
    content: bytes
    verifications: dict[str, str]


class RunNotRunningException(BaseException):
    """Raised when a run is not running."""


class InvalidRunStatusException(BaseException):
    """Raised when an RPC is invalidated by the RunStatus."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# OIDC account authentication types
@dataclass
class AccountAuthLoginDetails:
    """Account authentication login details."""

    authn_type: str
    device_code: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass
class AccountAuthCredentials:
    """Account authentication tokens."""

    access_token: str
    refresh_token: str


@dataclass
class AccountInfo:
    """User information for event log."""

    flwr_aid: str | None
    account_name: str | None


@dataclass
class Actor:
    """Event log actor."""

    actor_id: str | None
    description: str | None
    ip_address: str


@dataclass
class Event:
    """Event log description."""

    action: str
    run_id: int | None
    fab_hash: str | None


@dataclass
class LogEntry:
    """Event log record."""

    timestamp: str
    actor: Actor
    event: Event
    status: str


@dataclass
class Federation:  # pylint: disable=R0902
    """Federation details."""

    name: str
    description: str
    members: list[Member]
    nodes: list[NodeInfo]
    runs: list[Run]
    archived: bool
    simulation: bool
    config: SimulationConfig | None
