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
"""Flower run definitions."""


from dataclasses import dataclass

from flwr.app.user_config import UserConfig
from flwr.supercore.constant import TaskType


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
    primary_task_type: str = ""
    series_id: int = 0
    account_name: str = ""

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
            primary_task_type=TaskType.SERVER_APP,
            series_id=0,
            account_name="",
        )


class RunNotRunningException(BaseException):
    """Raised when a run is not running."""


class InvalidRunStatusException(BaseException):
    """Raised when an RPC is invalidated by the RunStatus."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
