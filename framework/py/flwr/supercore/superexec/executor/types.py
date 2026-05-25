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
"""Executor types for SuperExec TaskExecutor processes."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from flwr.supercore.constant import TaskType


@dataclass(frozen=True)
class ExecutionSpec:  # pylint: disable=too-many-instance-attributes
    """Describe one TaskExecutor process execution requested by SuperExec."""

    task_type: TaskType
    appio_api_address: str
    token: str
    insecure: bool
    root_certificates_path: str | None
    runtime_dependency_install: bool
    parent_pid: int | None
    suppress_output: bool


class LaunchResultStatus(StrEnum):
    """Immediate outcome of an executor launch attempt."""

    ACCEPTED = "accepted"
    CAPACITY_REJECTED = "capacity_rejected"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LaunchResult:
    """Structured result returned by an executor launch attempt."""

    status: LaunchResultStatus
    message: str | None = None

    @classmethod
    def accepted(cls) -> "LaunchResult":
        """Return a result for a launch accepted by the backend."""
        return cls(status=LaunchResultStatus.ACCEPTED)

    @classmethod
    def capacity_rejected(cls, message: str | None = None) -> "LaunchResult":
        """Return a result for capacity rejection after launch was attempted."""
        return cls(status=LaunchResultStatus.CAPACITY_REJECTED, message=message)

    @classmethod
    def failed(cls, message: str | None = None) -> "LaunchResult":
        """Return a result for a known launch failure."""
        return cls(status=LaunchResultStatus.FAILED, message=message)

    @classmethod
    def unknown(cls, message: str | None = None) -> "LaunchResult":
        """Return a result for an ambiguous launch outcome."""
        return cls(status=LaunchResultStatus.UNKNOWN, message=message)


class Executor(Protocol):
    """SuperExec component that starts TaskExecutor processes from an ExecutionSpec.

    An executor only starts processes and reports the immediate launch outcome;
    it does not wait, monitor, terminate, reconcile, or report task status.
    """

    def launch(self, spec: ExecutionSpec) -> LaunchResult:
        """Start the TaskExecutor process described by the execution spec."""
