"""Executor types for SuperExec TaskExecutor processes."""

from dataclasses import dataclass
from typing import Literal, Protocol

AppIoKind = Literal["clientappio", "serverappio"]


@dataclass(frozen=True)
class ExecutionSpec:  # pylint: disable=too-many-instance-attributes
    """Describe one TaskExecutor process execution requested by SuperExec."""

    command: str
    appio_api_address: str
    appio_api_kind: AppIoKind
    token: str
    insecure: bool
    root_certificates_path: str | None
    runtime_dependency_install: bool
    parent_pid: int | None
    suppress_output: bool


class Executor(Protocol):
    """SuperExec component that starts TaskExecutor processes from an ExecutionSpec.

    An executor only starts processes; it does not wait, monitor, terminate,
    reconcile, or report task status.
    """

    def launch(self, spec: ExecutionSpec) -> None:
        """Start the TaskExecutor process described by the execution spec."""
