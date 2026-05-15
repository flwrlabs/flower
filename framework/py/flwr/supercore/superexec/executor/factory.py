"""Executor factory for SuperExec TaskExecutor processes."""

from flwr.supercore.constant import ExecutorType

from .subprocess_executor import SubprocessExecutor
from .types import Executor

SUPPORTED_EXECUTORS = tuple(item.value for item in ExecutorType)


def get_executor(executor_type: str) -> Executor:
    """Return the executor for the configured executor type."""
    if executor_type == ExecutorType.SUBPROCESS:
        return SubprocessExecutor()

    raise ValueError(f"Unsupported executor: {executor_type}")
