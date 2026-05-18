"""Executor abstractions and implementations for SuperExec."""

from .factory import SUPPORTED_EXECUTORS, get_executor
from .subprocess_executor import SubprocessExecutor
from .types import ExecutionSpec, Executor

__all__ = [
    "ExecutionSpec",
    "Executor",
    "SUPPORTED_EXECUTORS",
    "SubprocessExecutor",
    "get_executor",
]
