"""Executor abstractions and implementations for SuperExec."""

from .factory import SUPPORTED_EXECUTORS, get_executor
from .subprocess_executor import SubprocessExecutor
from .types import AppIoKind, ExecutionSpec, Executor

__all__ = [
    "AppIoKind",
    "ExecutionSpec",
    "Executor",
    "SUPPORTED_EXECUTORS",
    "SubprocessExecutor",
    "get_executor",
]
