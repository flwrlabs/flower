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
"""Opt-in timing logs for trusted runtime profiling."""

from __future__ import annotations

import os
import sys
import time
from uuid import uuid4

from flwr.supercore.constant import TaskType

_RUNTIME_TIMING_LOGGING_ENV = "FLWR_RUNTIME_TIMING_LOGGING"
_PROFILED_TASK_TYPE_PREFIXES = {
    TaskType.AGENT_APP: "agent",
    TaskType.MODEL: "model",
}


def is_runtime_timing_logging_enabled() -> bool:
    """Return whether trusted runtime timing logs are enabled."""
    return os.getenv(_RUNTIME_TIMING_LOGGING_ENV, "").strip() == "1"


def new_runtime_timing_id() -> str:
    """Return an opaque identifier for one profiled task execution."""
    return uuid4().hex


def is_profiled_runtime_task_type(task_type: TaskType) -> bool:
    """Return whether a task type has an opt-in timing marker chain."""
    return task_type in _PROFILED_TASK_TYPE_PREFIXES


def runtime_timing_marker(task_type: TaskType, boundary: str) -> str | None:
    """Return the marker name for one profiled task-type boundary."""
    prefix = _PROFILED_TASK_TYPE_PREFIXES.get(task_type)
    return f"{prefix}_{boundary}" if prefix is not None else None


def log_runtime_timing(marker: str, *, timing_id: str | None, **fields: str) -> None:
    """Emit one safe, structured runtime timing marker when enabled."""
    if timing_id is None or not is_runtime_timing_logging_enabled():
        return

    details = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    # AgentApp mirrors ``sys.stdout`` and ``sys.stderr`` to PushLogs. The
    # original stdout still reaches the trusted TaskExecutor Pod log without
    # entering the user-visible run-log queue.
    output = sys.__stdout__
    if output is None:
        return
    output.write(
        "INFO     : runtime_timing "
        f"marker={marker} timing_id={timing_id} unix_time_ns={time.time_ns()}"
        f"{f' {details}' if details else ''}\n"
    )
    output.flush()
