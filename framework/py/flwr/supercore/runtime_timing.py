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
"""Opt-in, privacy-safe timing markers for the deployment runtime."""

from __future__ import annotations

import os
import time
from logging import INFO
from threading import Lock
from typing import Literal

from flwr.supercore.logger import FLOWER_LOGGER

RUNTIME_TIMING_LOGGING_ENV = "FLWR_RUNTIME_TIMING_LOGGING"
RUNTIME_TIMING_MESSAGE = "runtime.timing"

_task_lineage: dict[tuple[int, int], tuple[int, int]] = {}
_task_lineage_lock = Lock()

RuntimeTimingComponent = Literal["superlink", "superexec", "agent_task", "model_task"]
RuntimeTimingOutcome = Literal["ok", "error", "cancelled", "timeout"]
RuntimeTimingErrorKind = Literal[
    "dependency",
    "provider",
    "publisher",
    "state",
    "timeout",
    "transport",
    "unknown",
]


def is_runtime_timing_logging_enabled() -> bool:
    """Return whether runtime timing markers are enabled for this process."""
    return os.getenv(RUNTIME_TIMING_LOGGING_ENV) == "1"


def is_runtime_timing_task(task_type: str) -> bool:
    """Return whether a task type is part of the Agent runtime timeline."""
    return task_type in {"flwr-agentapp", "flwr-model"}


def register_runtime_task_lineage(
    *, run_id: int, task_id: int, parent_task_id: int, root_task_id: int
) -> None:
    """Store server-owned task lineage once for the current SuperLink process."""
    with _task_lineage_lock:
        _task_lineage.setdefault((run_id, task_id), (parent_task_id, root_task_id))


def get_runtime_task_lineage(*, run_id: int, task_id: int) -> tuple[int, int] | None:
    """Return the server-owned task lineage recorded for a task, if available."""
    with _task_lineage_lock:
        return _task_lineage.get((run_id, task_id))


def discard_runtime_task_lineage(*, run_id: int, task_id: int) -> None:
    """Discard terminal task lineage from the current SuperLink process."""
    with _task_lineage_lock:
        _task_lineage.pop((run_id, task_id), None)


def emit_runtime_timing(  # pylint: disable=too-many-arguments
    event: str,
    *,
    component: RuntimeTimingComponent,
    run_id: int | None,
    task_id: int | None,
    task_type: str | None,
    parent_task_id: int | None = None,
    root_task_id: int | None = None,
    outcome: RuntimeTimingOutcome | None = None,
    error_kind: RuntimeTimingErrorKind | None = None,
    executor_mode: Literal["fresh", "warm"] | None = None,
    process_mode: Literal["new", "persistent"] | None = None,
) -> None:
    """Emit one structured lifecycle marker without affecting task execution.

    The fixed message and LogRecord ``extra`` attributes are intentionally used so
    standard logging handlers can export the fields without parsing log text. This
    helper never accepts task payloads, credentials, URLs, exception text, or other
    user-controlled values.
    """
    if not is_runtime_timing_logging_enabled():
        return

    attributes = {
        "event": event,
        "emitted_at_unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "run_id": run_id,
        "task_id": task_id,
        "parent_task_id": parent_task_id,
        "root_task_id": root_task_id,
        "task_type": task_type,
        "component": component,
        "outcome": outcome,
        "error_kind": error_kind,
        "executor_mode": executor_mode,
        "process_mode": process_mode,
    }
    try:
        FLOWER_LOGGER.log(INFO, RUNTIME_TIMING_MESSAGE, extra=attributes)
    except Exception:  # pylint: disable=broad-exception-caught
        # Runtime logging must remain observational, including with custom handlers.
        return
