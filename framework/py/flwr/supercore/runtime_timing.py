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

import json
import os
import time
from logging import INFO
from threading import Lock
from typing import TYPE_CHECKING, Literal

from flwr.supercore.logger import FLOWER_LOGGER

if TYPE_CHECKING:
    from flwr.proto.task_pb2 import Task
    from flwr.supercore.corestate import CoreState

RUNTIME_TIMING_LOGGING_ENV = "FLWR_RUNTIME_TIMING_LOGGING"
RUNTIME_TIMING_MESSAGE = "runtime.timing"

_task_lineage: dict[tuple[int, int], tuple[int, int]] = {}
_first_persisted_event_tasks: set[tuple[int, int]] = set()
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


def mark_runtime_task_first_event_persisted(*, run_id: int, task_id: int) -> bool:
    """Mark the first persisted event for a task and return whether it was new."""
    key = (run_id, task_id)
    with _task_lineage_lock:
        if key in _first_persisted_event_tasks:
            return False
        _first_persisted_event_tasks.add(key)
        return True


def discard_runtime_task_lineage(*, run_id: int, task_id: int) -> None:
    """Discard terminal task lineage from the current SuperLink process."""
    key = (run_id, task_id)
    with _task_lineage_lock:
        _task_lineage.pop(key, None)
        _first_persisted_event_tasks.discard(key)


def complete_runtime_timing_tasks(
    *,
    state: CoreState,
    tasks: list[Task],
    outcome: RuntimeTimingOutcome,
    error_kind: RuntimeTimingErrorKind | None = None,
) -> None:
    """Record and discard timing state for tasks finished outside runtime handlers."""
    for task in tasks:
        if not is_runtime_timing_task(task.type):
            continue

        if is_runtime_timing_logging_enabled():
            lineage = get_runtime_task_lineage(run_id=task.run_id, task_id=task.task_id)
            if lineage is None:
                try:
                    lineage = state.get_task_lineage(task.task_id)
                except Exception:  # pylint: disable=broad-exception-caught
                    lineage = None
            parent_task_id, root_task_id = (
                lineage
                if lineage is not None
                else (
                    None,
                    task.task_id if task.type == "flwr-agentapp" else None,
                )
            )
            emit_runtime_timing(
                "runtime.task.completed.persisted",
                component="superlink",
                run_id=task.run_id,
                task_id=task.task_id,
                parent_task_id=parent_task_id,
                root_task_id=root_task_id,
                task_type=task.type,
                outcome=outcome,
                error_kind=error_kind,
            )

        discard_runtime_task_lineage(run_id=task.run_id, task_id=task.task_id)


def complete_expired_runtime_timing_tasks(
    *, state: CoreState, tasks: list[Task]
) -> None:
    """Record and discard timing state for tasks that expired while running."""
    complete_runtime_timing_tasks(
        state=state,
        tasks=tasks,
        outcome="timeout",
        error_kind="timeout",
    )


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

    The fixed message prefix and JSON attributes let standard logging handlers carry
    the fields without custom formatting. The same attributes remain available on
    the LogRecord for structured handlers. This helper never accepts task payloads,
    credentials, URLs, exception text, or other user-controlled values.
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
    serialized_attributes = json.dumps(
        attributes, separators=(",", ":"), sort_keys=True
    )
    try:
        FLOWER_LOGGER.log(
            INFO,
            "%s %s",
            RUNTIME_TIMING_MESSAGE,
            serialized_attributes,
            extra=attributes,
        )
    except Exception:  # pylint: disable=broad-exception-caught
        # Runtime logging must remain observational, including with custom handlers.
        return
