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
"""Opt-in timing markers for Kubernetes Agent runtime profiling."""

import json
import os
import time
from logging import INFO

from flwr.supercore.logger import FLOWER_LOGGER

RUNTIME_TIMING_LOGGING_ENV = "FLWR_RUNTIME_TIMING_LOGGING"
_RUNTIME_TIMING_MESSAGE = "runtime.timing"


def emit_runtime_timing(
    event: str, *, run_id: int, task_id: int, root_task_id: int
) -> None:
    """Best-effort emit one opaque runtime timing marker when enabled."""
    if os.getenv(RUNTIME_TIMING_LOGGING_ENV) != "1":
        return

    attributes = {
        "event": event,
        "emitted_at_unix_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "run_id": run_id,
        "task_id": task_id,
        "root_task_id": root_task_id,
    }
    try:
        FLOWER_LOGGER.log(
            INFO,
            "%s %s",
            _RUNTIME_TIMING_MESSAGE,
            json.dumps(attributes, separators=(",", ":"), sort_keys=True),
            extra={"runtime_timing": True},
        )
    except Exception:  # pylint: disable=broad-exception-caught
        # Timing collection must not affect task execution.
        return
