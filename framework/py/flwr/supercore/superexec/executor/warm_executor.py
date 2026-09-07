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
"""Inert process for a pre-started warm TaskExecutor Pod."""

from __future__ import annotations

import signal
from pathlib import Path
from threading import Event
from types import FrameType

WARM_EXECUTOR_MODULE = "flwr.supercore.superexec.executor.warm_executor"
WARM_EXECUTOR_READY_DIRECTORY = "/tmp/flwr-warm-executor"
WARM_EXECUTOR_READY_FILE = f"{WARM_EXECUTOR_READY_DIRECTORY}/ready"
WARM_EXECUTOR_READINESS_COMMAND = (
    "python",
    "-c",
    (
        "from pathlib import Path; "
        f"raise SystemExit(not Path({WARM_EXECUTOR_READY_FILE!r}).is_file())"
    ),
)


def run_warm_executor(
    ready_file: Path = Path(WARM_EXECUTOR_READY_FILE),
    stop_event: Event | None = None,
) -> None:
    """Report readiness, then wait without claiming or executing a task."""
    if stop_event is None:
        stop_event = Event()

        def _request_stop(_signal_number: int, _frame: FrameType | None) -> None:
            stop_event.set()

        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)

    ready_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.touch()
    try:
        stop_event.wait()
    finally:
        ready_file.unlink(missing_ok=True)


if __name__ == "__main__":
    run_warm_executor()
