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
"""Supervise an AgentApp restart using only the host environment."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import FrameType


def main() -> int:
    """Run the app interpreter and recover tasks that fail before initialization."""
    runtime_env_dir = Path(os.environ["FLWR_AGENTAPP_RUNTIME_ENV"])
    state = json.loads((runtime_env_dir / "restart.json").read_text())
    environ = os.environ.copy()
    environ["PYTHONPATH"] = os.pathsep.join(state["pythonpath"])
    with subprocess.Popen(state["args"], env=environ) as process:

        def forward_signal(signum: int, _frame: FrameType | None) -> None:
            process.send_signal(signum)

        handlers = {
            sig: signal.signal(sig, forward_signal)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            returncode = process.wait()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if returncode != 0 and (runtime_env_dir / "task-input.pb").exists():
        # Import Flower only here, with the host paths and dependencies intact.
        from flwr.supercore.task_process.agent.run_agentapp import (  # pylint: disable=import-outside-toplevel
            _recover_failed_restart,
        )

        _recover_failed_restart(runtime_env_dir)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
