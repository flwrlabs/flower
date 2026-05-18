# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Simple Flower SuperExec plugin for ServerApp."""


import subprocess
from collections.abc import Sequence
from logging import ERROR
from typing import Any

from flwr.common.logger import log
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611

from .base_exec_plugin import BaseExecPlugin
from .superlink_task_command import (
    resolve_superlink_task_command,
    select_superlink_task,
)


class ServerAppExecPlugin(BaseExecPlugin):
    """Simple Flower SuperExec plugin for ServerApp.

    The plugin selects the first SuperLink task it supports.
    """

    appio_api_address_arg = "--serverappio-api-address"

    def select_task(self, candidate_tasks: Sequence[Task]) -> Task | None:
        """Select a supported SuperLink task to execute."""
        return select_superlink_task(candidate_tasks)

    def get_popen_kwargs(self) -> dict[str, Any]:
        """Isolate ServerApp stdio from the parent SuperLink process streams."""
        return {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }

    def launch_task(self, token: str, task: Task) -> None:
        """Launch the process to execute the given task using the given token."""
        command = resolve_superlink_task_command(task.type)
        if command is None:
            log(
                ERROR,
                "Unknown task type '%s' for task_id %d.",
                task.type,
                task.task_id,
            )
            return

        self.command = command

        # Launch the executor process
        super().launch_task(token, task)
