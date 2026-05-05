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
"""Simple ephemeral Flower SuperExec plugin for ServerApp."""


from logging import ERROR

from flwr.common.logger import log
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TaskType

from .base_ephemeral_exec_plugin import BaseEphemeralExecPlugin


class ServerAppEphemeralExecPlugin(BaseEphemeralExecPlugin):
    """Simple ephemeral Flower SuperExec plugin for ServerApp processes."""

    appio_api_address_arg = "--serverappio-api-address"

    def launch_app(self, token: str, task: Task) -> None:
        """Launch the application associated with a given task and token."""
        # Determine the command to launch based on the task type
        if task.task_type == TaskType.SERVER_APP:
            self.command = "flwr-serverapp"
        elif task.task_type == TaskType.SIMULATION:
            self.command = "flwr-simulation"
        else:
            log(
                ERROR,
                "Unknown task type '%s' for task_id %d.",
                task.task_type,
                task.task_id,
            )
            return

        # Launch the executor process
        super().launch_app(token, task)
