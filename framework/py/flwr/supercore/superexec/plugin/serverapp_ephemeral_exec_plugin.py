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
from flwr.supercore.constant import RunType

from .base_ephemeral_exec_plugin import BaseEphemeralExecPlugin


class ServerAppEphemeralExecPlugin(BaseEphemeralExecPlugin):
    """Simple ephemeral Flower SuperExec plugin for ServerApp processes."""

    appio_api_address_arg = "--serverappio-api-address"

    def launch_app(self, token: str, run_id: int) -> None:
        """Launch the application associated with a given run ID and token."""
        # Determine the command to launch based on the run type
        run = self.get_run(run_id)
        if run.run_type == RunType.SERVER_APP:
            self.command = "flwr-serverapp"
        elif run.run_type == RunType.SIMULATION:
            self.command = "flwr-simulation"
        else:
            log(ERROR, "Unknown run type '%s' for run_id %d.", run.run_type, run_id)
            return

        # Launch the executor process
        super().launch_app(token, run_id)
