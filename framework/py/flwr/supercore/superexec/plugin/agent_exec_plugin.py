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
"""Stub Flower SuperExec plugin for Agent."""


import os
import subprocess
from typing import Any

from .base_exec_plugin import BaseExecPlugin


class AgentExecPlugin(BaseExecPlugin):
    """Stub Flower SuperExec plugin for Agent."""

    command = "flower-agent"
    appio_api_address_arg = "--appio-api-address"

    def launch_app(self, token: str, run_id: int) -> None:
        """Launch Flower Agent using the same surface as Flower SuperExec."""
        _ = (token, run_id)
        cmds = [self.command, "--insecure"]
        cmds += [self.appio_api_address_arg, self.appio_api_address]
        cmds += ["--parent-pid", str(os.getpid())]
        if self.runtime_dependency_install:
            cmds += ["--allow-runtime-dependency-installation"]
        # pylint: disable-next=consider-using-with
        subprocess.Popen(cmds, **self.get_popen_kwargs())

    def get_popen_kwargs(self) -> dict[str, Any]:
        """Return subprocess keyword arguments when launching agent processes."""
        return {}
