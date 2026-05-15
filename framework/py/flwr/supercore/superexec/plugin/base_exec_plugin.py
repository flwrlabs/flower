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
"""Simple base Flower SuperExec plugin for app processes."""


import os
from collections.abc import Sequence
from typing import ClassVar

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.superexec.launch import (
    AppIoKind,
    LaunchSpec,
    SubprocessLaunchBackend,
)

from .exec_plugin import ExecPlugin


class BaseExecPlugin(ExecPlugin):
    """Simple Flower SuperExec plugin for app processes.

    The plugin always selects the first candidate task.
    """

    # Placeholders to be defined in subclasses
    command = ""
    appio_api_kind: ClassVar[AppIoKind]
    suppress_output = False

    def select_run_id(self, candidate_run_ids: Sequence[int]) -> int | None:
        """Select a run ID to execute from a sequence of candidates."""
        if not candidate_run_ids:
            return None
        return candidate_run_ids[0]

    def select_task(self, candidate_tasks: Sequence[Task]) -> Task | None:
        """Select a Task to execute from a sequence of candidates."""
        if not candidate_tasks:
            return None
        return candidate_tasks[0]

    def launch_task(self, token: str, task: Task) -> None:
        """Launch the process to execute the given task using the given token."""
        backend = self.launch_backend or SubprocessLaunchBackend()
        backend.launch(self._build_launch_spec(token=token, task=task))

    def _build_launch_spec(  # pylint: disable=unused-argument
        self, token: str, task: Task
    ) -> LaunchSpec:
        """Build the launch spec for the selected task."""
        return LaunchSpec(
            command=self.command,
            appio_api_address=self.appio_api_address,
            appio_api_kind=self.appio_api_kind,
            token=token,
            insecure=self.insecure,
            root_certificates_path=self.root_certificates_path,
            runtime_dependency_install=self.runtime_dependency_install,
            parent_pid=os.getpid(),
            suppress_output=self.suppress_output,
        )
