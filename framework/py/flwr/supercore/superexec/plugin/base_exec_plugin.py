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
from logging import ERROR
from typing import ClassVar

from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.constant import TaskType
from flwr.supercore.runtime_timing import is_runtime_timing_logging_enabled
from flwr.supercore.superexec.executor import ExecutionSpec, Executor, LaunchResult

from .exec_plugin import ExecPlugin


class BaseExecPlugin(ExecPlugin):
    """Simple Flower SuperExec plugin for app processes.

    The plugin always selects the first candidate task.
    """

    # Placeholders to be defined in subclasses
    supported_task_types: ClassVar[frozenset[TaskType]]
    suppress_output = False
    visible_output_task_types: ClassVar[frozenset[TaskType]] = frozenset()

    def __init__(  # pylint: disable=R0913, R0917
        self,
        runtime_api_address: str,
        insecure: bool,
        root_certificates_path: str | None,
        runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
        *,
        executor: Executor,
    ) -> None:
        super().__init__(
            runtime_api_address=runtime_api_address,
            insecure=insecure,
            root_certificates_path=root_certificates_path,
            runtime_dependency_install=runtime_dependency_install,
            executor=executor,
        )
        self.executor: Executor = executor

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

    def launch_task(self, token: str, task: Task) -> LaunchResult:
        """Launch the process to execute the given task using the given token."""
        task_type = self._get_supported_task_type(task)
        if task_type is None:
            return LaunchResult.failed(
                f"Unknown task type '{task.type}' for task_id {task.task_id}."
            )
        return self.executor.launch(
            self._build_execution_spec(
                token=token,
                task_type=task_type,
                task_id=task.task_id,
                run_id=task.run_id,
            )
        )

    def _build_execution_spec(
        self, token: str, task_type: TaskType, task_id: int, run_id: int
    ) -> ExecutionSpec:
        """Build the execution spec for the selected task."""
        return ExecutionSpec(
            task_type=task_type,
            runtime_api_address=self.runtime_api_address,
            token=token,
            insecure=self.insecure,
            root_certificates_path=self.root_certificates_path,
            runtime_dependency_install=self.runtime_dependency_install,
            parent_pid=os.getpid(),
            suppress_output=(
                self.suppress_output and task_type not in self.visible_output_task_types
            ),
            task_id=task_id,
            run_id=run_id,
            runtime_timing_logging=is_runtime_timing_logging_enabled(),
        )

    def _get_supported_task_type(self, task: Task) -> TaskType | None:
        """Return the task type if it is supported by the plugin."""
        try:
            task_type = TaskType(task.type)
        except ValueError:
            task_type = None

        if task_type not in self.supported_task_types:
            log(
                ERROR,
                "Unknown task type '%s' for task_id %d.",
                task.type,
                task.task_id,
            )
            return None

        return task_type
