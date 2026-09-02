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
"""Simple base ephemeral Flower SuperExec plugin for app processes."""


import os
import subprocess
from collections.abc import Callable, Sequence

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.runtime_timing import (
    emit_runtime_timing,
    is_runtime_timing_logging_enabled,
    is_runtime_timing_task,
)

from .exec_plugin import ExecPlugin


class BaseEphemeralExecPlugin(ExecPlugin):
    """Simple ephemeral Flower SuperExec plugin for app processes.

    The plugin always selects the first candidate task, launches the corresponding app
    process, waits for it to finish, and then terminates the SuperExec process.
    """

    # Placeholders to be defined in subclasses
    command = ""
    runtime_api_address_arg = ""
    cleanup_before_launch: Callable[[], None] | None = None

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

    def launch_task(self, token: str, task: Task) -> None:  # type: ignore[override]
        """Launch the process to execute the given task using the given token."""
        cmds = [self.command]
        if self.insecure:
            cmds += ["--insecure"]
        elif self.root_certificates_path:
            cmds += ["--root-certificates", self.root_certificates_path]
        cmds += [self.runtime_api_address_arg, self.runtime_api_address]
        cmds += ["--token", token]
        cmds += ["--parent-pid", str(os.getpid())]
        if self.runtime_dependency_install:
            cmds += ["--allow-runtime-dependency-installation"]
        timing_enabled = is_runtime_timing_logging_enabled() and is_runtime_timing_task(
            task.type
        )
        if timing_enabled:
            cmds += [
                "--runtime-timing-run-id",
                str(task.run_id),
                "--runtime-timing-task-id",
                str(task.task_id),
            ]
            root_task_id = task.task_id if task.type == "flwr-agentapp" else None
        # Perform any cleanup before launching the app
        if self.cleanup_before_launch is not None:
            self.cleanup_before_launch()
        # Start the app process before recording that the executor accepted it.
        with subprocess.Popen(cmds):
            if timing_enabled:
                emit_runtime_timing(
                    "runtime.executor.submission.accepted",
                    component="superexec",
                    run_id=task.run_id,
                    task_id=task.task_id,
                    root_task_id=root_task_id,
                    task_type=task.type,
                    executor_mode="fresh",
                )
                emit_runtime_timing(
                    "runtime.task.dispatch.accepted",
                    component="superexec",
                    run_id=task.run_id,
                    task_id=task.task_id,
                    root_task_id=root_task_id,
                    task_type=task.type,
                    executor_mode="fresh",
                )
                emit_runtime_timing(
                    "runtime.executor.child.spawn.started",
                    component="superexec",
                    run_id=task.run_id,
                    task_id=task.task_id,
                    root_task_id=root_task_id,
                    task_type=task.type,
                    executor_mode="fresh",
                )
        flwr_exit(ExitCode.SUCCESS, "App process finished, exiting SuperExec.")
