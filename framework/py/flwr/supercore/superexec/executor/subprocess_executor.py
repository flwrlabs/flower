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
"""Subprocess executor for SuperExec TaskExecutor processes."""


import subprocess

from flwr.supercore.constant import (
    TASK_TYPE_TO_APPIO_API_ADDRESS_ARG,
    TASK_TYPE_TO_COMMAND,
)

from .types import ExecutionSpec


class SubprocessExecutor:
    """Run TaskExecutor processes as local subprocesses."""

    def launch(self, spec: ExecutionSpec) -> None:
        """Start the TaskExecutor process described by the execution spec."""
        args = [
            TASK_TYPE_TO_COMMAND[spec.task_type],
            TASK_TYPE_TO_APPIO_API_ADDRESS_ARG[spec.task_type],
            spec.appio_api_address,
            "--token",
            spec.token,
        ]

        if spec.insecure:
            args.append("--insecure")
        elif spec.root_certificates_path is not None:
            args.extend(["--root-certificates", spec.root_certificates_path])

        if spec.parent_pid is not None:
            args.extend(["--parent-pid", str(spec.parent_pid)])

        if spec.runtime_dependency_install:
            args.append("--allow-runtime-dependency-installation")

        if spec.suppress_output:
            subprocess.Popen(  # pylint: disable=consider-using-with
                args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return

        subprocess.Popen(args)  # pylint: disable=consider-using-with
