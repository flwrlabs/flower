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
"""Subprocess launch backend for SuperExec executor processes."""


import subprocess

from .backend import AppIoKind, LaunchSpec

APPIO_ADDRESS_ARGS: dict[AppIoKind, str] = {
    "clientappio": "--clientappio-api-address",
    "serverappio": "--serverappio-api-address",
}


class SubprocessLaunchBackend:
    """Launch TaskExecutor work as a local subprocess."""

    def launch(self, spec: LaunchSpec) -> None:
        """Launch the TaskExecutor work described by the spec."""
        cmds = [spec.command]
        if spec.insecure:
            cmds.append("--insecure")
        elif spec.root_certificates_path:
            cmds += ["--root-certificates", spec.root_certificates_path]
        cmds += [APPIO_ADDRESS_ARGS[spec.appio_api_kind], spec.appio_api_address]
        cmds += ["--token", spec.token]
        if spec.parent_pid is not None:
            cmds += ["--parent-pid", str(spec.parent_pid)]
        if spec.runtime_dependency_install:
            cmds += ["--allow-runtime-dependency-installation"]

        # Launch without waiting for completion. Since SuperExec does not manage
        # this subprocess lifecycle, avoid using a `with` statement.
        if spec.suppress_output:
            subprocess.Popen(  # pylint: disable=consider-using-with
                cmds,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(cmds)  # pylint: disable=consider-using-with
