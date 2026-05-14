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
"""Launch-only backend contract for SuperExec executor processes."""


from dataclasses import dataclass
from typing import Literal, Protocol

AppIoKind = Literal["clientappio", "serverappio"]


@dataclass(frozen=True)
class LaunchSpec:
    """Describe one TaskExecutor launch."""

    command: str
    appio_api_address: str
    appio_api_kind: AppIoKind
    token: str
    insecure: bool
    root_certificates_path: str | None
    runtime_dependency_install: bool
    parent_pid: int | None
    suppress_output: bool


class LaunchBackend(Protocol):
    """Launch TaskExecutor work from a launch spec."""

    def launch(self, spec: LaunchSpec) -> None:
        """Launch the TaskExecutor work described by the spec."""
        ...
