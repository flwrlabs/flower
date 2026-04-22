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
"""Flower Agent runtime stub."""


from flwr.common import EventType
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL
from flwr.common.exit import ExitCode, flwr_exit


def run_flower_agent(
    appio_api_address: str,
    parent_pid: int | None = None,
    health_server_address: str | None = None,
    superexec_auth_secret: bytes | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower Agent.

    This runtime is intentionally a stub until AgentApp execution support is added.
    """
    _ = (
        appio_api_address,
        parent_pid,
        health_server_address,
        superexec_auth_secret,
        runtime_dependency_install,
    )
    flwr_exit(
        ExitCode.SERVERAPP_EXCEPTION,
        "`flower-agent` is not implemented yet.",
        event_type=EventType.RUN_AGENT_LEAVE,
    )
