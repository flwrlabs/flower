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
"""Space AgentApp session setup, independent of Grid prompt delivery."""

from flwr.agentapp import AgentEvents
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import PullTaskInputResponse  # pylint: disable=E0611
from flwr.superlink.grid import HttpGrid

from .grid import RuntimeAgentGrid
from .session import AgentRuntime, RuntimeAgentConnectors, RuntimeAgentSession


def create_space_session(
    grid: HttpGrid, events: AgentEvents, task_input: PullTaskInputResponse
) -> RuntimeAgentSession:
    """Initialize a session whose AgentApp retrieves its own Space messages.

    Parameters
    ----------
    grid : HttpGrid
        Task-authenticated Runtime connection.
    events : AgentEvents
        Task event publisher shared with the worker lifecycle.
    task_input : PullTaskInputResponse
        Accepted Run, FAB, and task identifiers.

    Returns
    -------
    RuntimeAgentSession
        Session without an initial prompt or Grid instruction to reply to.
    """
    runtime = AgentRuntime(
        stub=grid._runtime_client,  # pylint: disable=protected-access
        run_id=task_input.run.run_id,
        task_id=task_input.task_id,
        start_run_request=StartRunRequest(
            fab=task_input.fab,
            override_config=task_input.run.override_config,
        ),
        events=events,
    )
    return RuntimeAgentSession(
        prompt="",
        connectors=RuntimeAgentConnectors(runtime),
        events=events,
        grid=RuntimeAgentGrid(grid, events, task_input.context.node_id),
    )
