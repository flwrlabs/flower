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
"""Run AgentApp."""


from logging import DEBUG

from flwr.agentapp import AgentApp, AgentSession, LoadAgentAppError
from flwr.app import Context
from flwr.common.logger import log
from flwr.common.object_ref import load_app


def run(
    agent: AgentSession,
    context: Context,
    agent_app_dir: str,
    agent_app_attr: str | None = None,
    loaded_agent_app: AgentApp | None = None,
) -> Context:
    """Run AgentApp with a given AgentSession."""
    if not (agent_app_attr is None) ^ (loaded_agent_app is None):
        raise ValueError(
            "Either `agent_app_attr` or `loaded_agent_app` should be set "
            "but not both."
        )

    def _load() -> AgentApp:
        if agent_app_attr:
            agent_app: AgentApp = load_app(
                agent_app_attr, LoadAgentAppError, agent_app_dir
            )

            if not isinstance(agent_app, AgentApp):
                raise LoadAgentAppError(
                    f"Attribute {agent_app_attr} is not of type {AgentApp}",
                ) from None

        if loaded_agent_app:
            agent_app = loaded_agent_app
        return agent_app

    agent_app = _load()

    agent_app(agent=agent, context=context)

    log(DEBUG, "AgentApp finished running.")
    return context
