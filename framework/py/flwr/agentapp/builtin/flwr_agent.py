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
"""Built-in pass-through flwr agent."""


from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

# The model to use
_MODEL = "hf/flwrlabs/lizzy-dpo-gptq"

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Forward the initial user input to the GPT chat model with streaming enabled."""
    agent_input = context.run_config.get("agent.input")
    if not isinstance(agent_input, str) or not agent_input:
        raise ValueError(
            "context.run_config['agent.input'] must be a non-empty string."
        )
    agent.responses.create(
        {
            "model": _MODEL,
            "input": agent_input,
            "stream": True,
        }
    )
