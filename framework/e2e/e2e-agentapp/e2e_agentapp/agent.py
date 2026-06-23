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
"""Custom AgentApp used to demonstrate local FAB bundling and execution."""


from __future__ import annotations

from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

_INPUT_KEY = "agent.input"
_INSTRUCTIONS_KEY = "agent.instructions"
_MAX_OUTPUT_TOKENS_KEY = "agent.max-output-tokens"
_MODEL_KEY = "agent.model"
_WEB_SEARCH_KEY = "agent.web-search"

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Run a custom single-turn AgentApp."""
    run_config = context.run_config

    agent_input = run_config.get(_INPUT_KEY)
    if not isinstance(agent_input, str) or not agent_input.strip():
        raise ValueError(f"`{_INPUT_KEY}` must be a non-empty string.")

    model = run_config.get(_MODEL_KEY)
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"`{_MODEL_KEY}` must be a non-empty string.")

    instructions = run_config.get(_INSTRUCTIONS_KEY)
    if not isinstance(instructions, str):
        raise ValueError(f"`{_INSTRUCTIONS_KEY}` must be a string.")

    max_output_tokens = run_config.get(_MAX_OUTPUT_TOKENS_KEY)
    if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
        raise ValueError(f"`{_MAX_OUTPUT_TOKENS_KEY}` must be a positive integer.")

    use_web_search = run_config.get(_WEB_SEARCH_KEY)
    if not isinstance(use_web_search, bool):
        raise ValueError(f"`{_WEB_SEARCH_KEY}` must be a boolean.")

    request: dict[str, Any] = {
        "model": model,
        "input": agent_input,
        "stream": True,
        "max_output_tokens": max_output_tokens,
    }
    if instructions:
        request["instructions"] = instructions
    if use_web_search:
        request["tools"] = ["web_search"]
        request["tool_choice"] = "required"

    agent.responses.create(request)
