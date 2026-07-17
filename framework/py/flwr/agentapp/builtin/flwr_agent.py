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


from datetime import UTC, datetime
from typing import cast

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from flwr.supercore.typing import JSONObject

# The model to use
_MODEL = "openai/gpt-5.5"
_MAX_TOOL_ROUNDS = 8

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Run the built-in agent with automation tool support."""
    agent_input = context.run_config.get("agent.input")
    if not isinstance(agent_input, str) or not agent_input:
        raise ValueError(
            "context.run_config['agent.input'] must be a non-empty string."
        )
    tool = agent.connectors.tools(["start_automation"])[0]
    instructions = (
        "Use start_automation only when the user explicitly requests future or "
        "recurring execution. Ask for clarification when the schedule is "
        "ambiguous. Current UTC time: "
        f"{datetime.now(tz=UTC).isoformat(timespec='seconds')}."
    )
    request: JSONObject = {
        "model": _MODEL,
        "input": agent_input,
        "stream": True,
        "tools": [tool],
        "instructions": instructions,
    }

    for _ in range(_MAX_TOOL_ROUNDS):
        response = agent.responses.create(request)
        tool_calls = _automation_tool_calls(response)
        if not tool_calls:
            return

        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise RuntimeError("Model tool response is missing a response ID.")
        outputs = [agent.connectors.call(tool_call) for tool_call in tool_calls]
        request = {
            "model": _MODEL,
            "input": outputs,
            "previous_response_id": response_id,
            "stream": True,
            "tools": [tool],
            "instructions": instructions,
        }

    raise RuntimeError("Model exceeded the maximum number of automation tool rounds.")


def _automation_tool_calls(response: JSONObject) -> list[JSONObject]:
    """Return start-automation function calls from a model response."""
    output = response.get("output")
    if not isinstance(output, list):
        return []
    return [
        cast(JSONObject, item)
        for item in output
        if isinstance(item, dict)
        and item.get("type") == "function_call"
        and item.get("name") == "start_automation"
    ]
