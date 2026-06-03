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
"""Tests for the built-in GPT chat AgentApp."""


from flwr.agentapp import AgentResponses, AgentSession
from flwr.app import Context, RecordDict
from flwr.supercore.typing import JSONObject

from .gpt_chat import app


class _Responses(AgentResponses):
    """Capturing AgentResponses test double."""

    def __init__(self) -> None:
        self.requests: list[JSONObject] = []

    def create(self, request: JSONObject) -> JSONObject:
        """Capture a model request."""
        self.requests.append(request)
        return {"object": "response", "status": "completed", "output": []}


class _Session(AgentSession):
    """AgentSession test double."""

    def __init__(self, responses: AgentResponses) -> None:
        self._responses = responses

    @property
    def responses(self) -> AgentResponses:
        """Return capturing responses."""
        return self._responses


def test_gpt_chat_forwards_agent_input_with_streaming_enabled() -> None:
    """The built-in GPT chat agent should be a streaming pass-through."""
    responses = _Responses()
    session = _Session(responses)
    context = Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config={"agent.input": "Hello"},
    )

    app(session, context)

    assert responses.requests == [
        {
            "model": "openai/gpt-5.5",
            "input": "Hello",
            "stream": True,
        }
    ]
