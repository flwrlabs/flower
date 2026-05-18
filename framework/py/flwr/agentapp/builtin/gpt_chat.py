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
"""Built-in pass-through GPT chat AgentApp."""


from __future__ import annotations

from typing import cast

from flwr.agentapp import AgentApp, AgentModelResponse, AgentSession
from flwr.supercore.task_message import JsonObject, JsonValue

GPT_CHAT_STARTED_EVENT = "agent.gpt_chat.started"
GPT_CHAT_COMPLETED_EVENT = "agent.gpt_chat.completed"

app = AgentApp()

__all__ = ["GPT_CHAT_COMPLETED_EVENT", "GPT_CHAT_STARTED_EVENT", "app"]


@app.main()
def main(session: AgentSession) -> None:
    """Run one pass-through GPT chat turn."""
    model = session.model.default_model
    model_input = _normalize_model_input(session.input)

    session.emit_event(GPT_CHAT_STARTED_EVENT, {"model": model})
    result = session.model.response(
        input=cast(JsonValue, model_input),
        model=model,
        stream=True,
    )
    session.conversation.add_items([_assistant_item(result, model)])
    session.emit_event(
        GPT_CHAT_COMPLETED_EVENT,
        {
            "model": model,
            **(
                {"response_id": result.response_id}
                if result.response_id is not None
                else {}
            ),
        },
    )


def _normalize_model_input(value: JsonValue) -> list[JsonObject]:
    """Return AgentApp input as a list of Responses-compatible input items."""
    if isinstance(value, dict):
        return [cast(JsonObject, value)]
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return [cast(JsonObject, item) for item in value]
    raise ValueError("gpt-chat input must be a JSON object or list of JSON objects.")


def _assistant_item(result: AgentModelResponse, model: str) -> JsonObject:
    """Create one conversation item from a model response."""
    item: JsonObject = {
        "role": "assistant",
        "content": result.output_text,
        "response_id": result.response_id,
        "model": model,
    }
    if result.output_text == "":
        item["response"] = result.response
    return item
