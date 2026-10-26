"""A minimal Flower AgentApp."""

from __future__ import annotations

import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

from .utils import (
    MODEL,
    compact_active_context_if_needed,
    construct_model_input,
    get_active_context,
)

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Send the conversation to the model."""
    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )

    active_context = get_active_context(agent, context.run_id)
    compact_active_context_if_needed(agent, client, active_context)
    agent.events.emit(
        {"type": "active_context.created", "token_count": active_context.token_count}
    )
    input_items = construct_model_input(active_context, agent.prompt)

    stream = client.responses.create(
        model=MODEL,
        input=input_items,
        stream=True,
    )

    output_text = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type == "response.output_text.delta":
            output_text.append(event.delta)

    print("".join(output_text))
