"""A minimal Flower AgentApp."""

import json
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import ConfigRecord, Context
from openai import OpenAI

MODEL = "openai/gpt-5.6-sol"

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Send the configured input to the model."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    stream = client.responses.create(
        model=MODEL,
        input=prompt.strip(),
        stream=True,
    )

    output_text = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type == "response.output_text.delta":
            output_text.append(event.delta)

    final_text = "".join(output_text)
    message = {"type": "message", "role": "assistant", "content": final_text}
    with context.locked():
        items_record = context.state.config_records.setdefault(
            "items", ConfigRecord({"json": []})
        )
        items = items_record.get("json")
        if not isinstance(items, list):
            raise TypeError("Context items must be a list")
        items.append(json.dumps(message))
    print(final_text)
