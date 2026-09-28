"""A minimal Flower AgentApp."""

from __future__ import annotations

import os
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

MODEL = "dedicated/flowerai/Kimi-K2.7-Code-1OUHWL"

app = AgentApp()


def _message_text(content: Any) -> str:
    """Extract text from a trace message."""
    if isinstance(content, str):
        return content
    return "\n".join(part["text"] for part in content if "text" in part)


def _conversation(agent: AgentSession, context: Context) -> list[dict[str, str]]:
    """Rebuild user and assistant messages from this run series."""
    messages: list[dict[str, str]] = []
    current_prompt_seen = False
    for event in agent.events.get_trace():
        data = event["data"]
        if data.get("type") == "message" and data.get("role") in {
            "user",
            "assistant",
        }:
            text = _message_text(data["content"])
            messages.append({"type": "message", "role": data["role"], "content": text})
            current_prompt_seen |= (
                data["role"] == "user"
                and event.get("run_id") == context.run_id
                and text.strip() == agent.prompt.strip()
            )
        elif data.get("type") == "response.completed":
            for item in data["response"]["output"]:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    messages.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": _message_text(item["content"]),
                        }
                    )
    if not current_prompt_seen:
        messages.append(
            {"type": "message", "role": "user", "content": agent.prompt.strip()}
        )
    return messages


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Send the conversation to the model."""
    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    stream = client.responses.create(
        model=MODEL,
        input=_conversation(agent, context),
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
