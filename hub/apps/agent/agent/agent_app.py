"""A minimal Flower AgentApp."""

from __future__ import annotations

import os
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

MODEL = "openai/gpt-5.6-sol"

app = AgentApp()


def _message_text(content: Any) -> str:
    """Extract text from a trace message."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        text = part.get("text", part.get("refusal"))
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _conversation(agent: AgentSession, context: Context) -> list[dict[str, str]]:
    """Rebuild user and assistant messages from this run series."""
    run_order: list[int] = []
    turns_by_run: dict[int, list[dict[str, str]]] = {}
    assistant_parts_by_run: dict[int, list[str]] = {}
    current_prompt_seen = False
    for event in agent.events.get_trace():
        run_id = event.get("run_id")
        event_type = event.get("event")
        data = event["data"]
        if not isinstance(run_id, int):
            continue

        if event_type == "message" and data.get("role") == "user":
            assistant_parts_by_run.pop(run_id, None)
            text = _message_text(data["content"])
            if run_id not in turns_by_run:
                run_order.append(run_id)
            turns_by_run[run_id] = [
                {"type": "message", "role": "user", "content": text}
            ]
            current_prompt_seen |= (
                run_id == context.run_id and text.strip() == agent.prompt.strip()
            )
        elif event_type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            delta = data.get("delta")
            if isinstance(delta, str):
                assistant_parts_by_run.setdefault(run_id, []).append(delta)
        elif event_type == "response.completed":
            assistant_parts = assistant_parts_by_run.pop(run_id, [])
            turn = turns_by_run.get(run_id)
            if assistant_parts and turn is not None:
                turn.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": "".join(assistant_parts),
                    }
                )
        elif event_type in {"error", "response.failed", "response.incomplete"}:
            assistant_parts_by_run.pop(run_id, None)

    messages = [message for run_id in run_order for message in turns_by_run[run_id]]
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
