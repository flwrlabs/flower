"""Load, compact, and format the active conversation context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from flwr.agentapp import RUN_SUMMARY_EVENT_TYPES, AgentSession
from openai import OpenAI

MODEL = "openai/gpt-6-luna"
COMPACTION_TOKEN_THRESHOLD = 800_000


def _filter_event_fields(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep conversation content without response metadata or encrypted reasoning."""
    filtered_events = []
    for event in events:
        data = event["data"]
        filtered = {"id": event["id"], "event": event["event"]}
        if event["event"] == "message":
            filtered["data"] = {
                key: data[key] for key in ("role", "content") if key in data
            }
        elif event["event"] in {"response.completed", "response.incomplete"}:
            response = data["response"]
            output = []
            for item in response.get("output", []):
                if item.get("type") == "reasoning":
                    summaries = [
                        part["text"]
                        for part in item.get("summary", [])
                        if part.get("text")
                    ]
                    if summaries:
                        output.append({"type": "reasoning", "summary": summaries})
                elif item.get("type") == "message":
                    content = [
                        part["text"]
                        for part in item.get("content", [])
                        if part.get("text")
                    ]
                    if content:
                        output.append(
                            {
                                "type": "message",
                                "role": item.get("role"),
                                "content": content,
                            }
                        )
            filtered["data"] = {"output": output}
            if event["event"] == "response.incomplete":
                filtered["data"]["incomplete_details"] = response.get(
                    "incomplete_details"
                )
        else:
            filtered["data"] = {
                key: data[key]
                for key in ("call_id", "name", "arguments", "output")
                if key in data
            }
        filtered_events.append(filtered)
    return filtered_events


def _model_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the event boundary IDs before sending history to a model."""
    return [{"type": event["event"], **event["data"]} for event in events]


@dataclass
class ActiveContext:
    """Conversation data and the event boundary selected for compaction."""

    summary: str
    events: list[dict[str, Any]]
    current_index: int
    token_count: int

    def to_model_input(self) -> dict[str, Any]:
        """Return the conversation fields sent to the model."""
        return {"summary": self.summary, "events": _model_events(self.events)}


def get_active_context(agent: AgentSession, current_run_id: int) -> ActiveContext:
    """Get the active context from the saved summary and subsequent events."""
    contexts = agent.events.get_compacted_contexts()
    latest = contexts[0] if contexts else None
    summary = cast(str, latest["text"]) if latest else ""
    # Resume after the saved boundary so summarized events are not replayed.
    trace = agent.events.get_trace(
        event_types=RUN_SUMMARY_EVENT_TYPES,
        after_event_id=cast(int, latest["through_event_id"]) if latest else None,
    )
    current_index = next(
        (i for i, event in enumerate(trace) if event["run_id"] == current_run_id),
        len(trace),
    )
    token_count = cast(int, latest["token_count"]) if latest else 0
    for event in trace[:current_index]:
        if event["event"] in {"response.completed", "response.incomplete"}:
            usage = event["data"]["response"].get("usage") or {}
            token_count += usage.get("output_tokens") or 0
    return ActiveContext(
        summary,
        _filter_event_fields(trace[:current_index]),
        current_index,
        token_count,
    )


def _needs_compaction(active_context: ActiveContext) -> bool:
    """Check whether the selected events reached the compaction threshold."""
    return (
        active_context.current_index > 0
        and active_context.token_count >= COMPACTION_TOKEN_THRESHOLD
    )


def _save_summary(
    agent: AgentSession,
    client: OpenAI,
    traces: list[dict[str, Any]],
    previous_summary: str,
) -> tuple[str, int]:
    """Summarize selected events and store the result."""
    # The boundary covers every event included in this summary.
    through_event_id = int(traces[-1]["id"])
    agent.events.emit(
        {
            "type": "compacted_context.summary.started",
            "through_event_id": through_event_id,
        }
    )
    response = client.responses.create(
        model=MODEL,
        instructions=(
            "Update the previous summary with the new events for a future "
            "assistant. Aim for at most 2,500 words. Preserve current goals, "
            "user preferences, constraints, decisions, and unfinished work. For "
            "completed topics, retain what was discussed and the key outcomes, "
            "but shorten details unlikely to matter in a later follow-up. If "
            "space is tight, remove repetition and transient process details "
            "first, then less relevant examples and specifics from older "
            "completed topics. Keep important older facts when they remain "
            "relevant. When web research informed an answer, keep exact source "
            "titles and URLs only for the current task, unresolved questions, "
            "or claims likely to be revisited. For completed topics no longer "
            "relevant, retain useful findings without their reference lists. "
            "Omit unused searches and results. Incorporate important new facts, "
            "correct superseded facts, and do not invent anything. Return only "
            "the summary text."
        ),
        input=json.dumps(
            {"previous_summary": previous_summary, "events": _model_events(traces)},
            ensure_ascii=False,
        ),
    )
    summary = response.output_text.strip()
    # Keep the old boundary if summary generation failed.
    if response.status != "completed" or not summary or response.usage is None:
        raise RuntimeError("Summary generation failed")
    token_count = response.usage.output_tokens
    agent.events.put_compacted_context(
        through_event_id=through_event_id, text=summary, token_count=token_count
    )
    agent.events.emit(
        {
            "type": "compacted_context.summary.completed",
            "through_event_id": through_event_id,
        }
    )
    return summary, token_count


def compact_active_context_if_needed(
    agent: AgentSession, client: OpenAI, active_context: ActiveContext
) -> None:
    """Summarize selected events when the active context reaches the token limit."""
    if not _needs_compaction(active_context):
        return
    active_context.summary, active_context.token_count = _save_summary(
        agent,
        client,
        active_context.events[: active_context.current_index],
        active_context.summary,
    )
    active_context.events = active_context.events[active_context.current_index :]
    active_context.current_index = 0


def construct_model_input(
    active_context: ActiveContext, prompt: str
) -> list[dict[str, Any]]:
    """Construct model input from the active context and current prompt."""
    return [
        {
            "role": "user",
            "content": json.dumps(
                {
                    "active_context": active_context.to_model_input(),
                    "current_prompt": prompt,
                },
                ensure_ascii=False,
            ),
        }
    ]
