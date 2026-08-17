# Build a collaborative research agent

Build an AgentApp that can search and fetch public web sources, execute several
model-requested function calls, preserve conversation messages, recover from a
connector failure, and always end its tool loop.

The finished project uses the complete public `AgentSession` surface:

- `agent.responses.create` for model requests;
- `agent.connectors.tools` for runtime-provided schemas; and
- `agent.connectors.call` for function calls.

It uses only `web_search` and `web_fetch`. Neither requires an external account.

## Create the project

```console
$ mkdir research-agent
$ cd research-agent
$ mkdir research_agent
$ touch research_agent/__init__.py
```

Create:

```text
research-agent/
├── .gitignore
├── pyproject.toml
└── research_agent/
    ├── __init__.py
    └── agent_app.py
```

Add `.gitignore`:

```text
.venv/
*.fab
__pycache__/
```

## Configure the project

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "research-agent"
version = "0.1.0"
description = "A bounded public-web research AgentApp"
license = "Apache-2.0"
requires-python = ">=3.11"
dependencies = ["flwr==1.34.0"]

[tool.hatch.build.targets.wheel]
packages = ["research_agent"]

[tool.flwr.app]
publisher = "local"
display-name = "Research Agent"
flwr-version-target = "1.34.0"
fab-include = ["research_agent/**/*.py"]

[tool.flwr.app.config.agent]
input = "Find two public sources that explain federated AI and compare them."

[tool.flwr.app.components]
agentapp = "research_agent.agent_app:app"
```

## Implement the AgentApp

Create `research_agent/agent_app.py`:

```python
from __future__ import annotations

import json
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

MODEL = "openai/gpt-5.6-sol"
TOOL_REFS = ("web_search", "web_fetch")
MAX_TOOL_TURNS = 3

app = AgentApp()


def message_text(content: Any) -> str:
    """Normalize a stored Responses message to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    raise TypeError("Message content must be text or a list of text parts")


def conversation_messages(context: Context) -> list[dict[str, Any]]:
    """Replay only user and assistant messages from the run series."""
    messages: list[dict[str, Any]] = []
    items = context.state.config_records.get("items", {}).get("json", [])
    for item_json in items:
        item = json.loads(item_json)
        if item.get("type") != "message":
            continue
        messages.append(
            {
                "type": "message",
                "role": item["role"],
                "content": message_text(item["content"]),
            }
        )
    return messages


def private_response(
    agent: AgentSession,
    context: Context,
    request: dict[str, Any],
) -> dict[str, Any]:
    """Make a planning request without retaining its draft model output."""
    had_items = "items" in context.state
    previous_items = list(context.state["items"].get("json", ())) if had_items else None
    try:
        return agent.responses.create(request)
    finally:
        if had_items:
            context.state["items"]["json"] = previous_items
        elif "items" in context.state:
            del context.state["items"]


def connector_error_output(
    tool_call: dict[str, Any], exc: RuntimeError
) -> dict[str, Any]:
    """Return an error item the model can handle in its next turn."""
    return {
        "type": "function_call_output",
        "call_id": tool_call["call_id"],
        "output": json.dumps({"error": str(exc)}),
    }


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Research the configured prompt with a bounded connector loop."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    input_items = conversation_messages(context)
    if not any(
        item["role"] == "user" and item["content"].strip() == prompt.strip()
        for item in input_items
    ):
        input_items.append(
            {"type": "message", "role": "user", "content": prompt.strip()}
        )

    tools = agent.connectors.tools(TOOL_REFS)

    for _ in range(MAX_TOOL_TURNS):
        response = private_response(
            agent,
            context,
            {
                "model": MODEL,
                "input": input_items,
                "instructions": (
                    "Research the user's question using public sources when useful. "
                    "Request all independent tool calls for a turn together."
                ),
                "tools": tools,
                "tool_choice": "auto",
                "stream": False,
            },
        )
        tool_calls = [
            dict(item)
            for item in response.get("output", [])
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        if not tool_calls:
            break

        function_outputs = []
        for tool_call in tool_calls:
            try:
                function_outputs.append(agent.connectors.call(tool_call))
            except RuntimeError as exc:
                function_outputs.append(connector_error_output(tool_call, exc))

        input_items.extend(tool_calls)
        input_items.extend(function_outputs)

    agent.responses.create(
        {
            "model": MODEL,
            "input": input_items,
            "instructions": (
                "Answer the user's question from the available evidence. "
                "Mention any failed source access and do not invent results."
            ),
            "stream": True,
        }
    )
```

## Follow the control flow

The runtime records the current `agent.input` as a user message before calling
the app. `conversation_messages` loads user and assistant messages from the
series so follow-up runs can use them.

Each tool turn then:

1. obtains the registered `web_search` and `web_fetch` schemas;
1. lets the model request zero, one, or several function calls;
1. executes every requested call;
1. turns a connector failure into model-readable output; and
1. adds calls and outputs to the next request.

The planning response is private because its draft output is removed from
conversation state. The final request has no tools and streams one clean answer.
`MAX_TOOL_TURNS` prevents an untrusted model decision from creating an unbounded
loop.

```{note}
The connector activity itself is still recorded for run inspection. The app
replays only message items on the next run, so connector events and orphaned
function outputs are not treated as conversation messages.
```

## Build and run

```console
$ uv sync
$ uv run flwr build
$ uvx --from flwr==1.34.0 flwr login supergrid
$ uv run flwr run . supergrid --stream
```

Override the research prompt:

```console
$ uv run flwr run . supergrid \
    --run-config 'agent.input="Compare two recent public explanations of federated AI."' \
    --stream
```

```{admonition} Success checkpoint
:class: tip

The run finishes with one streamed answer. In SuperGrid run activity, you can
see zero or more search/fetch calls and any connector failure that the final
answer had to handle.
```

## Adapt it safely

- Keep `TOOL_REFS` limited to the capabilities the task needs.
- Keep a finite tool-turn limit even when you change models.
- Validate every required run-config value before making a model call.
- Never put credentials in prompts or connector arguments.
- Use [Connect accounts](../how-to-guides/connect-accounts.md) before adding an
  account connector, and remember that those runs are personal-workspace-only.
- Expose `start_automation` only when the app must honor explicit future or
  recurring requests; see [Create automations](../how-to-guides/create-automations.md).
