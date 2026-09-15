# Build a collaborative research agent

Build an AgentApp that searches and fetches public web sources through multiple
bounded rounds of model-directed tool use. It preserves conversation messages,
recovers from connector failures, and always ends its tool loop.

The finished project uses:

- the OpenAI SDK for model requests
- `agent.connectors.tools` for runtime-provided schemas
- `agent.connectors.call` for function calls
- `agent.events.emit` for frontend-visible model events
- `agent.events.get_trace` for conversation history

It uses only `web_search` and `web_fetch`. Neither requires an external account.

## Create the project

Start from the AgentApp template on Flower Hub:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr new @flwrlabs/agent
$ cd agent
```

You will replace the generated `agent/agent_app.py` while keeping its project
structure:

```text
agent/
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── agent/
    ├── __init__.py
    └── agent_app.py
```

## Configure the project

Keep the generated build-system and Hatch sections. Update the AgentApp-related
parts of `pyproject.toml`:

```{code-block} toml
:substitutions:

[project]
name = "research-agent"
version = "0.1.0"
description = "A bounded public-web research AgentApp"
license = { file = "LICENSE" }
requires-python = ">=3.11,<4.0"
dependencies = ["flwr>=|stable_flwr_version|,<2.0", "openai>=2.16.0,<3.0.0"]

[tool.flwr.app]
publisher = "local"
fab-format-version = 1
flwr-version-target = "|stable_flwr_version|"
fab-include = ["agent/**/*.py", "LICENSE"]

[tool.flwr.app.config.agent]
input = "Find two public sources that explain federated AI and compare them."

[tool.flwr.app.components]
agentapp = "agent.agent_app:app"
```

The configuration pins the runtime contract, includes the SDK, provides a
default input, and tells Flower where to load the `AgentApp` object.

## Implement the AgentApp

Build `agent/agent_app.py` one section at a time. Add the following snippets in
order.

### Define the app and its limits

Every `AgentApp` entry point receives:

- `AgentSession` for connectors and frontend-visible events
- `Context` for run configuration and state shared by the run series

The OpenAI client sends model requests through the runtime URL and credential
injected into the AgentApp process. Keep the model, connector set, and tool-turn
limit near the top of the file. The finite limit prevents an unbounded tool
loop.

```python
from __future__ import annotations

import json
import os
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

MODEL = "openai/gpt-5.6-sol"
TOOL_REFS = ("web_search", "web_fetch")
MAX_TOOL_TURNS = 3

app = AgentApp()
```

### Rebuild conversation input

Each chat message starts a new run. Flower keeps related runs in a run series,
but the model sees only the input passed to `client.responses.create`. To
support follow-up questions, replay the stored user and assistant messages.

Flower stores the user input and AgentApp-published events in the run-series
trace. The trace also contains connector and reasoning activity, so load only
user-message events and completed assistant output:

```python
def message_text(content: Any) -> str:
    """Normalize a stored Responses message to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise TypeError("Message content parts must be objects")
            value = part.get("text", part.get("refusal"))
            if not isinstance(value, str):
                raise TypeError("Message content parts must contain text or refusal")
            parts.append(value)
        return "\n".join(parts)
    raise TypeError("Message content must be text or a list of content parts")


def conversation_messages(agent: AgentSession) -> list[dict[str, Any]]:
    """Rebuild completed user and assistant messages from the event trace."""
    run_order: list[int] = []
    turns_by_run: dict[int, list[dict[str, Any]]] = {}
    assistant_parts_by_run: dict[int, list[str]] = {}

    for entry in agent.events.get_trace():
        run_id = entry.get("run_id")
        event_type = entry.get("event")
        data = entry.get("data")
        if not isinstance(run_id, int) or not isinstance(data, dict):
            continue

        if event_type == "message" and data.get("role") == "user":
            assistant_parts_by_run.pop(run_id, None)
            if run_id not in turns_by_run:
                run_order.append(run_id)
            turns_by_run[run_id] = [
                {
                    "type": "message",
                    "role": "user",
                    "content": message_text(data.get("content")),
                }
            ]
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

    return [message for run_id in run_order for message in turns_by_run[run_id]]
```

`message_text` raises an error for an unexpected shape instead of silently
sending incomplete history to the model. The loader groups each user message
and completed assistant response by run, then flattens those turns in user-event
order. Overlapping runs therefore cannot mix or reorder their output, and a
failed or incomplete response is not replayed as a finished answer.

### Let the model recover from connector failures

A connector can fail after the model requests it, and the model can return
malformed arguments. The next model turn still needs an output for that call
ID. Convert the exception into a `function_call_output` item so the model can
explain the limitation or finish with the evidence it already has:

```python
def connector_error_output(
    tool_call: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    """Return an error item the model can handle in its next turn."""
    return {
        "type": "function_call_output",
        "call_id": tool_call["call_id"],
        "output": json.dumps({"error": str(exc)}),
    }
```

### Orchestrate the tool loop

The main function has five phases:

1. Validate `agent.input` and rebuild the conversation messages from the trace
1. Create the OpenAI client and request the connector tool schemas
1. Execute up to `MAX_TOOL_TURNS` rounds of model-requested function calls
1. Make one final model request without tools and publish its stream
1. Log the completed assistant message

Add the entry point:

```python
@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Research the configured prompt with a bounded connector loop."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    input_items = conversation_messages(agent)

    tools = agent.connectors.tools(TOOL_REFS)
    allowed_tool_names = {
        tool["name"] for tool in tools if isinstance(tool.get("name"), str)
    }

    for _ in range(MAX_TOOL_TURNS):
        response = client.responses.create(
            model=MODEL,
            input=input_items,
            instructions=(
                "Research the user's question using public sources when useful. "
                "Request all independent tool calls for a turn together."
            ),
            tools=tools,
            tool_choice="auto",
        )
        response_output = [item.to_dict() for item in response.output]
        tool_calls = [
            item for item in response_output if item.get("type") == "function_call"
        ]
        if not tool_calls:
            break

        function_outputs = []
        for tool_call in tool_calls:
            if tool_call.get("name") not in allowed_tool_names:
                function_outputs.append(
                    connector_error_output(
                        tool_call,
                        RuntimeError(
                            f"Tool {tool_call.get('name')!r} was not exposed"
                        ),
                    )
                )
                continue
            try:
                arguments = tool_call.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool call arguments must be a JSON object")
                function_outputs.append(agent.connectors.call(tool_call))
            except (RuntimeError, ValueError) as exc:
                function_outputs.append(connector_error_output(tool_call, exc))

        input_items.extend(response_output)
        input_items.extend(function_outputs)

    stream = client.responses.create(
        model=MODEL,
        input=input_items,
        instructions=(
            "Answer the user's question from the available evidence. "
            "Mention any failed source access and do not invent results."
        ),
        stream=True,
    )
    output_text = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed", "response.incomplete"}:
            raise RuntimeError(f"Model response did not complete: {event}")
        if event.type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            output_text.append(event.delta)

    final_text = "".join(output_text)
    print(final_text)
```

The trace already contains the current `agent.input` event when the AgentApp
starts. The planning calls remain local to this run because the app publishes
only the final streamed response. The complete planning output and connector
outputs stay in `input_items` for subsequent tool turns within this run; the
trace loader does not replay them on later runs.

The allowed names come from the returned schemas because one connector
reference can expose several tools. The final request omits `tools`, which
forces an answer instead of another connector round. The stream collects both
answer and refusal text, then publishes that result. If the stream is
incomplete, the app raises an error and the trace loader discards its partial
text on the next run.

```{note}
Connector calls still record their outputs and activity for run inspection.
The trace loader ignores those event types, so connector activity and function
outputs are not treated as conversation messages.
```

### Copy the complete file

If you prefer to start from the finished version, expand the block below and
copy it into `agent/agent_app.py`.

```{raw} html
<details>
<summary><strong>Complete <code>agent/agent_app.py</code></strong></summary>
```

```python
from __future__ import annotations

import json
import os
from typing import Any

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

MODEL = "openai/gpt-5.6-sol"
TOOL_REFS = ("web_search", "web_fetch")
MAX_TOOL_TURNS = 3

app = AgentApp()


def message_text(content: Any) -> str:
    """Normalize a stored Responses message to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise TypeError("Message content parts must be objects")
            value = part.get("text", part.get("refusal"))
            if not isinstance(value, str):
                raise TypeError("Message content parts must contain text or refusal")
            parts.append(value)
        return "\n".join(parts)
    raise TypeError("Message content must be text or a list of content parts")


def conversation_messages(agent: AgentSession) -> list[dict[str, Any]]:
    """Rebuild completed user and assistant messages from the event trace."""
    run_order: list[int] = []
    turns_by_run: dict[int, list[dict[str, Any]]] = {}
    assistant_parts_by_run: dict[int, list[str]] = {}

    for entry in agent.events.get_trace():
        run_id = entry.get("run_id")
        event_type = entry.get("event")
        data = entry.get("data")
        if not isinstance(run_id, int) or not isinstance(data, dict):
            continue

        if event_type == "message" and data.get("role") == "user":
            assistant_parts_by_run.pop(run_id, None)
            if run_id not in turns_by_run:
                run_order.append(run_id)
            turns_by_run[run_id] = [
                {
                    "type": "message",
                    "role": "user",
                    "content": message_text(data.get("content")),
                }
            ]
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

    return [message for run_id in run_order for message in turns_by_run[run_id]]


def connector_error_output(
    tool_call: dict[str, Any], exc: Exception
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

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    input_items = conversation_messages(agent)

    tools = agent.connectors.tools(TOOL_REFS)
    allowed_tool_names = {
        tool["name"] for tool in tools if isinstance(tool.get("name"), str)
    }

    for _ in range(MAX_TOOL_TURNS):
        response = client.responses.create(
            model=MODEL,
            input=input_items,
            instructions=(
                "Research the user's question using public sources when useful. "
                "Request all independent tool calls for a turn together."
            ),
            tools=tools,
            tool_choice="auto",
        )
        response_output = [item.to_dict() for item in response.output]
        tool_calls = [
            item for item in response_output if item.get("type") == "function_call"
        ]
        if not tool_calls:
            break

        function_outputs = []
        for tool_call in tool_calls:
            if tool_call.get("name") not in allowed_tool_names:
                function_outputs.append(
                    connector_error_output(
                        tool_call,
                        RuntimeError(
                            f"Tool {tool_call.get('name')!r} was not exposed"
                        ),
                    )
                )
                continue
            try:
                arguments = tool_call.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool call arguments must be a JSON object")
                function_outputs.append(agent.connectors.call(tool_call))
            except (RuntimeError, ValueError) as exc:
                function_outputs.append(connector_error_output(tool_call, exc))

        input_items.extend(response_output)
        input_items.extend(function_outputs)

    stream = client.responses.create(
        model=MODEL,
        input=input_items,
        instructions=(
            "Answer the user's question from the available evidence. "
            "Mention any failed source access and do not invent results."
        ),
        stream=True,
    )
    output_text = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed", "response.incomplete"}:
            raise RuntimeError(f"Model response did not complete: {event}")
        if event.type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            output_text.append(event.delta)

    final_text = "".join(output_text)
    print(final_text)
```

```{raw} html
</details>
```

## Build and run

```console
$ uv sync
$ uv run flwr build
$ uv run flwr login supergrid
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
see zero or more search or fetch calls and any connector failure that the final
answer had to handle.
```

## Adapt it safely

- Keep `TOOL_REFS` limited to the capabilities the task needs
- Keep a finite tool-turn limit even when you change models
- Validate every required run-config value before making a model call
- Never put credentials in prompts or connector arguments
- Use [Connect accounts](../how-to-guides/connect-accounts.md) before adding an
  account connector, and remember that those runs are personal-workspace-only
- Follow [Create automations](../how-to-guides/create-automations.md) before
  exposing `start_automation` for explicit future or recurring requests
