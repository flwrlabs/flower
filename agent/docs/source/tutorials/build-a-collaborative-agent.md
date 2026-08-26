# Build a collaborative research agent

Build an AgentApp that searches and fetches public web sources through multiple
bounded rounds of model-directed tool use. It preserves conversation messages,
recovers from connector failures, and always ends its tool loop.

The finished project uses:

- the OpenAI SDK for model requests
- `agent.connectors.tools` for runtime-provided schemas
- `agent.connectors.call` for function calls
- `agent.events.emit` for frontend-visible model events
- Flower `Context` for conversation state

It uses only `web_search` and `web_fetch`. Neither requires an external account.

## Create the project

Start from the AgentApp template on Flower Hub:

```console
$ uvx --from flwr==1.35.0 flwr new @flwrlabs/agent
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

```toml
[project]
name = "research-agent"
version = "0.1.0"
description = "A bounded public-web research AgentApp"
license = { file = "LICENSE" }
requires-python = ">=3.11,<4.0"
dependencies = ["flwr>=1.35.0,<2.0", "openai>=2.16.0,<3.0.0"]

[tool.flwr.app]
publisher = "local"
fab-format-version = 1
flwr-version-target = "1.35.0"
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
from flwr.app import ConfigRecord, Context
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

Conversation items live in a `ConfigRecord` named `items`. The state also
contains tool activity, so load only message items and normalize their content
to plain text:

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


def conversation_messages(context: Context) -> list[dict[str, Any]]:
    """Replay only user and assistant messages from the run series."""
    messages: list[dict[str, Any]] = []
    items_record = context.state.config_records.get("items")
    items = items_record.get("json", []) if items_record is not None else []
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
```

`message_text` raises an error for an unexpected shape instead of silently
sending incomplete history to the model.

### Persist the final answer

The runtime records a non-empty `agent.input` as a user message before calling
the app. SDK responses are not automatically added to `Context`, which keeps
private planning turns out of the conversation by default.

Store only the final assistant message after its stream completes:

```python
def append_assistant_message(context: Context, text: str) -> None:
    """Persist the final assistant message for the next run in the series."""
    message = {"type": "message", "role": "assistant", "content": text}
    with context.locked():
        items_record = context.state.config_records.setdefault(
            "items", ConfigRecord({"json": []})
        )
        items = items_record.get("json")
        if not isinstance(items, list):
            raise TypeError("Context items must be a list")
        items.append(json.dumps(message))
```

Locking the context keeps the update atomic if the app later introduces
parallel work.

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

1. Validate `agent.input` and rebuild the conversation messages
1. Create the OpenAI client and request the connector tool schemas
1. Execute up to `MAX_TOOL_TURNS` rounds of model-requested function calls
1. Make one final model request without tools and publish its stream
1. Persist and log the completed assistant message

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
    input_items = conversation_messages(context)
    if not any(
        item["role"] == "user" and item["content"].strip() == prompt.strip()
        for item in input_items
    ):
        input_items.append(
            {"type": "message", "role": "user", "content": prompt.strip()}
        )

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
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            output_text.append(event.delta)

    final_text = "".join(output_text)
    append_assistant_message(context, final_text)
    print(final_text)
```

The duplicate check accounts for the user message Flower already stored. The
planning calls remain local to this run because SDK responses are not appended
to `Context`. The app keeps each complete model output next to its connector
outputs so later turns retain function-call context.

The allowed names come from the returned schemas because one connector
reference can expose several tools. The final request omits `tools`, which
forces an answer instead of another connector round. The stream collects both
answer and refusal text, then publishes and persists that result.

```{note}
Connector calls still record their outputs and activity for run inspection.
The app replays only message items on the next run, so connector activity and
function outputs are not treated as conversation messages.
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
from flwr.app import ConfigRecord, Context
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


def conversation_messages(context: Context) -> list[dict[str, Any]]:
    """Replay only user and assistant messages from the run series."""
    messages: list[dict[str, Any]] = []
    items_record = context.state.config_records.get("items")
    items = items_record.get("json", []) if items_record is not None else []
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


def append_assistant_message(context: Context, text: str) -> None:
    """Persist the final assistant message for the next run in the series."""
    message = {"type": "message", "role": "assistant", "content": text}
    with context.locked():
        items_record = context.state.config_records.setdefault(
            "items", ConfigRecord({"json": []})
        )
        items = items_record.get("json")
        if not isinstance(items, list):
            raise TypeError("Context items must be a list")
        items.append(json.dumps(message))


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
    input_items = conversation_messages(context)
    if not any(
        item["role"] == "user" and item["content"].strip() == prompt.strip()
        for item in input_items
    ):
        input_items.append(
            {"type": "message", "role": "user", "content": prompt.strip()}
        )

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
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type in {
            "response.output_text.delta",
            "response.refusal.delta",
        }:
            output_text.append(event.delta)

    final_text = "".join(output_text)
    append_assistant_message(context, final_text)
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
