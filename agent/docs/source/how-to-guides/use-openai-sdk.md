# Use the OpenAI SDK in an AgentApp

The OpenAI Python SDK is the standard way to make model requests from a Flower
AgentApp. Flower provides an OpenAI-compatible Responses endpoint and its
credentials while the AgentApp is running, so your code can use the SDK without
a model-provider API key.

This guide targets Flower {{ stable_flwr_version }}.

## Prepare an AgentApp project

Start with an AgentApp project from [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md). Its template already
includes compatible Flower and OpenAI SDK dependencies:

```{code-block} toml
:substitutions:

dependencies = ["flwr>=|stable_flwr_version|,<2.0", "openai>=2.16.0,<3.0.0"]
```

For an existing AgentApp, set its Flower target in `pyproject.toml`:

```{code-block} toml
:substitutions:

[tool.flwr.app]
flwr-version-target = "|stable_flwr_version|"
```

Then update its dependencies:

```{code-block} console
:substitutions:

$ uv add 'flwr>=|stable_flwr_version|,<2.0' 'openai>=2.16.0,<3.0.0'
```

Do not add a model-provider API key to the project or its configuration.

## Create the client inside the AgentApp

Flower starts the AgentApp process with two environment variables:

- `FLWR_RUNTIME_BASE_URL` is the base URL of its internal Runtime API
- `FLWR_RUNTIME_API_KEY` authenticates requests from that AgentApp process

Pass both values to `OpenAI` without modifying them. The SDK adds the
`/responses` path when it creates a response.

Create the client inside the main function so commands such as `flwr build` can
import the module without requiring a running Flower runtime:

```python
client = OpenAI(
    base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
    api_key=os.environ["FLWR_RUNTIME_API_KEY"],
    max_retries=0,
)
```

Set `max_retries=0` because each Responses request creates a Flower model task.
An automatic SDK retry could create the task more than once.

```{important}
`FLWR_RUNTIME_BASE_URL` is not `FLWR_MODEL_API_ENDPOINT`. The first is available
only inside a running AgentApp. The second configures the upstream model
provider for a self-hosted SuperLink and belongs outside AgentApp code.
```

## Stream the response to Flower clients

The runtime keeps model-task events private until the AgentApp chooses to
publish them. Iterate over the SDK stream and pass each event to
`agent.events.emit` so Flower Chat and the browser can render the response:

```python
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

MODEL = "openai/gpt-5.6-sol"

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Send the chat prompt to the model."""
    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    stream = client.responses.create(
        model=MODEL,
        input=agent.prompt,
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
```

`event.to_dict()` converts the typed SDK event into the JSON object expected by
Flower. The app also collects text deltas so the completed answer appears in
its logs.

(publish-agentapp-generated-text)=

## Publish AgentApp-generated text

`print(...)` writes to the AgentApp logs. It does not publish an assistant
response to Flower Chat.

When the AgentApp already has final user-facing text that did not come from an
SDK stream, publish an output delta followed by a completion event:

```python
assistant_text = "Hello from the AgentApp!"
agent.events.emit(
    {
        "type": "response.output_text.delta",
        "delta": assistant_text,
    }
)
agent.events.emit({"type": "response.completed"})
```

The output delta adds assistant text to the run-event stream. The completion
event tells Flower Chat and other run-event clients that the response has
finished. For model-generated output, prefer republishing the original SDK
events so clients receive the complete response event sequence.

`agent.events.emit(...)` publishes structured events; only event types that
run-event clients recognize as response output are rendered as assistant text.

Publishing these events makes them available to later runs through
`agent.events.get_trace()`; it does not add them to `Context`.

## Use the SDK with connectors

Model requests use `client.responses.create`. Connector discovery and
execution remain on the `AgentSession`:

- `agent.connectors.tools(...)` returns tool schemas to pass to the SDK
- `agent.connectors.call(...)` executes a model-requested function call
- `agent.events.emit(...)` publishes events to the run-event stream
- `agent.events.get_trace()` reads events from every run in the current series

The SDK returns typed output items. Convert a function-call item with
`item.to_dict()` before passing it to `agent.connectors.call`. See [Build a
research agent](../tutorials/build-a-research-agent.md) for a complete bounded tool
loop.

## Read conversation history

Before calling the AgentApp, Flower records `agent.prompt` as a user-message
event. Events published with `agent.events.emit(...)` and connector
activity are stored in the same run-series trace. Read that trace at the start of
a later run:

```python
trace = agent.events.get_trace()
for entry in trace:
    event_type = entry["event"]
    event_data = entry["data"]
```

Each entry also includes `id`, `timestamp`, `run_id`, and `task_id`. Filter out
connector, reasoning, failed, and incomplete events before constructing the
next model input. See [Build a research agent](../tutorials/build-a-research-agent.md) for a complete loader that
rebuilds user and assistant messages from the trace.

Use `Context` only when the AgentApp needs additional app-defined state beyond
the recorded event trace.

```{tip}
To test your AgentApp, see {ref}`load-an-agentapp-in-flower-chat`.
```

The runtime injects both environment variables when it starts the AgentApp. Do
not set, log, or persist `FLWR_RUNTIME_API_KEY` yourself.

## Troubleshoot the SDK client

- **`ModuleNotFoundError: openai`**: run `uv sync` or add the SDK dependency
- **Missing runtime URL or key**: run the app through Flower instead of starting
  the Python module directly
- **Authentication failure**: start a new run and use its injected credentials
- **Unsupported request field**: compare the request with the supported model
  fields in [The AgentApp runtime](../explanations/agentapp-runtime.md)

The endpoint is scoped to the running AgentApp. It is not a public API for
browsers, external services, or independently launched SDK clients.
