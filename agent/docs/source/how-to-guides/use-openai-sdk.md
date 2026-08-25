# Use the OpenAI SDK in an AgentApp

The OpenAI Python SDK is the standard way to make model requests from a Flower
AgentApp. Flower provides an OpenAI-compatible Responses endpoint and its
credentials while the AgentApp is running, so your code can use the SDK without
a model-provider API key.

This guide targets Flower 1.35.0.

## Start from the AgentApp template

Create a project from the AgentApp published on Flower Hub:

```console
$ uvx --from flwr==1.35.0 flwr new @flwrlabs/agent
$ cd agent
$ uv sync
```

The template already includes compatible Flower and OpenAI SDK dependencies:

```toml
dependencies = ["flwr>=1.35.0,<2.0", "openai>=2.16.0,<3.0.0"]
```

For an existing AgentApp, set its Flower target in `pyproject.toml`:

```toml
[tool.flwr.app]
flwr-version-target = "1.35.0"
```

Then update its dependencies:

```console
$ uv add 'flwr>=1.35.0,<2.0' 'openai>=2.16.0,<3.0.0'
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
)
```

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
    """Send the configured input to the model."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
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

    print("".join(output_text))
```

`event.to_dict()` converts the typed SDK event into the JSON object expected by
Flower. The app also collects text deltas so the completed answer appears in
its logs.

## Use the SDK with connectors

Model requests use `client.responses.create`. Connector discovery and
execution remain on the `AgentSession`:

- `agent.connectors.tools(...)` returns tool schemas to pass to the SDK
- `agent.connectors.call(...)` executes a model-requested function call
- `agent.events.emit(...)` publishes the model events selected by the app

The SDK returns typed output items. Convert a function-call item with
`item.to_dict()` before passing it to `agent.connectors.call`. See [Build a
collaborative research
agent](../tutorials/build-a-collaborative-agent.md) for a complete bounded tool
loop.

## Persist only the state you need

The runtime records a non-empty `agent.input` as a user message before calling
the AgentApp. Responses created through the SDK are not automatically appended
to the Flower `Context`. Store the final assistant message yourself when later
runs in the same series need to replay it.

Publishing an event with `agent.events.emit` makes it visible to run-event
clients but does not persist it in the conversation state. The collaborative
research agent tutorial shows both streaming and explicit message persistence.

## Build and run the AgentApp

```console
$ uv run flwr build
$ uv run flwr login supergrid
$ uv run flwr run . supergrid --stream
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
