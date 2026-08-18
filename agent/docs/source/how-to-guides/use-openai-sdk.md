# Use the OpenAI SDK in an AgentApp

Use the OpenAI Python SDK when you want its typed Responses API in an AgentApp
or need to adapt code that already uses it. Flower provides an internal Open
Responses-compatible endpoint and its credentials while the AgentApp is
running.

Start with a working project from [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md). This guide targets Flower
1.34.0.

## Add the SDK

From the project directory:

```console
$ uv add openai
```

This adds the SDK to the project dependencies and updates the lock file. Do not
add a model-provider API key to the project or its configuration.

The example below follows the [OpenAI Python SDK Responses API
pattern](https://developers.openai.com/api/docs/quickstart), with Flower's
runtime URL and credential supplied explicitly.

## Create the client inside the AgentApp

Flower starts the AgentApp process with two environment variables:

- `FLWR_RUNTIME_BASE_URL` is the base URL of the internal Runtime API
- `FLWR_RUNTIME_API_KEY` authenticates requests from the AgentApp process

Pass both values to `OpenAI` without modifying them. The SDK adds the
`/responses` path when it creates a response.

Create the client inside the main function so commands such as `flwr build` can
import the module without requiring a running Flower runtime:

```python
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

MODEL = "openai/gpt-5.6-sol"

app = AgentApp()


@app.main()
def main(_agent: AgentSession, context: Context) -> None:
    """Generate one response through the Flower runtime."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
    )
    response = client.responses.create(
        model=MODEL,
        input=prompt.strip(),
    )

    print(response.output_text)
```

The SDK returns its standard response object. In this example,
`response.output_text` collects the text output and writes it to the AgentApp
logs.

```{important}
`FLWR_RUNTIME_BASE_URL` is not `FLWR_MODEL_API_ENDPOINT`. The first is the
internal endpoint available to an AgentApp. The second configures the upstream
model provider for a self-hosted SuperLink and belongs outside AgentApp code.
```

## Build and run the AgentApp

```console
$ uv run flwr build
$ uv run flwr login supergrid
$ uv run flwr run . supergrid --stream
```

The runtime injects both environment variables when it starts the AgentApp. Do
not set, log, or persist `FLWR_RUNTIME_API_KEY` yourself.

## Choose between the SDK and `AgentSession`

Both interfaces send model requests through the Flower runtime:

- use `client.responses.create(...)` when you want the OpenAI SDK's typed API
  or are adapting existing SDK-based code
- use `agent.responses.create({...})` when you prefer JSON objects and do not
  need another dependency

Connector discovery and execution remain on `agent.connectors`. If the model
requests a function call, use the bounded tool loop described in [Build a
collaborative research
agent](../tutorials/build-a-collaborative-agent.md).

## Troubleshoot the SDK client

- **`ModuleNotFoundError: openai`**: run `uv add openai` or `uv sync`
- **Missing `FLWR_RUNTIME_BASE_URL` or `FLWR_RUNTIME_API_KEY`**: run the app
  through a Flower runtime that supports the Open Responses endpoint instead
  of starting the Python module directly
- **Authentication failure**: start a new run and use the credentials injected
  by its runtime rather than supplying your own key
- **Unsupported request field**: compare the request with the supported model
  fields in [The AgentApp runtime](../explanations/agentapp-runtime.md)

The runtime endpoint is internal to the AgentApp process. It is not a public
API for browsers, external services, or independently launched SDK clients.
