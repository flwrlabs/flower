# Run an AgentApp with a local SuperLink and Ollama

Run a Flower AgentApp entirely on your machine with an
[Ollama](https://ollama.com/) model and a local SuperLink. This setup does not
require a Flower account or model API key.

This guide builds on [Run an AgentApp with a local
SuperLink](run-with-local-superlink.md). The example streams responses from
`qwen3.5:4b`, but you can use another model that supports the required Ollama
endpoint.

```{warning}
This guide disables TLS and is intended for local development only. Do not
expose the Ollama or SuperLink ports to an untrusted network.
```

## Install the prerequisites

Install:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Ollama 0.13.3 or newer

Pull the default model:

```console
$ ollama pull qwen3.5:4b
```

Start Ollama if it is not already running through the desktop app:

```console
$ ollama serve
```

Ollama listens on `127.0.0.1:11434` by default. Leave this terminal open.

## Create the AgentApp

Create the project structure:

```console
$ mkdir agent-ollama
$ cd agent-ollama
$ mkdir agent
$ touch agent/__init__.py
```

Create `agent/agent_app.py`:

```python
"""A minimal Flower AgentApp backed by a local Ollama model."""

import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from openai import OpenAI

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Send the configured input to the configured Ollama model."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    model = context.run_config.get("agent.model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("agent.model must be a non-empty string")

    client = OpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    )
    stream = client.responses.create(
        model=model.strip(),
        input=prompt.strip(),
        stream=True,
    )

    output_text: list[str] = []
    for event in stream:
        agent.events.emit(event.to_dict())
        if event.type in {"error", "response.failed"}:
            raise RuntimeError(f"Model response failed: {event}")
        if event.type == "response.output_text.delta":
            output_text.append(event.delta)

    print("".join(output_text))
```

The app connects to Flower's runtime bridge instead of connecting directly to
Ollama. Flower supplies `FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY` to
the AgentApp process. Emitting each streamed event makes the response visible
as structured run activity, while the final `print` also writes the complete
text to the process log.

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "agent-ollama"
version = "0.1.0"
description = "A local Flower AgentApp backed by Ollama"
license = "Apache-2.0"
requires-python = ">=3.11,<4.0"
dependencies = [
    "flwr>=1.37.0,<2.0",
    "openai>=2.16.0,<3.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["agent"]

[tool.flwr.app]
publisher = "local"
display-name = "Ollama Agent"
flwr-version-target = "1.37.0"
fab-include = ["agent/**/*.py"]

[tool.flwr.app.components]
agentapp = "agent.agent_app:app"

[tool.flwr.app.config.agent]
model = "qwen3.5:4b"
input = "Explain why flowers turn toward light."
```

Install the dependencies and validate the Flower App Bundle (FAB):

```console
$ uv sync
$ uv run flwr build
```

## Connect SuperLink to Ollama

Start an insecure local SuperLink from the `agent-ollama` directory and point
its model provider at Ollama's OpenAI-compatible Responses endpoint:

```console
$ export FLWR_MODEL_API_ENDPOINT="http://127.0.0.1:11434/v1/responses"
$ uv run flower-superlink --insecure
```

You do not need to set `FLWR_MODEL_API_KEY` because the local Ollama endpoint
does not require authentication. Leave this terminal open.

## Add the local Flower connection

In another terminal, add the local SuperLink connection to
`~/.flwr/config.toml`:

```toml
[superlink.local-agent]
address = "127.0.0.1:8000"
insecure = true
```

The `local-agent` connection does not require `flwr login`.

## Run the app

From the `agent-ollama` directory, submit the app and stream its logs:

```console
$ uv run flwr run . local-agent --stream
```

To use another model, pull it before starting the run. Then override both the
declared model and prompt, for example:

```console
$ ollama pull qwen3.5:9b
$ uv run flwr run . local-agent \
    --run-config 'agent.model="qwen3.5:9b" agent.input="Explain why the sky is blue in one sentence."' \
    --stream
```

## Troubleshoot Ollama runs

- **`/v1/responses` returns `404`:** upgrade Ollama to version 0.13.3 or newer.
- **Port `11434` refuses connections:** start Ollama with `ollama serve`, or
  confirm that the desktop app is running.
- **Ollama reports that the model is missing:** run `ollama pull <model>` and
  retry the Flower run.
- **Port `8000` refuses connections:** confirm that SuperLink is running and
  that `local-agent` points to `127.0.0.1:8000`.
- **The run fails without enough detail:** inspect it with `uv run flwr list --run-id <run-id> local-agent` and `uv run flwr log <run-id> local-agent --show`.

Press {kbd}`Ctrl+C` in the SuperLink and `ollama serve` terminals when you are
finished.
