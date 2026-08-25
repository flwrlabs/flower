# Write your first AgentApp

Create a small AgentApp from the Flower Hub template, customize its prompt, and
run it on SuperGrid. The app makes one model request through the OpenAI SDK so
you can focus on the AgentApp lifecycle before adding connectors.

Complete [Chat in your terminal](get-started-with-flower-agent.md) first. This
tutorial targets Flower 1.35.0.

## Create the project

Download the AgentApp template from Flower Hub:

```console
$ uvx --from flwr==1.35.0 flwr new @flwrlabs/agent
$ cd agent
```

The command creates a ready-to-build project:

```text
agent/
├── .gitignore
├── agent/
│   ├── __init__.py
│   └── agent_app.py
├── LICENSE
├── README.md
└── pyproject.toml
```

Rename the project and change its `publisher` before publishing it under your
own account. You can keep the generated values while running it locally or on
SuperGrid.

## Understand the AgentApp

Open `agent/agent_app.py`:

```python
"""A minimal Flower AgentApp."""

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

`AgentApp.main` registers the function Flower calls. The runtime passes:

- `agent`, an `AgentSession` for connectors and frontend-visible events
- `context`, which contains the fused run configuration and persistent state

Flower also injects `FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY` into the
AgentApp process. The OpenAI client uses them to send the request through
Flower, so the project does not need a model-provider API key.

The SDK yields typed streaming events. The loop republishes each event through
`agent.events.emit` so Flower Chat and the browser can render the response. It
also collects text deltas and prints the completed answer to the run logs.

## Review the Flower configuration

The generated `pyproject.toml` includes the SDK and targets Flower 1.35.0:

```toml
[project]
dependencies = ["flwr>=1.35.0,<2.0", "openai>=2.16.0,<3.0.0"]

[tool.flwr.app]
flwr-version-target = "1.35.0"

[tool.flwr.app.config.agent]
input = "Explain why flowers turn toward light."

[tool.flwr.app.components]
agentapp = "agent.agent_app:app"
```

The component value uses `<module>:<attribute>`. Flower imports `app` from
`agent/agent_app.py`. The nested `config.agent.input` value becomes
`context.run_config["agent.input"]`.

Change the default prompt to something easy to recognize:

```toml
[tool.flwr.app.config.agent]
input = "Explain Flower Agent in one sentence."
```

## Create the environment

```console
$ uv sync
```

`uv` creates `.venv` and a lock file. You do not need to activate the
environment because the following commands use `uv run`.

```{admonition} Checkpoint
:class: tip

`uv sync` should resolve Flower 1.35 and the OpenAI SDK without a dependency
error.
```

## Validate the bundle

```console
$ uv run flwr build
```

The command should report the created `.fab` path. It validates the project
configuration and component reference before submission.

If Flower cannot load the component, check:

1. the `agent` package directory
1. the `agent_app.py` module
1. the `:app` object referenced in `pyproject.toml`

## Run on SuperGrid

Log in, then submit the project and stream its logs:

```console
$ uv run flwr login supergrid
$ uv run flwr run . supergrid --stream
```

Override the configured prompt for one run:

```console
$ uv run flwr run . supergrid \
    --run-config 'agent.input="Describe photosynthesis for a five-year-old."' \
    --stream
```

```{admonition} Success checkpoint
:class: tip

The command prints a run ID, the run reaches a finished state, and the streamed
model response appears in the run activity and logs.
```

If the run fails, use the printed ID with:

```console
$ uv run flwr list --run-id <run-id> supergrid
$ uv run flwr log <run-id> supergrid --show
```

## Understand this app's limits

The app makes one model request and exits. It does not:

- replay prior messages from a run series
- persist the assistant response for a later run
- expose connectors
- handle model-requested function calls
- create automations

Those behaviors belong in AgentApp code rather than appearing automatically.
Continue with [Build a collaborative research
agent](build-a-collaborative-agent.md) for a bounded connector loop with
conversation state, read [Use the OpenAI SDK in an
AgentApp](../how-to-guides/use-openai-sdk.md) for the runtime details, or
[publish the AgentApp to Flower
Hub](../how-to-guides/use-flower-hub.md) so others can run it.
