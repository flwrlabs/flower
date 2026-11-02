# Write your first AgentApp

Create a small AgentApp from the Flower Hub template, customize its prompt, and
run it on SuperGrid. The app makes one model request through the OpenAI SDK so
you can focus on the AgentApp lifecycle before adding connectors.

Complete [Chat in your terminal](get-started-with-flower-agent.md) first. This
tutorial targets Flower {{ stable_flwr_version }}.

## Create the project

Download the AgentApp template from Flower Hub:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr new @flwrlabs/agent
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

`AgentApp.main` registers the function Flower calls. The runtime passes:

- `agent`, an `AgentSession` with the prompt, connectors, and frontend-visible events
- `context`, which contains run configuration and persistent state

Flower also injects `FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY` into the
AgentApp process. The OpenAI client uses them to send the request through
Flower, so the project does not need a model-provider API key.

The SDK yields typed streaming events. The loop republishes each event through
`agent.events.emit` so Flower Chat and other run-event clients can render the
response. Calling `print` does not publish an assistant response; it writes the
completed answer only to the AgentApp logs.

## Review the Flower configuration

The generated `pyproject.toml` includes the SDK and targets Flower
{{ stable_flwr_version }}:

```{code-block} toml
:substitutions:

[project]
dependencies = ["flwr>=|stable_flwr_version|,<2.0", "openai>=2.16.0,<3.0.0"]

[tool.flwr.app]
flwr-version-target = "|stable_flwr_version|"

[tool.flwr.app.components]
agentapp = "agent.agent_app:app"
```

The component value uses `<module>:<attribute>`. Flower imports `app` from
`agent/agent_app.py`. When you chat, Flower passes your message to `app` as
`agent.prompt`.

## Create the environment

```console
$ uv sync
```

`uv` creates `.venv` and a lock file. You do not need to activate the
environment because the following commands use `uv run`.

```{admonition} Checkpoint
:class: tip

`uv sync` should resolve Flower {{ stable_flwr_version }} and the OpenAI SDK
without a dependency error.
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

From the project directory, log in and open Flower Chat:

```console
$ uv run flwr login supergrid
$ uv run flwr chat
```

At the chat prompt, load the app and send a message:

```text
/load .
Explain Flower Agent in one sentence.
```

```{admonition} Success checkpoint
:class: tip

The model response appears in the chat transcript.
```

If the run fails, see [Troubleshoot AgentApp
runs](../how-to-guides/troubleshoot-agent-runs.md).

## Understand this app's limits

The app makes one model request and exits. It does not:

- replay prior messages from a run series
- persist the assistant response for a later run
- expose connectors
- handle model-requested function calls
- create automations

Those behaviors belong in AgentApp code rather than appearing automatically.
Continue with [Build a research agent](build-a-research-agent.md) for a bounded connector loop with
conversation state, read [Use the OpenAI SDK in an
AgentApp](../how-to-guides/use-openai-sdk.md) for the runtime details, or
[publish the AgentApp to Flower
Hub](../how-to-guides/use-flower-hub.md) so others can run it.
