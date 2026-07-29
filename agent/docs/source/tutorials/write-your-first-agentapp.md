# Write your first AgentApp

Welcome back!

In the previous tutorial, you ran Flower's built-in AgentApp on SuperGrid. Now
it's time to build one of your own. You'll create a small AgentApp, package it
as a Flower App, and run it with a prompt you choose.

If you haven't already, complete [Get started with Flower
Agent](get-started-with-flower-agent.md) first. It will help you install `uv`
and authenticate your CLI with SuperGrid.

## Create the project

Start by creating a directory for your new app:

```console
$ mkdir hello-agent
$ cd hello-agent
```

You'll create these files:

```text
hello-agent/
├── .gitignore
├── hello_agent/
│   ├── __init__.py
│   └── agent_app.py
└── pyproject.toml
```

First, add the virtual environment to `.gitignore` so it isn't scanned when you
build the Flower App Bundle:

```text
.venv/
```

## Define the AgentApp

Create an empty `hello_agent/__init__.py`, then add the agent logic Flower will
run to `hello_agent/agent_app.py`:

```python
from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

MODEL = "openai/gpt-5.5"

app = AgentApp()


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Run the agent once for the configured prompt."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    agent.responses.create(
        {
            "model": MODEL,
            "input": prompt,
            "stream": True,
        }
    )
```

`AgentApp.main` registers the function Flower calls when the task starts. The
runtime passes two arguments:

- `agent` provides access to models and connectors;
- `context` provides the run configuration and persistent run state.

The call to `agent.responses.create` uses an Open Responses-compatible request
and returns the corresponding response object.

## Configure the Flower App

Next, create `pyproject.toml` to tell Flower how to package, configure, and load
your AgentApp:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "hello-agent"
version = "0.1.0"
description = "My first Flower AgentApp"
license = "Apache-2.0"
requires-python = ">=3.11"
dependencies = ["flwr>=1.33.0,<2.0"]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "local"
fab-include = ["hello_agent/**/*.py"]

[tool.flwr.app.config.agent]
input = "Explain why flowers turn toward light."

[tool.flwr.app.components]
agentapp = "hello_agent.agent_app:app"
```

The `agentapp` component is an object reference in the form
`<module>:<attribute>`. Here, Flower imports the `app` object from
`hello_agent/agent_app.py`. The nested `config.agent.input` value becomes the
flattened `context.run_config["agent.input"]` entry used by the app.

## Create the environment

Use `uv` to resolve the dependencies declared in `pyproject.toml`:

```console
$ uv sync
```

`uv` creates a virtual environment in `.venv` and writes a `uv.lock` file. You
don't need to activate the environment: `uv run` executes commands inside it.

## Check the bundle

Before sending anything to SuperGrid, build the Flower App Bundle (FAB):

```console
$ uv run flwr build
```

This validates the configuration and component reference before writing a
`.fab` file. The FAB contains the app code and metadata that SuperGrid needs to
start the run.

## Run the AgentApp

Submit the project directory through the `supergrid` connection:

```console
$ uv run flwr run . supergrid
```

The default prompt comes from `pyproject.toml`. Override it for one run with
`--run-config`:

```console
$ uv run flwr run . supergrid \
    --run-config 'agent.input="Describe photosynthesis for a five-year-old."'
```

Open the printed run ID in the SuperGrid dashboard to inspect the response and
run activity.

## Make it your own

The app currently makes one model request and then exits. Try changing:

- `MODEL` to another model available to your SuperGrid account;
- `instructions`, `reasoning`, or `max_output_tokens` in the response request;
  or
- the app flow to make several model requests or use the connector loop
  described in [Use connectors](../explanations/using-connectors.md).

Each invocation of `uv run flwr run . supergrid` builds and submits the current
local project, so saved changes are included in the next run.

## Final remarks

Congratulations, you've written and run your first custom AgentApp! 🎉

You now have all the pieces of a Flower Agent project:

- an `AgentApp` with a registered main function;
- an `AgentSession` for calling runtime-provided capabilities;
- a `Context` for reading run configuration; and
- a `pyproject.toml` that makes the app discoverable and configurable.

This example deliberately keeps the agent logic small. From here, you can add
instructions, make multiple model calls, or give the model a connector that
lets it search the web.

Continue with [Use connectors](../explanations/using-connectors.md) to build
your first tool-calling loop. To learn how to configure, observe, and stop a
run, see [Run an AgentApp on
SuperGrid](../how-to-guides/run-on-supergrid.md). For local development, see
[Run an AgentApp with a local
SuperLink](../how-to-guides/run-with-local-superlink.md).
