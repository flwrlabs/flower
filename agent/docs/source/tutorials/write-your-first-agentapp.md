# Write your first AgentApp

Build a small custom AgentApp, package it as a Flower App Bundle (FAB), and run
it on SuperGrid. The example makes one model request so you can isolate project
configuration from connector logic.

Complete [Chat in your terminal](get-started-with-flower-agent.md) first. This
tutorial targets Flower 1.35.0.

## Create the project

```console
$ mkdir hello-agent
$ cd hello-agent
$ mkdir hello_agent
$ touch hello_agent/__init__.py
```

Create this file tree:

```text
hello-agent/
├── .gitignore
├── hello_agent/
│   ├── __init__.py
│   └── agent_app.py
└── pyproject.toml
```

Add the generated environment and bundles to `.gitignore`:

```text
.venv/
*.fab
__pycache__/
```

## Define the AgentApp

Create `hello_agent/agent_app.py`:

```python
from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context

MODEL = "openai/gpt-5.6-sol"

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
            "input": prompt.strip(),
            "stream": True,
        }
    )
```

`AgentApp.main` registers the function Flower calls. The runtime passes:

- `agent`, an `AgentSession` for model and connector calls
- `context`, which contains the fused run configuration and persistent state

`agent.responses.create` accepts an Open Responses-compatible request. It does
not call a public HTTP endpoint from your app; the Flower runtime sends the
request to the configured model provider.

## Configure the Flower App

Create `pyproject.toml`:

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
dependencies = ["flwr>=1.35.0,<2.0"]

[tool.hatch.build.targets.wheel]
packages = ["hello_agent"]

[tool.flwr.app]
publisher = "local"
display-name = "Hello Agent"
flwr-version-target = "1.35.0"
fab-include = ["hello_agent/**/*.py"]

[tool.flwr.app.config.agent]
input = "Explain why flowers turn toward light."

[tool.flwr.app.components]
agentapp = "hello_agent.agent_app:app"
```

The component value uses `<module>:<attribute>`. Flower imports `app` from
`hello_agent/agent_app.py`. The nested `config.agent.input` value becomes
`context.run_config["agent.input"]`.

## Create the environment

```console
$ uv sync
```

`uv` creates `.venv` and a lock file. You do not need to activate the
environment; use `uv run` for project commands.

```{admonition} Checkpoint
:class: tip

`uv sync` should resolve a compatible Flower 1.x release and finish without a
dependency error.
```

## Validate the bundle

```console
$ uv run flwr build
```

The command should finish by reporting the created `.fab` path. It validates
the project configuration and component reference before submission.

If it reports that the component cannot be loaded, check all three names:

1. the `hello_agent` directory
1. the `agent_app.py` module
1. the `:app` object referenced in `pyproject.toml`

## Run on SuperGrid

Ensure you have logged in, then submit the project and stream its logs:

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

The command prints a run ID, the run reaches a finished state, and the model
response appears in the run activity. Keep the run ID for troubleshooting.
```

Open SuperGrid to inspect the structured response and persisted context. If the
run fails, use the printed ID with:

```console
$ uv run flwr list --run-id <run-id> supergrid
$ uv run flwr log <run-id> supergrid --show
```

## Understand this app's limits

The app makes one model request and exits. It does not:

- replay prior messages from a run series
- expose connectors
- handle model-requested function calls
- create automations

Those behaviors belong in AgentApp code rather than appearing automatically.
Continue with [Build a collaborative research
agent](build-a-collaborative-agent.md) for a complete, bounded connector loop.
