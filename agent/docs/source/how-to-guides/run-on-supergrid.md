# Run an AgentApp on SuperGrid

Submit an AgentApp, choose its federation, follow progress, inspect logs, and
stop the run when necessary.

Start with [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md) if you do not have a valid
AgentApp project. This guide targets Flower 1.35.0.

To run the same app without SuperGrid, see [Run an AgentApp with a local
SuperLink](run-with-local-superlink.md).

## Prepare the CLI

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and log in:

```console
$ uvx --from flwr==1.35.0 flwr login supergrid
```

Use `uvx --from flwr==1.35.0` for standalone commands. Use `uv run flwr` for
commands that must load the local project environment.

## Validate before submission

From the project directory:

```console
$ uv sync
$ uv run flwr build
```

Fix configuration, dependency, and component-reference errors locally before
starting a remote run.

## Run a local AgentApp project

```console
$ uv run flwr run . supergrid --stream
```

Flower builds the project, submits the FAB, prints a run ID, and streams process
logs. Override a declared configuration value for one run:

```console
$ uv run flwr run . supergrid \
    --run-config 'agent.input="Compare federated and centralized AI."' \
    --stream
```

An override key must already exist under `[tool.flwr.app.config]`.

For longer overrides, create `run-config.toml`:

```toml
[agent]
input = "Compare federated and centralized AI."
```

Then run:

```console
$ uv run flwr run . supergrid --run-config run-config.toml --stream
```

Do not combine a TOML run-config file and inline run-config values.

## Run a published agent

After you [find or publish an AgentApp on Flower Hub](use-flower-hub.md), use
its app spec instead of a local directory:

```console
$ uvx --from flwr==1.35.0 flwr run @publisher/agent supergrid \
    --run-config 'agent.input="Explain your task."'
```

SuperGrid resolves the app spec to an available version of the app. Availability
can depend on the target federation.

## Choose a federation

List the federations visible to your account:

```console
$ uvx --from flwr==1.35.0 flwr federation list supergrid
```

Without `--federation`, SuperGrid uses the account default. To choose another
federation, use its full ID:

```console
$ uv run flwr run . supergrid \
    --federation @account/federation-name \
    --run-config 'agent.input="Hello from this federation."'
```

The account must be a member and entitled to execute AgentApps there.

```{important}
Slack, Notion, GitHub, and Attio connector references are currently accepted
only for personal-workspace runs. Built-in tools selected by the AgentApp do not
use the account-connector selection flow.
```

## Observe the run

Use the printed run ID:

```console
$ uvx --from flwr==1.35.0 flwr list --run-id <run-id> supergrid
$ uvx --from flwr==1.35.0 flwr log <run-id> supergrid --show
```

`flwr log` streams by default. Use `--show` to print the available logs once.
Open the run in SuperGrid to inspect structured model output, connector
activity, federation, and persisted context.

Process logs are useful for app output and exceptions. Connector activity is a
better signal than a general **Working** label when diagnosing which child task
is active.

## Stop a run

```console
$ uvx --from flwr==1.35.0 flwr stop <run-id> supergrid
```

Wait for the run to reach a stopped terminal state before submitting a
replacement that could duplicate external work.

Stopping a run does not stop future automation executions. Stop the automation
separately under **Settings** > **Automations**.

## Recover from failure

Use [Troubleshoot AgentApp runs](troubleshoot-agent-runs.md) for authentication,
agent catalog, connector, heartbeat, interruption, and stuck-run recovery.

For a custom app, start with:

```console
$ uv run flwr build
$ uvx --from flwr==1.35.0 flwr list --run-id <run-id> supergrid
$ uvx --from flwr==1.35.0 flwr log <run-id> supergrid --show
```

Keep the run ID, series ID when visible, federation ID, app spec, Flower
version, and exact public error. Never include credentials or private connector
content in a support report.
