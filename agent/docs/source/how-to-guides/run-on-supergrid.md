# Run an AgentApp on SuperGrid

Run a local or downloaded AgentApp in Flower Chat, then inspect or stop its
SuperGrid runs.

Start with [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md) if you do not have a valid
AgentApp project. This guide targets Flower {{ stable_flwr_version }}.

To run the same app without SuperGrid, see [Run an AgentApp with a local
SuperLink](run-with-local-superlink.md).

## Prepare the CLI

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and log in:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr login supergrid
```

Use {substitution-code}`uvx --from flwr==|stable_flwr_version|` for standalone
commands. Use `uv run flwr` for commands that must load the local project
environment.

(get-an-agentapp-project)=

## Get an AgentApp project

Use a local AgentApp project, or browse [Flower Hub](https://flower.ai/apps)
and select **Agent** under **Types** to find one to download:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr new '@<publisher>/<agent>'
```

The command creates a local project directory. Use its path when loading the
app in Flower Chat. To check a project before chatting, run from its directory:

```console
$ uv sync
$ uv run flwr build
```

Fix configuration, dependency, and component-reference errors locally before
starting a remote run.

(load-an-agentapp-in-flower-chat)=

## Load and chat with the AgentApp

From an existing project directory, start Flower Chat with the project's Flower
installation:

```console
$ uv run flwr chat
```

For a downloaded project, you can start Flower Chat outside its directory with
the documented Flower version:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr chat
```

At the chat prompt, load the project and send a message:

```text
/load <path-to-app>
Explain Flower Agent in one sentence.
```

Use `/load .` when you started chat in the project directory. The path is
resolved relative to the directory where you started Flower Chat; quote paths
with spaces.

Flower builds the app when you load it and sends each message as a run. Continue
chatting at the prompt. Changes to the local app are rebuilt before the next
message. If a build fails, the previous build stays selected.

```{tip}
To switch federations or select an assigned agent, see [Use agents and
federations](use-agents-and-federations.md).
```

## Observe the run

The response and supported activity stream into Flower Chat. If you have a run
ID, inspect the run from the CLI:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr list --run-id <run-id> supergrid
$ uvx --from flwr==|stable_flwr_version| flwr log <run-id> supergrid --show
```

`flwr log` streams by default. Use `--show` to print the available logs once.
Open the run in SuperGrid to inspect structured model output, connector
activity, federation, and persisted context.

Process logs are useful for app output and exceptions. Connector activity is a
better signal than a general **Working** label when diagnosing which child task
is active.

## Stop a run

Press {kbd}`Ctrl+C` during a Flower Chat run to request that it stop. If you
have its run ID, you can also use:

```{code-block} console
:substitutions:

$ uvx --from flwr==|stable_flwr_version| flwr stop <run-id> supergrid
```

Wait for the run to reach a stopped terminal state before submitting a
replacement that could duplicate external work.

Stopping a run does not stop future automation executions. Stop the automation
separately under **Settings** > **Automations**.

## Recover from failure

Use [Troubleshoot AgentApp runs](troubleshoot-agent-runs.md) for authentication,
agent catalog, connector, heartbeat, interruption, and stuck-run recovery.

For a custom app, start with:

```{code-block} console
:substitutions:

$ uv run flwr build
$ uvx --from flwr==|stable_flwr_version| flwr list --run-id <run-id> supergrid
$ uvx --from flwr==|stable_flwr_version| flwr log <run-id> supergrid --show
```

Keep the run ID, series ID when visible, federation ID, app spec, Flower
version, and exact public error. Never include credentials or private connector
content in a support report.
