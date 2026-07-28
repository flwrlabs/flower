# Run an AgentApp on SuperGrid

Use SuperGrid to submit an AgentApp, follow its progress, inspect its logs, and
stop it when necessary.

If you haven't created an AgentApp yet, start with [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md).

## Prepare the CLI

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you
haven't already, then authenticate with SuperGrid:

```console
$ uvx flwr login supergrid
```

Your SuperGrid account must have access to the Flower Agent runtime. The
`supergrid` connection is included in Flower's default CLI configuration.

This guide uses `uvx flwr` for standalone CLI commands and `uv run flwr` for
commands that need a local project's environment.

If this is your first Flower Agent run, follow [Get started with Flower
Agent](../tutorials/get-started-with-flower-agent.md) first. That tutorial uses
the built-in AgentApp to check your account and CLI setup without a local
project.

## Run a local AgentApp

To run your own AgentApp, open a terminal in a project whose `pyproject.toml`
declares an `agentapp` component:

```console
$ uv run flwr run . supergrid
```

Flower validates the project, builds a Flower App Bundle, and submits it with
the run request. Override configured values for one run with `--run-config`:

```console
$ uv run flwr run . supergrid \
    --run-config 'agent.input="Compare federated learning and centralized learning."'
```

An override key must already exist in the app's
`[tool.flwr.app.config]` configuration.

For a longer set of overrides, put them in a TOML file:

```toml
# run-config.toml
[agent]
input = "Compare federated learning and centralized learning."
```

Then pass the file to `--run-config`:

```console
$ uv run flwr run . supergrid --run-config run-config.toml
```

Don't combine a TOML file with inline `--run-config` values in the same command.

## Run in another federation

Every SuperGrid account has a default federation, and the commands above use it
automatically. To run in another federation, pass its full ID:

```console
$ uv run flwr run . supergrid \
    --federation @<account>/<federation-name> \
    --run-config 'agent.input="Hello from this federation."'
```

The account must be a member of the target federation and entitled to start an
AgentApp run there.

## Observe the run

Once SuperGrid accepts the request, `flwr run` prints a run ID. Keep it handy:
you can use it to inspect the status and process logs:

```console
$ uvx flwr list --run-id <run-id> supergrid
$ uvx flwr log <run-id> supergrid
```

Add `--stream` to the original run command to follow logs immediately:

```console
$ uv run flwr run . supergrid --stream
```

Process logs show app output and exceptions. Open the run in the SuperGrid
dashboard to inspect structured model responses, connector activity, and the
persisted agent context.

## Stop a run

Stop an active run with:

```console
$ uvx flwr stop <run-id> supergrid
```

Flower sends a stop request to SuperGrid and records the stopped run status.

## Troubleshoot a failed run

Start with the detailed status and logs:

```console
$ uvx flwr list --run-id <run-id> supergrid
$ uvx flwr log <run-id> supergrid --stream
```

Common failures include:

- **Invalid component reference:** confirm that
  `[tool.flwr.app.components].agentapp` uses `<module>:<attribute>` and resolves
  to an `AgentApp`.
- **Invalid run configuration:** define the key under
  `[tool.flwr.app.config]` before overriding it.
- **Missing dependency:** add every imported third-party package to
  `[project].dependencies`.
- **Unsupported model or connector:** use a model and connector available to
  the account. Account-backed connectors must be connected and included by the
  person starting the run.
- **Federation or entitlement error:** verify the federation ID, membership,
  and Flower Agent access for the account.

To catch configuration and component-reference errors before submission, run:

```console
$ uv run flwr build
```
