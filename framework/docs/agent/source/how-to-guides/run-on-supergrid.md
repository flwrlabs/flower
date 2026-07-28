# Run an AgentApp on SuperGrid

Use SuperGrid to submit an AgentApp, follow its progress, inspect its logs, and
stop it when necessary.

If you haven't created an AgentApp yet, start with [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md).

## Prepare the CLI

First, install or update Flower and authenticate with SuperGrid:

```console
$ pip install -U flwr
$ flwr login supergrid
```

Your SuperGrid account must have access to the Flower Agent runtime. The
`supergrid` connection is included in Flower's default CLI configuration.

## Run the built-in AgentApp

Want to check your setup before submitting your own code? Run Flower's built-in
AgentApp without a local project:

```console
$ flwr run @flwragent/flwr-agent supergrid \
    --run-config 'agent.input="Summarize the benefits of federated AI."'
```

`@flwragent/flwr-agent` is resolved by SuperGrid as the built-in AgentApp.
`agent.input` must be a non-empty string.

## Run a local AgentApp

To run your own AgentApp, open a terminal in a project whose `pyproject.toml`
declares an `agentapp` component:

```console
$ flwr run . supergrid
```

Flower validates the project, builds a Flower App Bundle, and submits it with
the run request. Override configured values for one run with `--run-config`:

```console
$ flwr run . supergrid \
    --run-config 'agent.input="Compare federated learning and centralized learning."'
```

An override key must already exist in the app's
`[tool.flwr.app.config]` configuration.

## Select a federation

If your account or deployment requires a specific federation, pass its full ID:

```console
$ flwr run . supergrid \
    --federation @<account>/<federation-name> \
    --run-config 'agent.input="Hello from this federation."'
```

The account must be a member of the target federation and entitled to start an
AgentApp run there. Without `--federation`, SuperGrid resolves the account's
default federation.

## Observe the run

Once SuperGrid accepts the request, `flwr run` prints a run ID. Keep it handy:
you can use it to inspect the status and process logs:

```console
$ flwr list --run-id <run-id> supergrid
$ flwr log <run-id> supergrid
```

Add `--stream` to the original run command to follow logs immediately:

```console
$ flwr run . supergrid --stream
```

Process logs show app output and exceptions. Open the run in the SuperGrid
dashboard to inspect structured model responses, connector activity, and the
persisted agent context.

## Stop a run

Stop an active run with:

```console
$ flwr stop <run-id> supergrid
```

Flower sends a stop request to SuperGrid and records the stopped run status.

## Troubleshoot a failed run

Start with the detailed status and logs:

```console
$ flwr list --run-id <run-id> supergrid
$ flwr log <run-id> supergrid --stream
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
  the account. Notion and Slack must be connected before the run can use them.
- **Federation or entitlement error:** verify the federation ID, membership,
  and Flower Agent access for the account.

To catch configuration and component-reference errors before submission, run:

```console
$ flwr build
```
