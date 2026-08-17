# Troubleshoot AgentApp runs

Start with the least disruptive check and preserve safe identifiers as you go.
Avoid changing the app, federation, account connection, and login all at once;
one controlled change makes the cause easier to identify.

## Collect the first evidence

For a CLI-started run, record the run ID and inspect its status and logs:

```console
$ uvx --from flwr==1.34.0 flwr list --run-id <run-id> supergrid
$ uvx --from flwr==1.34.0 flwr log <run-id> supergrid --show
```

In the browser, keep the conversation open and note the visible federation,
selected agent, selected connector names, and approximate failure time.

## The `supergrid` connection is missing

**Symptom:** `flwr login supergrid` or `flwr chat` reports that the connection
does not exist.

Check `~/.flwr/config.toml`:

```toml
[superlink.supergrid]
address = "supergrid.flower.ai"
```

Preserve other connections and settings. Add only the missing section, then run
`flwr login supergrid` again.

## Authentication expired or was rejected

**Symptom:** the CLI fails before opening Flower Chat, or SuperGrid asks you to
sign in again.

1. Complete `uvx --from flwr==1.34.0 flwr login supergrid` again.
1. Ensure the browser flow uses the same Flower account that has Agent access.
1. Retry one deterministic prompt in a new chat.

Do not copy authentication tokens from local files into a support message.

(agent-isnt-listed)=

## Agent isn't listed

**Symptom:** typing `@` shows no completion, or the browser agent selector is
empty.

1. Wait for the catalog request to finish and refresh once.
1. Confirm the federation shown in the browser or configured under
   `[superlink.supergrid].federation`.
1. Run `flwr federation list supergrid` to confirm membership.
1. Try the default Flower Agent.

An agent being available in another federation does not make it available in
the current catalog. Persistent add/remove federation Agent management is not
yet a shipped recovery path.

## Connector is unavailable

**Symptom:** the selector does not show a provider, the AgentApp reports an
unsupported connector, or the run rejects selected connector references.

First distinguish the connector types:

- built-ins (`web_search`, `web_fetch`, `browser_use`, `start_automation`) are
  requested in AgentApp code;
- account connectors (Slack, Notion, GitHub, Attio) must be connected and
  selected for a browser run.

Account connectors are personal-workspace-only. Remove them from a
collaborative-federation run or move the task to your personal workspace. If a
built-in is missing from the deployed runtime, record it as an availability
problem rather than starting an OAuth flow.

(oauth-was-rejected-or-did-not-finish)=

## OAuth was rejected or did not finish

**Symptom:** the provider denies consent, the callback returns an error, or the
settings page remains disconnected.

1. Read the provider message before retrying.
1. Confirm you selected the intended workspace and are allowed to authorize
   integrations.
1. Return to **Settings** > **Connectors** and retry once.
1. If the connection existed before, use **Reconnect**.

For Notion, ensure the intended pages are shared with the integration. For
Slack or Attio, workspace policy may require an administrator. A rejected
consent flow is not fixed by putting provider credentials in AgentApp code.

## Connector timed out or failed

**Symptom:** connector activity remains active, returns a timeout, or the app
raises `Connector '<name>' failed`.

1. Keep the failed run ID and connector name.
1. Retry the smallest deterministic request once in a new run.
1. For an account connector, confirm the connection still shows **Connected**.
1. Reconnect only after a repeatable authorization or expiry error.
1. Design custom AgentApps to catch `RuntimeError` only when they have a useful
   fallback.

Do not loop retries around a provider outage. Report the repeated result with
timestamps.

## `No heartbeat received from task`

This message means the runtime stopped receiving health signals from a task. It
does not identify whether the original cause was app startup, model work, a
connector, or infrastructure.

1. Inspect run status and logs for the activity immediately before the heartbeat
   message.
1. Wait for the run to reach a terminal state.
1. Retry once with the default AgentApp and no account connectors.
1. If that succeeds, add the original agent and connectors back one at a time.

If the run never reaches a terminal state, record the identifiers and escalate
instead of repeatedly submitting duplicates.

## Run was interrupted

In Flower Chat, {kbd}`Ctrl+C` during a response requests a stop. Wait until the
prompt is idle before submitting again. If the old run remains active, use:

```console
$ uvx --from flwr==1.34.0 flwr stop <run-id> supergrid
```

Start a new conversation if the interrupted app left incomplete tool state that
should not be reused.

## Run appears stuck

1. Check whether the UI is still receiving response or tool activity.
1. Inspect status and logs from the CLI.
1. Allow an active model or browser task a reasonable completion window.
1. Stop the run once if it is no longer making progress.
1. Start one simplified run without optional connectors.

A global **Working** label does not prove which child task is active. Use the
structured activity and logs rather than opening duplicate runs.

## Custom AgentApp fails before responding

Run local validation first:

```console
$ uv run flwr build
```

Common causes are:

- an invalid `<module>:<attribute>` AgentApp component;
- a run-config override for a key not declared in `pyproject.toml`;
- an undeclared Python dependency;
- an unsupported model or connector; or
- missing or empty `agent.input`.

Compare with [Write your first
AgentApp](../tutorials/write-your-first-agentapp.md) before adding more control
flow.

## Prepare a safe support report

Include:

- Flower version;
- UTC time and timezone of the failure;
- run ID and, when visible, series ID;
- federation ID;
- selected app spec and connector names;
- exact public error text; and
- the smallest steps that reproduce it.

Do not include:

- access or refresh tokens;
- OAuth codes or callback URLs;
- passwords, API keys, or provider credentials;
- private Slack, Notion, GitHub, or Attio content; or
- an entire local configuration file without redaction.

Use [Flower Discuss](https://discuss.flower.ai/) or [Flower
Slack](https://flower.ai/join-slack), or use the support channel provided for
your Flower environment.
