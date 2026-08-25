# Chat in your terminal

Use Flower Chat to run AgentApps from your terminal. In this tutorial, you'll
sign in to SuperGrid, stream a response, switch agents, and see how Flower
groups messages into a conversation.

Prefer the browser? Start with [Chat in your browser](quickstart.md).

```{note}
This tutorial targets Flower 1.35.0. Flower Agent and `flwr chat` are
experimental and may change between releases.
```

## Prerequisites

You'll need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/);
- Python 3.11 or newer;
- a Flower account with Flower Agent access; and
- a terminal that can open the browser-based login flow.

You don't need to provide model credentials for a SuperGrid run.

## Check the Flower version

Use `uvx` to run the documented version in an isolated environment:

```console
$ uvx --from flwr==1.35.0 flwr --version
flwr, version 1.35.0
```

Running an explicit version keeps every command in this tutorial on the same
CLI. If the package isn't available, follow the package or installation
instructions for your Flower environment.

## Configure and log in to SuperGrid

Flower creates `~/.flwr/config.toml` the first time a CLI command needs it. The
default file includes this connection:

```toml
[superlink.supergrid]
address = "supergrid.flower.ai"
```

If you maintain a custom file, ensure that section exists. Then log in:

```console
$ uvx --from flwr==1.35.0 flwr login supergrid
```

Open the printed authentication link and complete sign-in. The CLI stores the
resulting account credentials for later SuperGrid commands.

## Start a chat

```console
$ uvx --from flwr==1.35.0 flwr chat
```

```{figure} ../_static/screenshots/flwr-chat.png
:alt: Flower Chat terminal interface with Flower Agent selected and an empty prompt.

Flower Chat shows the selected agent above the prompt.
```

Flower verifies your stored login before opening the full-screen interface. At
the `❯` prompt, ask:

```text
Explain Flower Agent in one sentence.
```

The selected AgentApp starts a SuperGrid run. Its response, reasoning summary,
and supported tool activity can stream into the transcript. When the run
finishes, the prompt becomes available again.

If the interface does not open, follow [Troubleshoot AgentApp
runs](../how-to-guides/troubleshoot-agent-runs.md) before changing your project
or account connectors.

## Continue the conversation

Ask a follow-up that depends on the previous response:

```text
Rewrite that explanation for a ten-year-old.
```

Flower keeps both runs in the same run series. The default Flower Agent replays
the stored messages, so it can use the earlier answer. Custom AgentApps need to
implement that behavior themselves; a run series doesn't automatically give
the model access to earlier messages.

## Select another agent

Type `@` at the start of an empty prompt. The completion menu lists agents
available in the federation configured for the `supergrid` connection. Select
one, add a request, and press {kbd}`Enter`:

```text
@publisher/agent Describe what you can help me with.
```

Only a leading app spec selects an agent. After a successful selection, the
label above the prompt changes. Selecting a different agent starts a new run
series, so context from the previous agent is not mixed into the new one.

If no agent appears, verify the current federation and account entitlement in
[Use agents and
federations](../how-to-guides/use-agents-and-federations.md).

## Use chat commands

Type `/` to open the command menu:

- `/help` lists available commands;
- `/new` makes the next message start a new run series; and
- `/quit` leaves Flower Chat.

You can also press {kbd}`Ctrl+C`:

- during a run, it requests the run to stop;
- with a draft, it clears the prompt; and
- from an empty idle prompt, it exits.

The CLI does not currently restore an older run series interactively after you
leave. Use the SuperGrid browser to inspect saved conversations.

## What happened

Each submitted message started one `AgentApp` run. SuperGrid supplied:

- an `AgentSession` for model and connector calls;
- a Flower `Context` containing run configuration and persistent series state;
- the selected AgentApp, resolved from its app spec or FAB hash; and
- a federation in which the run and its series are stored.

Read [Use agents and
federations](../how-to-guides/use-agents-and-federations.md) for the full mental
model.

## Next steps

- [Write your first AgentApp](write-your-first-agentapp.md).
- [Build a collaborative research agent](build-a-collaborative-agent.md).
- [Use connectors](../explanations/use-connectors.md).
- [Run an AgentApp on
  SuperGrid](../how-to-guides/run-on-supergrid.md).

```{tip}
If you get stuck, join the Flower community on [Flower
Discuss](https://discuss.flower.ai/) or [Flower
Slack](https://flower.ai/join-slack). Include safe run identifiers, but never
include credentials or provider tokens.
```
