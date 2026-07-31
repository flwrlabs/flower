# Get started with Flower Agent

Welcome to Flower Agent!

In this tutorial, you'll chat with Flower's built-in AgentApp on SuperGrid. You
won't need to write any code or provide model credentials. By the end, you'll
have used Flower Chat and seen how its main pieces fit together.

```{note}
Flower Agent is experimental. Its APIs and runtime behavior may change between releases.
```

## Prerequisites

You need:

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- a Flower SuperGrid account with access to Flower Agent
- a terminal where you can run the Flower CLI

Let's get started! 🌼

## Run Flower with uvx

The `uvx` command runs Flower in an isolated environment, so you don't need to
create or activate a virtual environment. Check that the CLI is available:

```console
$ uvx flwr --version
```

`uvx` downloads Flower the first time you run it and reuses the cached
environment for later commands.

## Log in to SuperGrid

Now connect the Flower CLI to your SuperGrid account using the built-in
`supergrid` connection.

The Flower CLI includes this connection when it creates
`~/.flwr/config.toml`. If you already have a custom configuration that doesn't
define `supergrid`, add:

```toml
[superlink.supergrid]
address = "supergrid.flower.ai"
```

Log in:

```console
$ uvx flwr login supergrid
```

Follow the authentication link shown by the command. The CLI stores the
resulting account credentials for later SuperGrid commands.

## Start a chat

Open the interactive Flower Agent chat:

```console
$ uvx flwr chat
```

The command connects to SuperGrid and opens Flower Chat. When the `❯` prompt
appears, try asking:

```console
❯ Explain Flower Agent in one sentence.
```

Flower's built-in AgentApp handles the prompt, and its reply streams directly
into the chat. After it finishes, you can submit another prompt. Each prompt is
handled independently, so include any context the AgentApp needs in the prompt.

## Use chat commands

Type `/` at the prompt to open the command menu. Flower Chat can autocomplete
these commands:

- Enter `/help` to list the available commands.
- Enter `/new` to start a new conversation.
- Enter `/quit` to leave the chat.

You can also press {kbd}`Ctrl+C`. If Flower Agent is replying, this stops the
current run and returns you to the prompt. If you've started typing, it clears
the prompt; from an empty prompt, it leaves the chat.

## What happened

Each message starts an `AgentApp` run on SuperGrid. The `flwr chat` command:

1. sends your message to the built-in AgentApp
1. groups successive runs into the same run series until you enter `/new`
1. streams the AgentApp's reply back to your terminal

Behind the scenes, SuperGrid supplies the app with an `AgentSession` and a
Flower `Context`. The `AgentSession` is the app's interface to runtime-provided
model and connector capabilities, while the `Context` contains its run
configuration and state.

The run series keeps related runs together in SuperGrid, but it doesn't provide
conversation history to the model. The built-in AgentApp forwards only the
current prompt, so repeat any details an answer should take into account.

## Final remarks

Congratulations, you've completed your first interactive Flower Agent session
on SuperGrid! 🎉

You ran Flower with `uvx`, authenticated with SuperGrid, and chatted with the
built-in AgentApp from your terminal. The same runtime will also run AgentApps
you write yourself. You only need to provide the agent logic and project
configuration.

## Next steps

- [Write your first AgentApp](write-your-first-agentapp.md) to create a custom
  Flower Agent project.
- [Understand the AgentApp
  runtime](../explanations/agentapp-runtime.md) to learn how a run is executed.
- [Use connectors](../explanations/use-connectors.md) to let a model search
  the web.
- [Run an AgentApp on
  SuperGrid](../how-to-guides/run-on-supergrid.md) to learn how to configure,
  observe, and stop a run.
- [Run an AgentApp with a local
  SuperLink](../how-to-guides/run-with-local-superlink.md) for local
  development.

```{tip}
If you get stuck, join the Flower community on [Flower
Discuss](https://discuss.flower.ai/) or [Flower
Slack](https://flower.ai/join-slack). We'd be happy to help!
```
