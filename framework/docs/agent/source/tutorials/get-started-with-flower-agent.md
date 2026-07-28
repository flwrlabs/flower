# Get started with Flower Agent

Welcome to Flower Agent!

In this tutorial, you'll launch your first agent on SuperGrid using Flower's
built-in AgentApp. You won't need to write any code or provide model
credentials. By the end, you'll have started an agent run, inspected its status,
and seen how the main Flower Agent pieces fit together.

```{note}
Flower Agent is experimental. Its APIs and runtime behavior may change between
releases.
```

## Prerequisites

You need:

- Python 3.11 or newer;
- a Flower SuperGrid account with access to Flower Agent; and
- a terminal where you can install and run the Flower CLI.

That's all. Let's get started! 🌼

## Install Flower

First, create and activate a virtual environment, then install Flower:

```console
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -U flwr
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Log in to SuperGrid

Now connect the Flower CLI to your SuperGrid account using the built-in
`supergrid` connection:

```console
$ flwr login supergrid
```

Follow the authentication link shown by the command. The CLI stores the
resulting account credentials for later SuperGrid commands.

## Start the built-in AgentApp

You're ready to launch your first agent! Run `@flwragent/flwr-agent` and provide
the initial prompt through `agent.input`:

```console
$ flwr run @flwragent/flwr-agent supergrid \
    --run-config 'agent.input="Explain Flower Agent in one sentence."'
```

The CLI prints the run ID after SuperGrid accepts the run. The built-in
AgentApp sends your prompt to its configured model and records the response as
agent activity for the run.

Open the run in the SuperGrid dashboard to see the response and follow its
activity. You can also check its status from the terminal:

```console
$ flwr list --run-id <run-id> supergrid
```

To stream the process logs while starting another run, add `--stream`:

```console
$ flwr run @flwragent/flwr-agent supergrid \
    --run-config 'agent.input="Give me three uses for Flower Agent."' \
    --stream
```

Process logs help diagnose app startup and failures. The structured model and
connector activity remains available on the run page in SuperGrid.

## What happened

Quite a lot happened behind that one command. It started a Flower App Bundle
containing an `AgentApp`, and SuperGrid:

1. resolved the built-in app;
1. combined its default configuration with your `agent.input` override;
1. created an AgentApp task;
1. supplied the task with an `AgentSession` and a Flower `Context`; and
1. ran the app and persisted its result.

The `AgentSession` is the app's interface to runtime-provided model and
connector capabilities. The `Context` contains the run configuration and
persistent run state.

## Final remarks

Congratulations, you've run your first Flower Agent on SuperGrid! 🎉

You installed Flower, authenticated with SuperGrid, started the built-in
AgentApp with your own prompt, and inspected the resulting run. The same
runtime will also run AgentApps you write yourself—you only need to provide the
agent logic and project configuration.

## Next steps

- [Write your first AgentApp](write-your-first-agentapp.md) to create a custom
  Flower Agent project.
- [Understand the AgentApp
  runtime](../explanations/agentapp-runtime.md) to learn how a run is executed.
- [Use connectors](../explanations/using-connectors.md) to let a model search,
  fetch, or interact with the web.

```{tip}
If you get stuck, join the Flower community on [Flower
Discuss](https://discuss.flower.ai/) or [Flower
Slack](https://flower.ai/join-slack). We'd be happy to help!
```
