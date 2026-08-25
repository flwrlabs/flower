# Flower Agent

Flower Agent lets you build and run agentic applications on Flower. An
`AgentApp` contains your agent logic, while Flower supplies model access,
connectors, run state, and a runtime that can execute the app locally or on
SuperGrid.

## Choose how to start

Start with whichever interface you prefer:

- [Chat in your browser](tutorials/quickstart.md) gets you to a first response
  without a local installation or model API key.
- [Chat in your
  terminal](tutorials/get-started-with-flower-agent.md) introduces the `flwr chat` terminal interface.

When you're ready to build:

- [Write your first AgentApp](tutorials/write-your-first-agentapp.md) builds the
  smallest useful custom app.
- [Build a collaborative research
  agent](tutorials/build-a-collaborative-agent.md) adds conversation context,
  multiple connector calls, and a bounded tool loop.

```{note}
Flower Agent is experimental. Its interfaces and behavior may change between
releases.
```

## Understand the pieces

- An **AgentApp** is packaged agent logic that Flower can execute.
- An **agent** is an AgentApp made available for selection by an app spec such
  as `@flwrlabs/flwr-agent` or by a specific Flower App Bundle (FAB).
- A **run** is one execution. Related runs can share a **run series**, which the
  chat interfaces present as a conversation.
- A **federation** is the SuperGrid workspace in which runs execute and are
  visible.
- A **connector** gives an AgentApp a runtime-provided tool. Some connectors are
  built in; account connectors use access granted by the signed-in user.

Read [Use agents and federations](how-to-guides/use-agents-and-federations.md)
to see how these pieces fit together and how to select agents and federations.

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/quickstart
tutorials/get-started-with-flower-agent
tutorials/write-your-first-agentapp
tutorials/build-a-collaborative-agent
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to-guides/use-agents-and-federations
how-to-guides/connect-accounts
how-to-guides/create-automations
how-to-guides/run-on-supergrid
how-to-guides/troubleshoot-agent-runs
how-to-guides/run-with-local-superlink
```

```{toctree}
:caption: Explanations
:maxdepth: 1

explanations/agentapp-runtime
explanations/use-connectors
```

## Documentation boundaries

This site contains Flower Agent tutorials, guides, concepts, and operational
documentation. The generated [`flwr.agentapp` Python API
reference](https://flower.ai/docs/framework/ref-api/flwr.agentapp.html) remains
in the Flower framework documentation.

Design proposals and partially implemented paths aren't documented as current
capabilities. The documentation describes behavior available in the Flower
framework and SuperGrid interfaces rather than planned product work.
