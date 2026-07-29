# Flower Agent

Flower Agent provides the runtime and building blocks you need to create
agentic applications on Flower. An `AgentApp` combines your agent logic with
runtime-provided access to models and connectors, then runs it as a Flower App
locally or on SuperGrid.

This documentation shows you how to:

- build and configure an `AgentApp`;
- call models and connectors through an `AgentSession`;
- understand what happens inside the AgentApp runtime;
- run and observe your agents on SuperGrid; and
- run an AgentApp with a local SuperLink.

```{note}
Flower Agent is experimental. Its APIs and runtime behavior may change between
releases.
```

## Start here

New to Flower Agent? Start with [Get started with Flower
Agent](tutorials/get-started-with-flower-agent.md). You'll run the built-in
AgentApp without writing any code. When you're ready to build something of your
own, continue with [Write your first
AgentApp](tutorials/write-your-first-agentapp.md).

```{toctree}
:caption: Tutorials
:maxdepth: 1

tutorials/get-started-with-flower-agent
tutorials/write-your-first-agentapp
```

```{toctree}
:caption: Explanations
:maxdepth: 1

explanations/agentapp-runtime
explanations/using-connectors
```

```{toctree}
:caption: How-to guides
:maxdepth: 1

how-to-guides/run-on-supergrid
how-to-guides/run-with-local-superlink
```

## Documentation boundaries

This site is compiled independently and contains Flower Agent tutorials,
guides, concepts, examples, and operational documentation. The `flwr.agentapp`
Python API reference remains in the Flower framework documentation, where it is
generated from the public API and its docstrings.

Design proposals and unimplemented features are not published here as product
capabilities. They remain engineering RFCs until the corresponding behavior is
available.
