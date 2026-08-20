---
tags: [agentapp]
dataset: []
framework: []
---

# Flower Agent

This minimal AgentApp sends the configured `agent.input` to a model and prints
the final response text. Use it as a starting point for a custom Flower Agent.

## Build

Install the project and build its Flower App Bundle:

```shell
uv sync
uv run flwr build
```

## Customize and run

Edit `agent/agent_app.py` to change the model or add your agent logic. Then log
in and run the app on SuperGrid:

```shell
uv run flwr login supergrid
uv run flwr run . supergrid --stream
```

Override the default input for a run with:

```shell
uv run flwr run . supergrid \
  --run-config 'agent.input="Explain federated learning in one paragraph."' \
  --stream
```

## Learn more

See the [Flower Agent documentation](https://flower.ai/docs/agent/) for more
tutorials and guides.
