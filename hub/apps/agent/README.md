---
tags: [agentapp]
dataset: []
framework: []
---

# Flower AgentApp

This minimal `AgentApp` uses the OpenAI SDK to send the agent prompt through
Flower Runtime. It republishes every streamed response event to the frontend
and prints the final response text. Use it as a starting point for a custom
Flower Agent.

Flower Runtime supplies the SDK base URL and task token, so the AgentApp does
not need provider credentials.

## Build

Install the project and build its Flower App Bundle (FAB):

```shell
uv sync
uv run flwr build
```

## Customize

Edit `agent/agent_app.py` to change the model or add your agent logic. The
prompt is available as `agent.prompt`.

## Learn more

See the [Flower Agent documentation](https://flower.ai/docs/agent/) for more
tutorials and guides.
