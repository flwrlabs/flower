---
tags: [agentapp]
dataset: []
framework: []
---

# Flower AgentApp

This minimal `AgentApp` uses the OpenAI SDK to send the current prompt and the
conversation's previous user and assistant messages through Flower Runtime. It
republishes every streamed response event to the frontend and prints the final
response text. Use it as a starting point for a custom Flower Agent.

Flower Runtime supplies the SDK base URL and task token, so the AgentApp does
not need provider credentials.

By default, the model provider allows 600 seconds of socket inactivity within
the Runtime's 900-second total response window. The OpenAI client waits up to
930 seconds without response data, which gives the Runtime a 30-second delivery
margin. Set `FLWR_RUNTIME_API_TIMEOUT` in the AgentApp environment to change the
client timeout. Keep it greater than `FLWR_MODEL_RESPONSE_TIMEOUT` when
overriding both.

## Build

Install the project and build its Flower App Bundle (FAB):

```shell
uv sync
uv run flwr build
```

## Customize

Edit `agent/agent_app.py` to change the model or add your agent logic. The
current prompt is available as `agent.prompt`, and the run-series history is
available through `agent.events.get_trace()`.

## Learn more

See the [Flower Agent documentation](https://flower.ai/docs/agent/) for more
tutorials and guides.
