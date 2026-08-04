# Understand the AgentApp runtime

An `AgentApp` contains the control flow for an agent: what the model should do,
which tools it can use, and when the task is complete. The Flower runtime
executes that app and provides model and connector access through an
`AgentSession`.

## Where your app meets the runtime

A small AgentApp project might look like this:

```text
flwr-agent/
├── flwr_agent/
│   ├── __init__.py
│   └── agent_app.py
└── pyproject.toml
```

The project declares its AgentApp component in `pyproject.toml`:

```toml
[tool.flwr.app.components]
agentapp = "flwr_agent.agent_app:app"
```

The value follows the `<module>:<attribute>` format. In this example, Flower
imports `app` from `flwr_agent/agent_app.py`. See [Configure
`pyproject.toml`](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html)
for more about Flower App components and the files included in a Flower App
Bundle (FAB).

When a run starts, Flower installs the FAB, imports the referenced object,
verifies that it is an `AgentApp`, and calls its registered main function:

```python
@app.main()
def main(agent: AgentSession, context: Context) -> None:
    ...
```

The function is synchronous. It returns when the app has completed its work. An
unhandled exception marks the AgentApp task as failed and makes the exception
message available in the run details and logs.

## AgentSession

Flower creates an `AgentSession` for each AgentApp task and passes it to your
main function. It exposes two capabilities:

- `agent.responses` creates model responses;
- `agent.connectors` exposes connector tool schemas and executes connector
  calls.

Calling either capability creates a child task. The AgentApp waits for that
child task's reply, then continues with the returned JSON object. This keeps
provider credentials and connector implementation details outside the app.

### Model responses

`agent.responses.create(request)` accepts an Open Responses-compatible JSON
object. The current runtime forwards these request fields when present:

- `model` and `input`;
- `stream`;
- `tools` and `tool_choice`;
- `instructions` and `previous_response_id`;
- `reasoning` and `max_output_tokens`;
- `metadata` and `text`.

`model` must be a non-empty string, and `input` must be a string or a sequence
of JSON objects. The call returns an Open Responses-compatible response object.
The app is responsible for deciding whether to make another model request.

### Connectors

`agent.connectors.tools(names)` returns function-tool definitions for registered
connectors. An app passes those definitions to a model request. If the model
returns a `function_call`, the app passes that item to
`agent.connectors.call(tool_call)`.

The connector call returns a `function_call_output` item suitable for the next
model request. Flower validates the connector name, executes it in a child
task, and records connector activity. See [Using
connectors](use-connectors.md) for a complete loop.

## Context

Alongside the `AgentSession`, your main function receives a Flower `Context`.
It connects the AgentApp to its configuration and state:

- `context.run_config` contains the defaults from `pyproject.toml` fused with
  per-run overrides;
- `context.state` persists records produced during the run;
- `context.run_id` identifies the run.

If `agent.input` is configured and non-empty, the runtime also records it as an
Open Responses user-message item before calling the AgentApp.

Model output items, connector output items, and built-in connector activity are
appended to `context.state` by the runtime. This persistence supports run
inspection and lets the app rebuild the input for its next model request.

Connector activity events are useful for run inspection, but they aren't valid
model input items. Filter them out when rebuilding model input. The items are
stored as JSON strings:

```python
import json

items_record = context.state.get("items")
stored_items = (
    [] if items_record is None else [json.loads(item) for item in items_record["json"]]
)
context_items = [
    item
    for item in stored_items
    if not str(item.get("type", "")).startswith("response.tool_call.")
]
```

The default model provider at `api.flower.ai` does not currently support
continuing a conversation with `previous_response_id`. Pass `context_items` as
the `input` of each follow-up request instead.

## Run lifecycle

Now that we've met the main pieces, let's follow a complete AgentApp run:

1. `flwr run` builds or resolves a FAB and submits it to the configured
   SuperLink.
1. SuperLink validates the app configuration and creates an AgentApp task.
1. A Flower executor starts the isolated AgentApp process.
1. The process installs the FAB and, when enabled, its declared dependencies.
1. Flower fuses run configuration, creates `AgentSession` and `Context`, and
   loads the configured `AgentApp`.
1. The main function creates model or connector child tasks as needed.
1. On return, Flower persists the final context and marks the task completed.
1. On an unhandled exception or stop request, Flower records the corresponding
   failed or stopped status.

## AgentApp and other Flower Apps

A Flower App Bundle currently supports either:

- one `agentapp` component; or
- a `serverapp` and a `clientapp`.

Don't combine an `agentapp` with a `serverapp` or `clientapp` in the same
bundle. AgentApp runs are handled as agent tasks rather than federated-learning
simulations.
