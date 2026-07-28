# Understand the AgentApp runtime

When you run an AgentApp, your code can focus on the agent itself: what the
model should do, which tools it can use, and when the task is complete. Flower
takes care of getting that code onto the runtime, connecting it to models and
connectors, and recording the result.

Two pieces work together:

- your `AgentApp` contains the agent's control flow;
- the Flower runtime provides model and connector execution through an
  `AgentSession`.

Let's look at where they meet and follow an AgentApp from submission to
completion.

## Where your app meets the runtime

An AgentApp project declares one component in `pyproject.toml`:

```toml
[tool.flwr.app.components]
agentapp = "agent:app"
```

The value is an object reference. When a run starts, Flower installs the Flower
App Bundle (FAB), imports the referenced object, verifies that it is an
`AgentApp`, and calls its registered main function:

```python
@app.main()
def main(agent: AgentSession, context: Context) -> None:
    ...
```

The function is synchronous. It returns when the app has completed its work. An
unhandled exception marks the AgentApp task as failed and makes the exception
message available in the run details and logs.

This boundary keeps your app small: you write the orchestration logic, while
Flower handles the runtime services around it.

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
connectors](using-connectors.md) for a complete loop.

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
inspection, but it does not replace the app's control flow: the app must still
pass the appropriate input or `previous_response_id` when it makes the next
model request.

## Run lifecycle

Now that we've met the main pieces, let's follow a complete AgentApp run:

1. `flwr run` builds or resolves a FAB and submits it to SuperGrid.
1. SuperGrid validates the app configuration and creates an AgentApp task.
1. A Flower executor starts the isolated AgentApp process.
1. The process installs the FAB and, when enabled, its declared dependencies.
1. Flower fuses run configuration, creates `AgentSession` and `Context`, and
   loads the configured `AgentApp`.
1. The main function creates model or connector child tasks as needed.
1. On return, Flower persists the final context and marks the task completed.
1. On an unhandled exception or stop request, Flower records the corresponding
   failed or stopped status.

The model and connector tasks are runtime services, not Python objects created
by the app. This is why the same AgentApp code can use provider-backed
capabilities without embedding provider API keys.

## AgentApp and other Flower Apps

An AgentApp-only bundle does not need `ServerApp` or `ClientApp` components.
AgentApp runs are handled as agent tasks rather than federated-learning
simulations. Keep agent orchestration in the AgentApp and declare only the
dependencies its code imports.

The important idea is the separation of responsibilities: your AgentApp
decides what the agent does, and Flower provides the infrastructure it needs to
do it. This keeps the app portable while letting the runtime manage execution,
credentials, events, and persisted state.
