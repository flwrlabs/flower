# Understand the AgentApp runtime

An `AgentApp` contains agent control flow: what the model should do, which tools
it can use, and when the task is complete. Flower executes the app and supplies
model and connector access through an `AgentSession`.

## Where your app meets the runtime

A project declares an AgentApp component in `pyproject.toml`:

```toml
[tool.flwr.app.components]
agentapp = "example_agent.agent_app:app"
```

The `<module>:<attribute>` value tells Flower where to import the `AgentApp`
object. Flower packages the project as a Flower App Bundle (FAB). A run can
resolve an app by app spec, local project, or specific FAB hash.

When the task starts, Flower installs the FAB and its declared dependencies,
loads the object, and calls the function registered with `AgentApp.main`:

```python
@app.main()
def main(agent: AgentSession, context: Context) -> None:
    ...
```

The function is synchronous and returns when the app has completed its work. An
unhandled exception marks the AgentApp task as failed and records the error in
run details and logs.

## AgentSession

Flower creates an `AgentSession` for each AgentApp task. Its public surface has
two capabilities:

- `agent.responses` creates model responses; and
- `agent.connectors` returns connector tools and executes function calls.

The runtime creates child tasks for model and connector work. Provider
credentials and connector implementations remain outside the FAB.

### Model responses

`agent.responses.create(request)` accepts an Open Responses-compatible JSON
object. The Flower 1.34.0 runtime recognizes:

- `model` and `input`;
- `stream`;
- `tools` and `tool_choice`;
- `instructions` and `previous_response_id`;
- `reasoning` and `max_output_tokens`; and
- `metadata` and `text`.

`model` must be a non-empty string. `input` can be text or a sequence of JSON
items. The call returns an Open Responses-compatible response object and appends
model output items to the Flower `Context`.

“Open Responses-compatible” describes the request and response shape inside the
runtime. It does not mean Flower currently exposes a public
`/v1/runtime/responses` HTTP endpoint.

The default model provider at `api.flower.ai` does not currently support
continuing with `previous_response_id`. Rebuild `input` from stored messages for
a follow-up request instead.

### Connectors

`agent.connectors.tools(refs)` returns model-facing tool definitions. A built-in
reference normally yields one tool; an account connector such as `slack` can
yield several related action tools.

When a model returns a `function_call`, pass that item to
`agent.connectors.call(tool_call)`. Flower resolves the action, starts the
connector child task, records its activity, and returns a
`function_call_output` item for the next model request.

The AgentApp owns the tool loop and must bound it. See [Use
connectors](use-connectors.md).

## Context

The main function also receives a Flower `Context`:

- `context.run_config` contains defaults from `pyproject.toml` fused with
  per-run overrides;
- `context.state` stores records persisted for the run series, and its public
  `config_records` view selects stored `ConfigRecord` values; and
- `context.run_id` identifies the current run.

If `agent.input` is a non-empty string, the runtime records it as an Open
Responses user-message item before invoking the AgentApp. Model output items,
connector outputs, and built-in connector activity are appended while the app
runs.

Related runs can receive the persisted series context. The app chooses what to
send to the model. A safe conversation loader selects only message items:

```python
import json

messages = []
items_record = context.state.config_records.get("items")
items = items_record.get("json", []) if items_record is not None else []
for item_json in items:
    item = json.loads(item_json)
    if item.get("type") == "message":
        messages.append(item)
```

Connector activity types such as `response.tool_call.started` are useful for
inspection but are not valid model conversation messages.

The current default Flower Agent converts stored user and assistant messages
back into model input. A simple custom AgentApp that forwards only
`context.run_config["agent.input"]` treats every run independently even when
the runs share a series.

## Run series and federations

A run belongs to one federation. A run series groups runs within that
federation and carries their persisted context. Browser chat presents a series
as a conversation; `flwr chat` reuses its current series ID until `/new` or an
agent change.

The series does not own a permanent federation Agent assignment. Persistent
add/remove management for federation agents remains planned.

## Run lifecycle

1. The CLI or browser resolves an AgentApp and submits a run to a federation.
1. SuperGrid validates account membership, app configuration, and selected
   account connectors.
1. SuperGrid creates the run, run series when needed, and AgentApp task.
1. An executor starts the isolated AgentApp process and loads its FAB.
1. Flower initializes `AgentSession` and the persisted `Context`.
1. The main function creates model and connector child tasks as needed.
1. Flower persists context and structured activity after each operation.
1. Returning completes the task; an exception or stop request records the
   corresponding terminal state.

## AgentApp and other Flower Apps

A FAB currently supports either:

- one `agentapp` component; or
- a `serverapp` and a `clientapp`.

Do not combine an `agentapp` with a `serverapp` or `clientapp` in the same
bundle. AgentApp runs are agent tasks rather than federated-learning
simulations.
