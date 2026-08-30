# Understand the AgentApp runtime

An `AgentApp` contains the control flow for an agent: what the model should do,
which tools it can use, and when its work is complete. Flower executes the app
and provides an OpenAI-compatible endpoint for model access. Connectors and
frontend-visible events are available through an `AgentSession`.

## Where your app meets the runtime

For example, if your AgentApp is defined in `your_package/agent_app.py`, declare
it in `pyproject.toml`:

```toml
[tool.flwr.app.components]
agentapp = "your_package.agent_app:app"
```

The `<module>:<attribute>` value tells Flower where to import the `AgentApp`
object. Flower packages the project as a Flower App Bundle (FAB). A run can
resolve an app by app spec, local project, or specific FAB hash.

When a run starts, Flower installs the FAB and its declared dependencies,
loads the object, and calls the function registered with `AgentApp.main`:

```python
@app.main()
def main(agent: AgentSession, context: Context) -> None:
    ...
```

The function is synchronous and returns when the app has completed its work. An
unhandled exception marks the run as failed and records the error in its details
and logs.

## AgentSession

Flower creates an `AgentSession` for each AgentApp run and passes it to your
main function. It exposes three capabilities:

- `agent.responses` provides a lower-level JSON model API
- `agent.connectors` returns connector tools and executes function calls
- `agent.events` publishes structured events selected by the AgentApp

Provider credentials and connector implementations remain outside the FAB. New
AgentApps normally make model requests with the OpenAI SDK and use
`AgentSession` for connectors and frontend-visible events.

### Model responses

Flower 1.35.0 exposes an OpenAI-compatible Responses endpoint inside the
AgentApp process. The runtime injects its URL and credential as
`FLWR_RUNTIME_BASE_URL` and `FLWR_RUNTIME_API_KEY`. Pass them to the OpenAI
client, then use its standard typed Responses API:

```python
import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
    api_key=os.environ["FLWR_RUNTIME_API_KEY"],
    max_retries=0,
)
stream = client.responses.create(
    model="openai/gpt-5.6-sol",
    input="Explain federated AI.",
    stream=True,
)
```

The runtime recognizes these request fields:

- `model` and `input`
- `stream`
- `tools` and `tool_choice`
- `instructions` and `previous_response_id`
- `reasoning` and `max_output_tokens`
- `metadata` and `text`

`model` must be a non-empty string. `input` can be text or a sequence of input
items. Streaming calls yield typed SDK events. The AgentApp decides which of
those events to publish and which output to persist in `Context`.

The endpoint is authenticated for the current AgentApp task. It is not a public
model API for an external client. See [Use the OpenAI SDK in an
AgentApp](../how-to-guides/use-openai-sdk.md) for a complete example.

`agent.responses.create(request)` remains available as a lower-level interface
for JSON-based workflows. It returns a JSON response object and automatically
appends its model output items to the Flower `Context`. New AgentApps should
prefer the OpenAI SDK when they need typed responses or streaming events.

The default model provider at `api.flower.ai` does not currently support
continuing with `previous_response_id`. Rebuild `input` from stored messages for
a follow-up request instead. See **Rebuild conversation input** in [Build a
collaborative research agent](../tutorials/build-a-collaborative-agent.md) for a
complete example.

### Connectors

`agent.connectors.tools(refs)` returns model-facing tool definitions. A built-in
reference normally yields one tool. An account connector such as `slack` can
yield several related action tools.

When a model returns a `function_call`, pass that item to
`agent.connectors.call(tool_call)`. Flower resolves the action, runs the
connector, records its activity, and returns a `function_call_output` item for
the next model request.

The AgentApp owns the tool loop and must bound it. See [Use
connectors](use-connectors.md).

### Run events

`agent.events.emit(event)` publishes one structured event to the run-event
stream consumed by Flower Chat and other clients. An SDK stream stays private
to the model task until the AgentApp republishes its events:

```python
for event in stream:
    agent.events.emit(event.to_dict())
```

Publishing an event does not append it to `Context`. This lets the AgentApp
separate frontend-visible progress from conversation state.

These operations have distinct destinations:

| Operation                               | Destination                                                |
| --------------------------------------- | ---------------------------------------------------------- |
| `print(...)`                            | AgentApp logs                                              |
| `agent.events.emit(...)`                | Run-event stream consumed by Flower Chat and other clients |
| Store an assistant message in `Context` | Persistent conversation state                              |

See {ref}`publish-agentapp-generated-text` for the event sequence used to
present text that does not come from an SDK stream.

## Context

Alongside the `AgentSession`, your main function receives a Flower `Context`:

- `context.run_config` contains defaults from `pyproject.toml` fused with
  per-run overrides
- `context.state` stores records persisted for the run series
- `context.run_id` identifies the current run

The runtime stores conversation items in a `ConfigRecord` named `items`. A
`ConfigRecord` is a specialized Python dictionary, so you can use methods such
as `get` when reading it through `context.state.config_records`.

If `agent.input` is a non-empty string, the runtime records it as an Open
Responses user-message item before calling the AgentApp. Connector calls append
their outputs and built-in activity. The lower-level `agent.responses` API also
appends model output, while SDK responses and events emitted with `agent.events`
are persisted only when the app stores them explicitly.

Runs in the same series can receive the persisted context. The app chooses what
to send to the model. A safe conversation loader selects only message items:

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
as a conversation. `flwr chat` reuses its current series ID until `/new`, an
agent change, or a federation change. `/history` can restore an earlier series
in the active federation.

## Run lifecycle

1. The CLI or browser resolves an AgentApp and submits a run to a federation
1. SuperGrid validates account membership, app configuration, and selected
   account connectors
1. SuperGrid creates the run and a run series when needed
1. An executor starts the isolated AgentApp process and loads its FAB
1. Flower initializes `AgentSession` and the persisted `Context`
1. The main function sends model requests and calls connectors as needed
1. The AgentApp publishes the model and connector events clients should see
1. During shutdown, Flower pushes the resulting `Context` once and records
   whether the run completed, failed, or stopped

## AgentApp and other Flower Apps

A FAB currently supports either:

- one `agentapp` component
- a `serverapp` and a `clientapp`

Do not combine an `agentapp` with a `serverapp` or `clientapp` in the same
bundle. AgentApp runs execute agent logic rather than federated-learning
simulations.
