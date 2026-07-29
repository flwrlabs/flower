# Use connectors

Connectors let an AgentApp expose runtime-provided tools to a model without
embedding their implementation or provider credentials in the app.

The examples below use `web_search`, which is available without connecting an
external account.

## Give tools to the model

Start by asking the runtime for the tool definitions you want to expose:

```python
tools = agent.connectors.tools(["web_search"])
```

Then include them in a model request:

```python
response = agent.responses.create(
    {
        "model": "openai/gpt-5.5",
        "input": "Find the latest Flower release and summarize what changed.",
        "tools": tools,
    }
)
```

`tools` returns the registered schemas rather than executing anything. A
connector can expose several related tools. The model can respond with normal
output, one function call, or multiple function calls.

## Execute function calls

The AgentApp owns the tool loop. When the model asks to use a connector, your
app executes the call and gives the result back to the model. For each output
item whose type is `function_call`, call the connector and send the resulting
`function_call_output` items back to the model:

This loop expects `agent.input` in the run configuration. Flower records that
prompt in `context.state` before the AgentApp starts:

```python
import json

from flwr.app import Context


def load_context_items(context: Context) -> list[dict[str, object]]:
    """Load the Open Responses items stored by the Flower runtime."""
    record = context.state.get("items")
    if record is None:
        return []
    stored_items = [json.loads(item) for item in record["json"]]
    return [
        item
        for item in stored_items
        if not str(item.get("type", "")).startswith("response.tool_call.")
    ]


tools = agent.connectors.tools(["web_search"])
response = agent.responses.create(
    {
        "model": "openai/gpt-5.5",
        "input": load_context_items(context),
        "tools": tools,
    }
)

tool_turns = 0
while True:
    tool_calls = [
        item
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    if not tool_calls:
        break
    if tool_turns == 5:
        raise RuntimeError("Agent exceeded the connector turn limit")

    for tool_call in tool_calls:
        agent.connectors.call(tool_call)
    response = agent.responses.create(
        {
            "model": "openai/gpt-5.5",
            "input": load_context_items(context),
            "tools": tools,
        }
    )
    tool_turns += 1
```

`agent.connectors.call` accepts the function-call item returned by the model. It
parses the call arguments, starts the connector task, and stores the output in
the Flower `Context`. The next call to `load_context_items` includes that output
with the same `call_id`. The helper filters out the connector activity events
that Flower also stores for run inspection because those events aren't valid
model input items.

The loop allows at most five connector turns. A limit prevents a model from
repeatedly requesting tools without reaching a final response.

Once the model returns no more function calls, the loop ends and `response`
contains the final model response.

## Choose the narrowest set of connectors

Only expose the connectors the task needs. This gives the model a smaller,
clearer set of tools to choose from.

## Handle errors

Connector calls can fail when a provider or target is unavailable. The call
raises a `RuntimeError`; if the app does not catch it, the AgentApp task fails
and the error is available in the run details and logs.

Catch an exception only when the app has a useful fallback, for example trying
a different source:

```python
try:
    output = agent.connectors.call(tool_call)
except RuntimeError as exc:
    print(f"Connector failed: {exc}")
```

Do not put secrets in model prompts or connector arguments. The model or
connected service receives those values when the tool runs.
