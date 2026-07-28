# Use connectors

Connectors let an AgentApp expose runtime-provided tools to a model without
embedding their implementation or provider credentials in the app.

For example, an agent can search for a recent Flower release, fetch the
relevant page, and use what it finds in its final response. The current Flower
runtime provides these built-in connectors:

| Name          | Model-facing argument | Purpose                                               |
| ------------- | --------------------- | ----------------------------------------------------- |
| `web_search`  | `query`               | Search the web for current information.               |
| `web_fetch`   | `url`                 | Fetch a public web page and extract readable content. |
| `browser_use` | `task`                | Use a headless browser to complete a web task.        |

## Give tools to the model

Start by asking the runtime for the tool definitions you want to expose:

```python
tools = agent.connectors.tools(["web_search", "web_fetch"])
```

Then include them in a model request:

```python
response = agent.responses.create(
    {
        "model": "openai/gpt-5.5",
        "input": "What changed in the latest Flower release?",
        "tools": tools,
    }
)
```

`tools` returns the registered schemas rather than executing anything. The
model can respond with normal output, one function call, or multiple function
calls.

## Execute function calls

The AgentApp owns the tool loop. When the model asks to use a connector, your
app executes the call and gives the result back to the model. For each output
item whose type is
`function_call`, call the connector and send the resulting
`function_call_output` items back to the model:

```python
tools = agent.connectors.tools(["web_search", "web_fetch"])
response = agent.responses.create(
    {
        "model": "openai/gpt-5.5",
        "input": "Summarize the latest Flower release using primary sources.",
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

    tool_outputs = [
        agent.connectors.call(tool_call) for tool_call in tool_calls
    ]
    response = agent.responses.create(
        {
            "model": "openai/gpt-5.5",
            "input": tool_outputs,
            "previous_response_id": response["id"],
            "tools": tools,
        }
    )
    tool_turns += 1
```

`agent.connectors.call` accepts the function-call item returned by the model. It
parses the call arguments, starts the connector task, and returns an item with
the same `call_id`.

The loop allows at most five connector turns. A limit prevents a model from
repeatedly requesting tools without reaching a final response.

Once the model returns no more function calls, the loop ends and `response`
contains the final model response.

## Choose the narrowest connector

Each connector has a different job. As a rule of thumb, use:

- `web_search` to discover relevant pages;
- `web_fetch` when the app already has a URL and only needs its readable
  content;
- `browser_use` when the task requires interaction or browser-rendered state.

`web_fetch` accepts only HTTP or HTTPS URLs and rejects private, local, or
otherwise non-public destinations. It validates redirect targets as well.

`browser_use` runs headlessly. Its tool schema accepts a natural-language
`task`; describe both the intended action and the target site precisely.

## Declare optional dependencies

The browser connector and local web content extraction use Flower's optional
Agent dependencies. If your deployment does not provide them, declare the
Agent extra in the app:

```toml
[project]
dependencies = ["flwr[agent]>=1.33.0,<2.0"]
```

SuperGrid installs declared app dependencies when the AgentApp runtime supports
runtime dependency installation.

## Handle errors

Connector calls can fail when a provider or target is unavailable. Flower
records failed built-in connector activity before propagating the error.

Catch an exception only when the app has a useful fallback, for example trying
a different source:

```python
try:
    output = agent.connectors.call(tool_call)
except RuntimeError as exc:
    print(f"Connector failed: {exc}")
```

Do not put secrets in model prompts or connector arguments. Model requests and
built-in connector activity are recorded as part of the run context for
inspection.
