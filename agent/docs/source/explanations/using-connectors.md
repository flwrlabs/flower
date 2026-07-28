# Use connectors

Connectors let an AgentApp expose runtime-provided tools to a model without
embedding their implementation or provider credentials in the app.

For example, an agent can look up public information, find project notes in
Notion, and review a related discussion in Slack. Flower Agent currently
provides these connectors:

| Name         | Purpose                                            |
| ------------ | -------------------------------------------------- |
| `web_search` | Search the public web for current information.     |
| `notion`     | Work with content in a connected Notion workspace. |
| `slack`      | Search and read from a connected Slack workspace.  |

## Give tools to the model

Start by asking the runtime for the tool definitions you want to expose:

```python
tools = agent.connectors.tools(["web_search", "notion", "slack"])
```

Then include them in a model request:

```python
response = agent.responses.create(
    {
        "model": "openai/gpt-5.5",
        "input": (
            "Summarize our latest launch notes from Notion and Slack, "
            "then verify the public details on the web."
        ),
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
item whose type is
`function_call`, call the connector and send the resulting
`function_call_output` items back to the model:

```python
tools = agent.connectors.tools(["web_search", "notion", "slack"])
response = agent.responses.create(
    {
        "model": "openai/gpt-5.5",
        "input": (
            "Summarize our latest launch notes from Notion and Slack, "
            "then verify the public details on the web."
        ),
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

- `web_search` for public information;
- `notion` for knowledge stored in a connected Notion workspace;
- `slack` for messages and discussions in a connected Slack workspace.

Only expose the connectors the task needs. This gives the model a smaller,
clearer set of tools to choose from.

## Connect Notion and Slack

Notion and Slack must be connected to your SuperGrid account and available to
the run before the AgentApp can use them. Flower supplies the connection to the
runtime, so the AgentApp does not need to handle OAuth tokens or provider
credentials.

`web_search` does not require an account connection.

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
