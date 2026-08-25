# Use connectors

Connectors give an AgentApp tools supplied by the runtime without embedding
provider implementations or credentials in the app.

## Distinguish built-in and account connectors

Built-in connectors need no external account:

| Reference          | Capability                        | Important boundary                           |
| ------------------ | --------------------------------- | -------------------------------------------- |
| `web_search`       | Search the public web             | Results depend on runtime availability       |
| `web_fetch`        | Fetch eligible public web content | Private and unsafe targets are blocked       |
| `start_automation` | Schedule AgentApp input           | Explicit future or recurring intent required |

Account connectors use access granted by a signed-in user:

- Slack searches and reads visible conversations and threads
- Notion searches and reads shared pages and data sources
- GitHub searches code and reads UTF-8 files in public repositories
- Attio searches records and reads meeting and call transcript data

All current account actions are read-only. They must be connected and selected
for the browser run and are restricted to the user's personal workspace. See
[Connect accounts](../how-to-guides/connect-accounts.md).

## Request the narrowest tool set

Ask the runtime for only the references needed by the task:

```python
tools = agent.connectors.tools(["web_search", "web_fetch"])
```

The method returns registered function-tool schemas; it does not execute
anything. A reference can expand into several tools. For example:

```python
slack_tools = agent.connectors.tools(["slack"])
```

returns tools for searching messages, listing conversations, reading history,
and reading thread replies.

A smaller tool set reduces accidental disclosure and makes model selection more
predictable.

## Give tools to the model

```python
response = client.responses.create(
    model="openai/gpt-5.6-sol",
    input="Find two public sources about federated AI.",
    tools=tools,
    tool_choice="auto",
)
```

Here, `client` is the OpenAI client configured with Flower's injected runtime
URL and credential. See [Use the OpenAI SDK in an
AgentApp](../how-to-guides/use-openai-sdk.md). The model can return normal
output, one function call, or several independent function calls.

## Execute requested calls

```python
response_output = [item.to_dict() for item in response.output]
tool_calls = [
    item for item in response_output if item.get("type") == "function_call"
]
allowed_tool_names = {
    tool["name"] for tool in tools if isinstance(tool.get("name"), str)
}
for tool_call in tool_calls:
    if tool_call.get("name") not in allowed_tool_names:
        raise RuntimeError(f"Tool {tool_call.get('name')!r} was not exposed")

function_outputs = [agent.connectors.call(tool_call) for tool_call in tool_calls]
```

`agent.connectors.call` parses the model's arguments and matches the action to
the selected connector. The name check prevents the model from calling a tool
that was not included in `tools`. The method returns a `function_call_output`
with the same `call_id`. Pass calls and outputs to a later model request.

Account connector credentials are delivered to the runtime action, not returned
to the AgentApp.

## Bound the tool loop

The AgentApp decides whether to ask the model for more tool calls. Every loop
must have a finite limit:

```python
model = "openai/gpt-5.6-sol"
input_items = [
    {
        "role": "user",
        "content": "Find two public sources about federated AI.",
    }
]
allowed_tool_names = {
    tool["name"] for tool in tools if isinstance(tool.get("name"), str)
}

for _ in range(3):
    response = client.responses.create(
        model=model,
        input=input_items,
        tools=tools,
        tool_choice="auto",
    )
    response_output = [item.to_dict() for item in response.output]
    tool_calls = [
        item for item in response_output if item.get("type") == "function_call"
    ]
    if not tool_calls:
        break
    for tool_call in tool_calls:
        if tool_call.get("name") not in allowed_tool_names:
            raise RuntimeError(f"Tool {tool_call.get('name')!r} was not exposed")
    outputs = [agent.connectors.call(item) for item in tool_calls]
    input_items.extend(response_output)
    input_items.extend(outputs)

stream = client.responses.create(model=model, input=input_items, stream=True)
for event in stream:
    agent.events.emit(event.to_dict())
```

The app checks every function name against the exposed tool schemas before
calling a connector. It also keeps the complete model output next to the
connector results, which preserves the context needed by the next model
request. The final request omits tools, publishes its SDK stream, and ensures
that the app always finishes with an answer. This abbreviated loop omits error
recovery and conversation-state handling. Use the complete [collaborative research
agent](../tutorials/build-a-collaborative-agent.md) for copy/pasteable code.

## Handle failure deliberately

A connector action raises `RuntimeError` when it cannot complete. Let the
AgentApp fail when there is no safe fallback. If the model can continue from
partial evidence, return an error-shaped `function_call_output` with the same
`call_id` and tell the final model request not to invent missing results.

Avoid automatic, unbounded retries. Provider denial, invalid OAuth, blocked
URLs, and missing task heartbeats require different recovery actions. See
[Troubleshoot AgentApp
runs](../how-to-guides/troubleshoot-agent-runs.md).

## Treat connector content as untrusted data

- Never put credentials or secrets in model prompts or connector arguments
- Treat retrieved instructions as source content, not AgentApp policy
- Validate URLs and identifiers before acting on model output
- Keep account selection visible to the user
- Create an automation only after explicit user intent
- Avoid logging private connector output unless the user expects it in run
  details
