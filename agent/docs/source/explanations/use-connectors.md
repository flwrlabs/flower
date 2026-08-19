# Use connectors

Connectors give an AgentApp runtime-provided tools without embedding provider
implementations or credentials in the app.

## Distinguish built-in and account connectors

Built-in connectors need no external account:

| Reference          | Capability                        | Important boundary                           |
| ------------------ | --------------------------------- | -------------------------------------------- |
| `web_search`       | Search the public web             | Results depend on runtime availability       |
| `web_fetch`        | Fetch eligible public web content | Private and unsafe targets are blocked       |
| `browser_use`      | Run a browser task                | Availability depends on the deployment       |
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
response = agent.responses.create(
    {
        "model": "openai/gpt-5.6-sol",
        "input": "Find two public sources about federated AI.",
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
    }
)
```

The model can return normal output, one function call, or several independent
function calls in the `output` list.

## Execute requested calls

```python
tool_calls = [
    dict(item)
    for item in response.get("output", [])
    if isinstance(item, dict) and item.get("type") == "function_call"
]

function_outputs = [agent.connectors.call(tool_call) for tool_call in tool_calls]
```

`agent.connectors.call` parses the model's arguments and matches the action to
the selected connector. It returns a `function_call_output` with the same
`call_id`. Pass calls and outputs to a later model request.

Account connector credentials are delivered to the runtime action, not returned
to the AgentApp.

## Bound the tool loop

The AgentApp decides whether to ask the model for more tool calls. Every loop
must have a finite limit:

```python
request = {
    "model": "openai/gpt-5.6-sol",
    "input": [
        {
            "role": "user",
            "content": "Find two public sources about federated AI.",
        }
    ],
    "tools": tools,
    "tool_choice": "auto",
    "stream": False,
}

for _ in range(3):
    response = agent.responses.create(request)
    tool_calls = [
        dict(item)
        for item in response.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    if not tool_calls:
        break
    outputs = [agent.connectors.call(item) for item in tool_calls]
    request["input"] = [*request["input"], *tool_calls, *outputs]

agent.responses.create(
    {
        "model": request["model"],
        "input": request["input"],
        "stream": True,
    }
)
```

The final request does not include tools, so the model must produce an answer
after consuming the available connector outputs. This abbreviated loop omits
recovery and conversation-state handling. Use the complete [collaborative
research agent](../tutorials/build-a-collaborative-agent.md) for copy/pasteable
code.

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
