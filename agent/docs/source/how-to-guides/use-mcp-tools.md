# Use tools from an MCP server

Connect a Flower AgentApp to tools exposed by a Model Context Protocol (MCP)
server. This integration currently runs inside your AgentApp; Flower does not
provide a first-class MCP client or register arbitrary MCP servers with
`agent.connectors`.

Flower connectors and MCP tools therefore follow different execution paths:

- `agent.connectors.tools(...)` and `agent.connectors.call(...)` use tools
  registered and hosted by the Flower runtime.
- An MCP client created by your AgentApp connects to the MCP server, discovers
  its tools, and executes its calls.

## Add an MCP client

Add the MCP Python SDK version used by Flower to the AgentApp dependencies:

```console
$ uv add 'mcp>=1.26.0,<2.0'
```

Bridge Flower's synchronous entry point to the asynchronous MCP client and keep
the client lifecycle inside context managers:

```python
import asyncio
import os

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

app = AgentApp()


async def run_mcp(agent: AgentSession, context: Context) -> None:
    async with streamable_http_client(os.environ["MCP_ENDPOINT"]) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as mcp_client:
            await mcp_client.initialize()
            await use_mcp_tools(mcp_client, agent, context)


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    asyncio.run(run_mcp(agent, context))
```

Set `MCP_ENDPOINT` to the URL of a running Streamable HTTP MCP server. The
following sections build `use_mcp_tools`.

## Discover and allowlist tools

Inside the asynchronous helper, list the server's tools and select an explicit
set of tool names:

```python
ALLOWED_MCP_TOOLS = {
    "search_documents",
    "read_document",
}


async def use_mcp_tools(
    mcp_client: ClientSession,
    agent: AgentSession,
    context: Context,
) -> None:
    list_result = await mcp_client.list_tools()
    selected_tools = [
        tool for tool in list_result.tools if tool.name in ALLOWED_MCP_TOOLS
    ]
    model_tools = [as_response_tool(tool) for tool in selected_tools]
    # Pass model_tools to the model and dispatch any returned tool calls.
```

The exact client setup and result container depend on the MCP library and
transport. The important boundary is that a server advertising a new tool does
not make that tool model-accessible automatically.

## Convert definitions to Open Responses tools

Define the schema converter at module scope. `use_mcp_tools` calls it while
`selected_tools` is still in scope:

```python
def as_response_tool(tool: object) -> dict[str, object]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }
```

Some MCP clients expose the schema as `input_schema` instead of `inputSchema`.
Use the field provided by your client without otherwise changing the JSON
Schema. Pass `model_tools` through the `tools` field of
`client.responses.create(...)`.

## Dispatch function calls back to MCP

When the OpenAI SDK returns a `function_call`, first convert it with
`function_call.to_dict()`. Then reject any name outside the allowlist, parse its
arguments, call the matching MCP tool, and create a `function_call_output` item
with the same `call_id`:

```python
import json


async def call_mcp_tool(
    mcp_client: ClientSession,
    tool_call: dict[str, object],
) -> dict[str, object]:
    name = tool_call.get("name")
    if name not in ALLOWED_MCP_TOOLS:
        raise RuntimeError(f"MCP tool {name!r} was not exposed")

    raw_arguments = tool_call.get("arguments", "{}")
    arguments = (
        json.loads(raw_arguments)
        if isinstance(raw_arguments, str)
        else raw_arguments
    )
    result = await mcp_client.call_tool(name, arguments)
    return {
        "type": "function_call_output",
        "call_id": tool_call["call_id"],
        "output": serialize_mcp_result(result),
    }
```

`serialize_mcp_result` represents application-specific normalization. Serialize
the MCP result's text, structured content, or error to a string accepted by your
model provider. Do not pass client-library objects directly to the model
request.

Append both the model's original function-call item and the corresponding
function output to the next request. Bound the loop exactly as you would for
Flower connectors; see {ref}`bound-connector-tool-loop`.

## Current limitations

Flower does not manage the MCP server, its credentials, or its activity events.
The AgentApp must close the client, handle failures and timeouts, and receive
secrets through the deployment rather than through `agent.input`.
