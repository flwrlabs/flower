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

Add the MCP Python SDK to the AgentApp dependencies. This example targets its
1.x API:

```console
$ uv add 'mcp>=1.26.0,<2.0'
```

## Implement the AgentApp

The following AgentApp performs one bounded MCP tool round and then streams the
final model response:

```python
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, cast

from flwr.agentapp import AgentApp, AgentSession
from flwr.app import Context
from mcp import CallToolResult, ClientSession, Tool
from mcp.client.streamable_http import streamable_http_client
from mcp.types import PaginatedRequestParams
from openai import AsyncOpenAI

MODEL = "openai/gpt-5.6-sol"
ALLOWED_MCP_TOOLS = {"search_documents", "read_document"}

app = AgentApp()


def as_response_tool(tool: Tool) -> dict[str, Any]:
    """Convert an MCP tool definition to an Open Responses function tool."""
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description or "",
        "parameters": tool.inputSchema,
    }


def serialize_mcp_result(result: CallToolResult) -> str:
    """Serialize an MCP result for a function_call_output item."""
    return result.model_dump_json(by_alias=True, exclude_none=True)


async def list_all_tools(mcp_client: ClientSession) -> list[Tool]:
    """List every tool exposed by a paginated MCP server."""
    tools = []
    cursor = None
    while True:
        params = (
            PaginatedRequestParams(cursor=cursor)
            if cursor is not None
            else None
        )
        page = await mcp_client.list_tools(params=params)
        tools.extend(page.tools)
        cursor = page.nextCursor
        if cursor is None:
            return tools


async def call_mcp_tool(
    mcp_client: ClientSession,
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    """Validate and execute one model-requested MCP tool call."""
    name = tool_call.get("name")
    if not isinstance(name, str) or name not in ALLOWED_MCP_TOOLS:
        raise RuntimeError(f"MCP tool {name!r} was not exposed")

    raw_arguments = tool_call.get("arguments", "{}")
    arguments = (
        json.loads(raw_arguments)
        if isinstance(raw_arguments, str)
        else raw_arguments
    )
    if not isinstance(arguments, dict):
        raise ValueError("MCP tool arguments must be a JSON object")

    call_id = tool_call.get("call_id")
    if not isinstance(call_id, str):
        raise ValueError("MCP tool call requires a string call_id")

    result = await mcp_client.call_tool(name, arguments)
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": serialize_mcp_result(result),
    }


async def run_mcp(agent: AgentSession, context: Context) -> None:
    """Discover MCP tools, execute one tool round, and publish the answer."""
    prompt = context.run_config.get("agent.input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("agent.input must be a non-empty string")

    async with AsyncOpenAI(
        base_url=os.environ["FLWR_RUNTIME_BASE_URL"],
        api_key=os.environ["FLWR_RUNTIME_API_KEY"],
        max_retries=0,
    ) as client:
        input_items: list[dict[str, Any]] = [
            {"type": "message", "role": "user", "content": prompt.strip()}
        ]

        async with streamable_http_client(os.environ["MCP_ENDPOINT"]) as (
            read,
            write,
            _,
        ):
            async with ClientSession(read, write) as mcp_client:
                await mcp_client.initialize()
                selected_tools = [
                    tool
                    for tool in await list_all_tools(mcp_client)
                    if tool.name in ALLOWED_MCP_TOOLS
                ]
                if not selected_tools:
                    raise RuntimeError("The MCP server exposed no allowed tools")
                model_tools = [as_response_tool(tool) for tool in selected_tools]

                response = await client.responses.create(
                    model=MODEL,
                    input=input_items,
                    tools=model_tools,
                    tool_choice="required",
                )
                response_output = [
                    cast(dict[str, Any], item.to_dict())
                    for item in response.output
                ]
                tool_calls = [
                    item
                    for item in response_output
                    if item.get("type") == "function_call"
                ]
                if not tool_calls:
                    raise RuntimeError("The model did not request an MCP tool")

                function_outputs = [
                    await call_mcp_tool(mcp_client, tool_call)
                    for tool_call in tool_calls
                ]
                input_items.extend(response_output)
                input_items.extend(function_outputs)

        stream = await client.responses.create(
            model=MODEL,
            input=input_items,
            stream=True,
        )
        output_text = []
        async for event in stream:
            agent.events.emit(event.to_dict())
            if event.type in {
                "error",
                "response.failed",
                "response.incomplete",
            }:
                raise RuntimeError(f"Model response did not complete: {event}")
            if event.type in {
                "response.output_text.delta",
                "response.refusal.delta",
            }:
                output_text.append(event.delta)

        print("".join(output_text))


@app.main()
def main(agent: AgentSession, context: Context) -> None:
    """Run the asynchronous MCP workflow from Flower's synchronous entry point."""
    asyncio.run(run_mcp(agent, context))
```

Set `MCP_ENDPOINT` to a running Streamable HTTP MCP server and replace the model
and allowlist with values for your deployment. Only allowlisted MCP definitions
are exposed to the model, and the final request omits tools so the workflow
cannot start another tool round.

## Current limitations

Flower does not manage the MCP server, its credentials, or its activity events.
The AgentApp must close the client, handle failures and timeouts, and receive
secrets through the deployment rather than through `agent.input`.
