# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Attio MCP discovery, execution, and OAuth."""

import traceback
from collections.abc import Iterator
from typing import cast
from unittest.mock import Mock, call, create_autospec, patch
from urllib.parse import parse_qs, urlparse

import pytest
from mcp import ClientSession, types

from flwr.supercore.typing import JSONObject

from .. import registry
from ..oauth import OAuthFlow
from .definition import PROVIDER
from .mcp import AttioApiError

_CREDENTIALS: JSONObject = {"access_token": "attio-secret"}


@pytest.fixture(name="session")
def fixture_session() -> Iterator[Mock]:
    """Mock the MCP session and transport while retaining connector error handling."""
    session = create_autospec(ClientSession, instance=True)
    session.__aenter__.return_value = session
    with (
        patch("mcp.ClientSession", return_value=session),
        patch("mcp.client.streamable_http.streamable_http_client") as transport,
    ):
        transport.return_value.__aenter__.return_value = (Mock(), Mock(), Mock())
        yield session


def test_mcp_discovery_and_execution(session: Mock) -> None:
    """Preserve native tool names, schemas, and results across paginated discovery."""
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    session.list_tools.side_effect = [
        types.ListToolsResult(
            tools=[
                types.Tool(
                    name="search-records",
                    description="Search records",
                    inputSchema=schema,
                )
            ],
            nextCursor="next",
        ),
        types.ListToolsResult(tools=[types.Tool(name="whoami", inputSchema={})]),
    ]
    assert not registry.get_connector_tools("attio")
    tools = registry.invoke_connector(
        "attio__discover_tools", {}, Mock(), _CREDENTIALS, {}
    )
    assert isinstance(tools, list)
    assert [cast(JSONObject, tool)["name"] for tool in tools] == [
        "attio_search-records",
        "attio_whoami",
    ]
    assert cast(JSONObject, tools[0])["description"] == "Search records"
    assert cast(JSONObject, tools[0])["parameters"] == schema

    output = types.CallToolResult(
        content=[types.TextContent(type="text", text="done")],
        structuredContent={"count": 1},
    )
    session.send_request.return_value = output
    arguments: JSONObject = {"query": "Flower"}
    result = registry.invoke_connector(
        "attio_search-records", arguments, Mock(), _CREDENTIALS, {}
    )
    assert result == output.model_dump(mode="json", by_alias=True, exclude_none=True)
    session.send_request.assert_awaited_once()
    request = session.send_request.await_args.args[0].root
    assert request.params.name == "search-records"
    assert request.params.arguments == arguments
    assert session.list_tools.await_args_list == [
        call(cursor=None),
        call(cursor="next"),
    ]


def test_mcp_tool_error_redacts_credentials(session: Mock) -> None:
    """Keep tool diagnostics without exposing the token in the exception traceback."""
    session.send_request.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="Denied attio-secret")],
        isError=True,
    )
    with pytest.raises(AttioApiError) as error:
        registry.invoke_connector("attio_whoami", {}, Mock(), _CREDENTIALS, {})
    assert error.value.code == "tool_error"
    assert error.value.message == "Denied [REDACTED]"
    assert "attio-secret" not in "".join(traceback.format_exception(error.value))


def test_oauth_uses_mcp_endpoints_and_pkce() -> None:
    """Target Attio's MCP resource during authorization, exchange, and refresh."""
    redirect_uri = "https://client.example/oauth/attio"
    flow = OAuthFlow(
        PROVIDER,
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=redirect_uri,
    )
    url = urlparse(
        flow.build_authorization_url(
            redirect_uri=redirect_uri, state="state", pkce_challenge="challenge"
        )
    )
    assert f"{url.scheme}://{url.netloc}{url.path}" == (
        "https://app.attio.com/oidc/authorize"
    )
    query = parse_qs(url.query)
    assert query["resource"] == ["https://mcp.attio.com/mcp"]
    assert query["scope"] == ["mcp offline_access"]
    assert query["code_challenge"] == ["challenge"]
    assert query["code_challenge_method"] == ["S256"]

    response = Mock(status_code=200)
    response.json.return_value = {"access_token": "mcp-access", "token_type": "Bearer"}
    with patch(
        "flwr.supercore.task_process.connector.oauth.requests.post",
        return_value=response,
    ) as post:
        credentials, _ = flow.exchange_code(
            code="code", redirect_uri=redirect_uri, pkce_verifier="verifier"
        )
        assert post.call_args.args == ("https://app.attio.com/oidc/token",)
        assert post.call_args.kwargs["data"]["code_verifier"] == "verifier"
        assert post.call_args.kwargs["data"]["resource"] == "https://mcp.attio.com/mcp"

        flow.refresh_credentials({"refresh_token": "refresh", "expires_at": 0})
        assert post.call_count == 2
        assert post.call_args.kwargs["data"]["grant_type"] == "refresh_token"
        assert post.call_args.kwargs["data"]["resource"] == "https://mcp.attio.com/mcp"
    assert credentials["access_token"] == "mcp-access"
