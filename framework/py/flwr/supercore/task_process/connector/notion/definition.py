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
"""Notion connector definition."""

from ..definition import (
    ConnectorDefinition,
    ConnectorToolSource,
    OAuth2Definition,
    ProviderDefinition,
)
from ..oauth import load_oauth_flow
from .mcp import discover_tools, execute_tool

NOTION_CONNECTOR_REF = "notion"

PROVIDER = ProviderDefinition(
    ref=NOTION_CONNECTOR_REF,
    display_name="Notion",
    description="Use Notion tools available to the connected workspace.",
    actions=(),
    oauth=OAuth2Definition(
        authorization_url="https://mcp.notion.com/authorize",
        token_url="https://mcp.notion.com/token",
        client_id_env="FLWR_NOTION_CLIENT_ID",
        client_secret_env="FLWR_NOTION_CLIENT_SECRET",
        redirect_uri_env="FLWR_NOTION_REDIRECT_URI",
        scopes=("default",),
        token_auth_method="client_secret_post",
        authorization_params={"resource": "https://mcp.notion.com/mcp"},
        token_params={"resource": "https://mcp.notion.com/mcp"},
        token_headers={"Accept": "application/json"},
        use_pkce=True,
        expected_token_type="bearer",
        config_fields=(
            "user_id",
            "workspace_id",
            "email_domain",
            "bot_id",
            "workspace_name",
            "workspace_icon",
        ),
        display_name_fields=("workspace_name",),
    ),
)


CONNECTOR = ConnectorDefinition.from_provider(
    provider=PROVIDER,
    executors={},
    oauth_flow=load_oauth_flow(PROVIDER),
    tool_source=ConnectorToolSource(discover=discover_tools, execute=execute_tool),
)
