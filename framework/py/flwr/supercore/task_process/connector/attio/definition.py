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
# ===============================================================================
"""Attio connector definition."""

from ..definition import (
    ConnectorDefinition,
    ConnectorToolSource,
    OAuth2Definition,
    ProviderDefinition,
)
from ..oauth import load_oauth_flow
from .mcp import discover_tools, execute_tool

ATTIO_CONNECTOR_REF = "attio"


PROVIDER = ProviderDefinition(
    ref=ATTIO_CONNECTOR_REF,
    display_name="Attio",
    description="Use Attio tools available to the connected account.",
    actions=(),
    oauth=OAuth2Definition(
        authorization_url="https://app.attio.com/oidc/authorize",
        token_url="https://app.attio.com/oidc/token",
        client_id_env="FLWR_ATTIO_CLIENT_ID",
        client_secret_env="FLWR_ATTIO_CLIENT_SECRET",
        redirect_uri_env="FLWR_ATTIO_REDIRECT_URI",
        scopes=("mcp", "offline_access"),
        token_auth_method="client_secret_post",
        use_pkce=True,
        authorization_params={"resource": "https://mcp.attio.com/mcp"},
        token_params={"resource": "https://mcp.attio.com/mcp"},
        expected_token_type="bearer",
    ),
)

CONNECTOR = ConnectorDefinition.from_provider(
    provider=PROVIDER,
    executors={},
    oauth_flow=load_oauth_flow(PROVIDER),
    tool_source=ConnectorToolSource(discover=discover_tools, execute=execute_tool),
)
