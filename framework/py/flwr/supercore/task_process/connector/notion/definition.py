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

from ..definition import ConnectorDefinition, OAuth2Definition, ProviderDefinition
from ..oauth import load_oauth_flow
from .actions import ACTIONS
from .executors import EXECUTORS, NOTION_API_VERSION

NOTION_CONNECTOR_REF = "notion"

PROVIDER = ProviderDefinition(
    ref=NOTION_CONNECTOR_REF,
    display_name="Notion",
    description="Search and read pages and data sources.",
    actions=ACTIONS,
    oauth=OAuth2Definition(
        authorization_url="https://api.notion.com/v1/oauth/authorize",
        token_url="https://api.notion.com/v1/oauth/token",
        client_id_env="FLWR_NOTION_CLIENT_ID",
        client_secret_env="FLWR_NOTION_CLIENT_SECRET",
        redirect_uri_env="FLWR_NOTION_REDIRECT_URI",
        authorization_params={"owner": "user"},
        token_request_format="json",
        token_headers={"Notion-Version": NOTION_API_VERSION},
        config_fields=("workspace_id", "workspace_name", "bot_id"),
    ),
    api_base_url="https://api.notion.com/v1",
    api_headers={"Notion-Version": NOTION_API_VERSION},
)

CONNECTOR = ConnectorDefinition(
    provider=PROVIDER,
    executors=EXECUTORS,
    oauth_flow=load_oauth_flow(PROVIDER),
)
