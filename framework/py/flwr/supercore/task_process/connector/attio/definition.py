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

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorDefinition, OAuth2Definition, ProviderDefinition
from ..oauth import load_oauth_flow
from .actions import ACTIONS
from .executors import EXECUTORS

ATTIO_CONNECTOR_REF = "attio"


def _attio_display_name(payload: JSONObject) -> str:
    """Prefer workspace names, falling back to the token's workspace ID."""
    data = payload.get("data")
    workspace = data.get("workspace") if isinstance(data, dict) else None
    if isinstance(workspace, dict):
        name = workspace.get("name") or workspace.get("id")
    else:
        name = payload.get("workspace_name") or payload.get("sub")
    return name.strip() if isinstance(name, str) else ""


PROVIDER = ProviderDefinition(
    ref=ATTIO_CONNECTOR_REF,
    display_name="Attio",
    description="Identify workspace members, search records, and read meetings.",
    actions=ACTIONS,
    oauth=OAuth2Definition(
        authorization_url="https://app.attio.com/authorize",
        token_url="https://app.attio.com/oauth/token",
        client_id_env="FLWR_ATTIO_CLIENT_ID",
        client_secret_env="FLWR_ATTIO_CLIENT_SECRET",
        redirect_uri_env="FLWR_ATTIO_REDIRECT_URI",
        token_auth_method="client_secret_post",
        display_name_url="https://api.attio.com/v2/self",
        display_name_resolver=_attio_display_name,
    ),
)

CONNECTOR = ConnectorDefinition.from_provider(
    provider=PROVIDER,
    executors=EXECUTORS,
    oauth_flow=load_oauth_flow(PROVIDER),
)
