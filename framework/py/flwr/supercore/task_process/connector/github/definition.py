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
"""GitHub connector definition."""

from ..definition import ConnectorDefinition, OAuth2Definition, ProviderDefinition
from ..oauth import load_oauth_flow
from .actions import ACTIONS
from .executors import EXECUTORS

GITHUB_CONNECTOR_REF = "github"

PROVIDER = ProviderDefinition(
    ref=GITHUB_CONNECTOR_REF,
    display_name="GitHub",
    description="Search code and read files in public repositories.",
    actions=ACTIONS,
    oauth=OAuth2Definition(
        authorization_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        client_id_env="FLWR_GITHUB_CLIENT_ID",
        client_secret_env="FLWR_GITHUB_CLIENT_SECRET",
        redirect_uri_env="FLWR_GITHUB_REDIRECT_URI",
        token_auth_method="client_secret_post",
        token_headers={"Accept": "application/json"},
        use_pkce=True,
        allow_additional_scopes=False,
        expected_token_type="bearer",
    ),
)

CONNECTOR = ConnectorDefinition(
    provider=PROVIDER,
    executors=EXECUTORS,
    oauth_flow=load_oauth_flow(PROVIDER),
)
