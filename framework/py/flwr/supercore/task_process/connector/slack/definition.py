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
"""Slack connector definition."""

from ..definition import ConnectorDefinition, OAuth2Definition, ProviderDefinition
from ..oauth import load_declarative_oauth_provider
from .actions import ACTIONS
from .executors import EXECUTORS

SLACK_CONNECTOR_REF = "slack"
SLACK_USER_SCOPES = (
    "search:read",
    "channels:read",
    "groups:read",
    "im:read",
    "mpim:read",
    "channels:history",
    "groups:history",
    "im:history",
    "mpim:history",
)

PROVIDER = ProviderDefinition(
    ref=SLACK_CONNECTOR_REF,
    display_name="Slack",
    description="Search and read messages, conversations, and threads.",
    actions=ACTIONS,
    oauth=OAuth2Definition(
        authorization_url="https://slack.com/oauth/v2/authorize",
        token_url="https://slack.com/api/oauth.v2.access",
        client_id_env="FLWR_SLACK_CLIENT_ID",
        client_secret_env="FLWR_SLACK_CLIENT_SECRET",
        redirect_uri_env="FLWR_SLACK_REDIRECT_URI",
        scopes=SLACK_USER_SCOPES,
        scope_parameter="user_scope",
        scope_separator=",",
        token_response_path=("authed_user",),
        success_field="ok",
    ),
)

CONNECTOR = ConnectorDefinition(
    provider=PROVIDER,
    executors=EXECUTORS,
    oauth_provider=load_declarative_oauth_provider(PROVIDER),
)
