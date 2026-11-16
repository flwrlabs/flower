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
"""Google Calendar connector definition."""

from ..definition import (
    ConnectorDefinition,
    ConnectorToolSource,
    OAuth2Definition,
    ProviderDefinition,
)
from ..oauth import load_oauth_flow
from .mcp import GoogleMcpClient

GOOGLE_CALENDAR_CONNECTOR_REF = "google_calendar"

OAUTH = OAuth2Definition(
    authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
    token_url="https://oauth2.googleapis.com/token",
    client_id_env="FLWR_GOOGLE_CALENDAR_CLIENT_ID",
    client_secret_env="FLWR_GOOGLE_CALENDAR_CLIENT_SECRET",
    redirect_uri_env="FLWR_GOOGLE_CALENDAR_REDIRECT_URI",
    scopes=(
        "https://www.googleapis.com/auth/calendar.calendarlist.readonly",
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/calendar.events.freebusy",
        "https://www.googleapis.com/auth/userinfo.email",
    ),
    token_auth_method="client_secret_post",
    use_pkce=True,
    authorization_params={"access_type": "offline", "prompt": "consent select_account"},
    config_fields=("scope",),
    expected_token_type="Bearer",
    display_name_url="https://openidconnect.googleapis.com/v1/userinfo",
    display_name_fields=("email",),
)

PROVIDER = ProviderDefinition(
    ref=GOOGLE_CALENDAR_CONNECTOR_REF,
    display_name="Google Calendar",
    description="Read Google calendars, find availability, and manage events.",
    actions=(),
    oauth=OAUTH,
)

CLIENT = GoogleMcpClient(url="https://calendarmcp.googleapis.com/mcp/v1")

CONNECTOR = ConnectorDefinition.from_provider(
    provider=PROVIDER,
    executors={},
    oauth_flow=load_oauth_flow(PROVIDER),
    tool_source=ConnectorToolSource(
        discover=CLIENT.discover_tools, execute=CLIENT.execute_tool
    ),
)
