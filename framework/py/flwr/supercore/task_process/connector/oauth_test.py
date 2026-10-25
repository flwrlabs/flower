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
"""Tests for shared OAuth token handling."""

from unittest.mock import Mock, patch

import pytest

from flwr.supercore.typing import JSONObject

from .definition import ProviderDefinition
from .github.definition import PROVIDER as GITHUB_PROVIDER
from .notion.definition import PROVIDER as NOTION_PROVIDER
from .oauth import OAuthFlow
from .slack.definition import PROVIDER as SLACK_PROVIDER


def _flow(provider: ProviderDefinition = NOTION_PROVIDER) -> OAuthFlow:
    return OAuthFlow(
        provider,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )


def test_exchange_records_absolute_expiry() -> None:
    """Record expiry on initial authorization so the first task can reuse its token."""
    response = Mock(status_code=200)
    response.json.return_value = {
        "access_token": "token",
        "refresh_token": "refresh",
        "token_type": "bearer",
        "expires_in": 3600,
    }
    with (
        patch("flwr.supercore.task_process.connector.oauth.time", return_value=1000),
        patch(
            "flwr.supercore.task_process.connector.oauth.requests.post",
            return_value=response,
        ),
    ):
        credentials, _ = _flow().exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )

    assert credentials == {
        "access_token": "token",
        "refresh_token": "refresh",
        "token_type": "bearer",
        "expires_in": 3600,
        "expires_at": 4600,
    }


@pytest.mark.parametrize(
    ("provider", "token_type"),
    [
        (GITHUB_PROVIDER, "bearer"),
        (NOTION_PROVIDER, "bearer"),
        (SLACK_PROVIDER, "user"),
    ],
    ids=["github", "notion", "slack"],
)
def test_refresh_preserves_replacement_and_omitted_refresh_tokens(
    provider: ProviderDefinition,
    token_type: str,
) -> None:
    """Keep the replacement token when the following response omits one."""
    flow = _flow(provider)
    response = Mock(status_code=200)
    response.json.side_effect = [
        {
            "ok": True,
            "access_token": "first-token",
            "refresh_token": "rotated-token",
            "token_type": token_type,
            "expires_in": 3600,
        },
        {"ok": True, "access_token": "second-token", "token_type": token_type},
    ]
    with (
        patch(
            "flwr.supercore.task_process.connector.oauth.time", return_value=1000
        ) as clock,
        patch(
            "flwr.supercore.task_process.connector.oauth.requests.post",
            return_value=response,
        ),
    ):
        credentials = flow.refresh_credentials(
            {"refresh_token": "original-token", "expires_at": 0}
        )
        clock.return_value = 4540
        credentials = flow.refresh_credentials(credentials)

    assert credentials == {
        "access_token": "second-token",
        "refresh_token": "rotated-token",
        "token_type": token_type,
    }


def test_credentials_without_expiry_are_reused() -> None:
    """Do not refresh tokens whose provider does not report an expiry."""
    credentials: JSONObject = {"access_token": "token", "refresh_token": "refresh"}
    with patch("flwr.supercore.task_process.connector.oauth.requests.post") as post:
        assert _flow().refresh_credentials(credentials) == credentials
    post.assert_not_called()
