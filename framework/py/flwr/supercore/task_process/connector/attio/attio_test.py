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
"""Tests for the Attio connector."""

from typing import cast
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from flwr.supercore.typing import JSONObject, JSONValue

from .. import registry
from ..definition import ActionAccess
from ..oauth import OAuthFlow
from .actions import ACTIONS
from .definition import ATTIO_CONNECTOR_REF, PROVIDER
from .executors import AttioApiError

_HTTP_REQUEST = "flwr.supercore.task_process.connector.http.requests.request"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"
_CREDENTIALS: JSONObject = {"access_token": "attio-secret"}
_REDIRECT_URI = "https://client.example/oauth/attio"


def _response(payload: object, status_code: int = 200) -> Mock:
    """Return a minimal HTTP response mock."""
    response = Mock(status_code=status_code)
    response.json.return_value = payload
    return response


def _flow() -> OAuthFlow:
    return OAuthFlow(
        PROVIDER,
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )


def _invoke(name: str, arguments: JSONObject) -> JSONValue:
    return registry.invoke_connector(name, arguments, Mock(), _CREDENTIALS, {})


def test_attio_actions_are_registered_as_read_only() -> None:
    """Attio should expose six account-scoped read actions."""
    assert [action.name for action in ACTIONS] == [
        "identify",
        "get_workspace_member",
        "search_records",
        "list_meetings",
        "list_call_recordings",
        "get_call_transcript",
    ]
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    tools = registry.get_connector_tools(ATTIO_CONNECTOR_REF)
    assert [tool["name"] for tool in tools] == [
        f"{ATTIO_CONNECTOR_REF}_{action.name}" for action in ACTIONS
    ]
    schemas = {action.name: action.input_schema for action in ACTIONS}
    search_schema = schemas["search_records"]
    assert "request_as" in cast(list[JSONValue], search_schema["required"])
    meeting_properties = cast(JSONObject, schemas["list_meetings"]["properties"])
    assert cast(JSONObject, meeting_properties["participants"])["type"] == "string"
    for schema in schemas.values():
        properties = cast(JSONObject, schema["properties"])
        if limit := properties.get("limit"):
            assert "maximum" not in cast(JSONObject, limit)


def test_attio_actions_forward_requests() -> None:
    """Attio actions should forward inputs without semantic rewriting."""
    member_id = "CB59AB17-AD15-460C-A126-0715617C0853"
    search: JSONObject = {
        "query": "",
        "objects": ["people"],
        "limit": 26,
        "request_as": {"type": "workspace"},
    }
    meetings: JSONObject = {
        "limit": 201,
        "cursor": "0",
        "linked_record_id": "me",
        "participants": "me",
        "sort": "latest",
        "ends_from": "2026-08-01T00:00:00Z",
        "starts_before": "2026-09-01T00:00:00Z",
        "timezone": "Europe/Berlin",
    }
    cases: list[tuple[str, JSONObject, str, str, JSONObject]] = [
        ("attio_identify", {}, "GET", "/self", {"params": {}}),
        (
            "attio_get_workspace_member",
            {"workspace_member_id": member_id},
            "GET",
            f"/workspace_members/{member_id}",
            {"params": {}},
        ),
        (
            "attio_search_records",
            search,
            "POST",
            "/objects/records/search",
            {"json": search},
        ),
        (
            "attio_list_meetings",
            meetings,
            "GET",
            "/meetings",
            {"params": {key: str(value) for key, value in meetings.items()}},
        ),
        ("attio_list_meetings", {}, "GET", "/meetings", {"params": {}}),
    ]
    response = _response({"data": []})
    for name, arguments, method, path, expected_kwargs in cases:
        with patch(_HTTP_REQUEST, return_value=response) as request:
            assert _invoke(name, arguments) == {"data": []}

        assert request.call_args.args == (method, f"https://api.attio.com/v2{path}")
        for key, value in expected_kwargs.items():
            assert request.call_args.kwargs[key] == value


def test_api_errors_include_attio_code_and_message() -> None:
    """Attio's documented error fields should remain readable to callers."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response(
                {
                    "code": "invalid_query",
                    "message": "Participants must be email addresses",
                },
                status_code=400,
            ),
        ),
        pytest.raises(AttioApiError) as error,
    ):
        _invoke(
            "attio_search_records",
            {
                "query": "Flower",
                "objects": ["people"],
                "request_as": {"type": "workspace"},
            },
        )

    assert error.value.code == "invalid_query"
    assert str(error.value) == (
        "Attio API request failed: invalid_query (400): "
        "Participants must be email addresses."
    )


def test_oauth_builds_url_and_exchanges_code() -> None:
    """OAuth should validate redirects and return Attio credentials."""
    url = _flow().build_authorization_url(
        redirect_uri=_REDIRECT_URI,
        state="oauth-state",
        pkce_challenge="shared-pkce-challenge",
    )
    query = parse_qs(urlparse(url).query)
    assert query["redirect_uri"] == [_REDIRECT_URI]
    assert "code_challenge" not in query
    with pytest.raises(ValueError):
        _flow().resolve_redirect_uri("https://attacker.example/callback")

    response = _response({"access_token": "attio-access"})
    with patch(_TOKEN_REQUEST, return_value=response) as post:
        credentials, config = _flow().exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="shared-pkce-verifier",
        )

    assert credentials == {"access_token": "attio-access"}
    assert not config
    assert post.call_args.kwargs["data"] == {
        "client_id": "client-id",
        "client_secret": "client-secret",
        "grant_type": "authorization_code",
        "code": "authorization-code",
        "redirect_uri": _REDIRECT_URI,
    }
    with pytest.raises(ValueError):
        _flow().exchange_code(
            code="authorization-code",
            redirect_uri="https://attacker.example/callback",
            pkce_verifier="shared-pkce-verifier",
        )
