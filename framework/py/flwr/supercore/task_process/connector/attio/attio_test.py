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
    """Attio should expose four account-scoped read actions."""
    assert len(ACTIONS) == 4
    assert all(action.access is ActionAccess.READ for action in ACTIONS)
    tools = registry.get_connector_tools(ATTIO_CONNECTOR_REF)
    assert [tool["name"] for tool in tools] == [
        f"{ATTIO_CONNECTOR_REF}_{action.name}" for action in ACTIONS
    ]
    meeting_parameters = tools[1]["parameters"]
    assert isinstance(meeting_parameters, dict)
    meeting_properties = meeting_parameters["properties"]
    assert isinstance(meeting_properties, dict)
    participants = meeting_properties["participants"]
    linked_record_id = meeting_properties["linked_record_id"]
    assert isinstance(participants, dict)
    assert isinstance(linked_record_id, dict)
    assert participants["type"] == "array"
    participant_items = participants["items"]
    assert isinstance(participant_items, dict)
    assert participant_items["format"] == "email"
    assert linked_record_id["format"] == "uuid"


def test_search_records_calls_attio() -> None:
    """Record search should pass the documented request to Attio."""
    response = _response({"data": []})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = _invoke(
            "attio_search_records", {"query": "Flower", "objects": ["companies"]}
        )

    assert request.call_args.args == (
        "POST",
        "https://api.attio.com/v2/objects/records/search",
    )
    assert request.call_args.kwargs["json"] == {
        "query": "Flower",
        "objects": ["companies"],
        "request_as": {"type": "workspace"},
        "limit": 25,
    }
    assert result == {"data": []}


def test_list_meetings_forwards_validated_filters_and_sort() -> None:
    """Meeting reads should forward documented Attio query formats."""
    response = _response({"data": [], "pagination": {"next_cursor": None}})
    with patch(_HTTP_REQUEST, return_value=response) as request:
        result = _invoke(
            "attio_list_meetings",
            {
                "limit": 1,
                "linked_object": "people",
                "linked_record_id": "CB59AB17-AD15-460C-A126-0715617C0853",
                "participants": ["ada@example.com", " grace@example.com "],
                "sort": "start_desc",
            },
        )

    assert result == response.json.return_value
    assert request.call_args.kwargs["params"] == {
        "limit": "1",
        "linked_object": "people",
        "linked_record_id": "cb59ab17-ad15-460c-a126-0715617c0853",
        "participants": "ada@example.com,grace@example.com",
        "sort": "start_desc",
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {"linked_object": "people"},
        {"linked_record_id": "cb59ab17-ad15-460c-a126-0715617c0853"},
        {
            "linked_object": "people",
            "linked_record_id": "not-a-uuid",
        },
        {"participants": "Ada Lovelace"},
        {"participants": "example.com"},
        {"participants": "me"},
        {"participants": []},
        {"sort": "latest"},
    ],
)
def test_list_meetings_rejects_invalid_filters(arguments: JSONObject) -> None:
    """Invalid meeting filters should fail before reaching Attio."""
    with patch(_HTTP_REQUEST) as request, pytest.raises(ValueError):
        _invoke("attio_list_meetings", arguments)

    request.assert_not_called()


def test_api_errors_are_secret_safe() -> None:
    """API errors should not expose tokens or response text."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response(
                {"message": "attio-secret"},
                status_code=401,
            ),
        ),
        pytest.raises(AttioApiError) as error,
    ):
        _invoke("attio_search_records", {"query": "Flower", "objects": ["companies"]})

    assert error.value.code == "http_error"
    assert "attio-secret" not in str(error.value)


def test_api_errors_return_safe_attio_code() -> None:
    """Attio's structured code should be useful without exposing its message."""
    with (
        patch(
            _HTTP_REQUEST,
            return_value=_response(
                {
                    "code": "invalid_query",
                    "message": "attio-secret",
                },
                status_code=400,
            ),
        ),
        pytest.raises(AttioApiError) as error,
    ):
        _invoke("attio_search_records", {"query": "Flower", "objects": ["people"]})

    assert error.value.code == "invalid_query"
    assert "attio-secret" not in str(error.value)


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
