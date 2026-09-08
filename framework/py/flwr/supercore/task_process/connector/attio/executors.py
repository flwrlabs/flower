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
"""Attio action executors."""

from urllib.parse import quote

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object

_ATTIO_API_BASE_URL = "https://api.attio.com/v2"


class AttioApiError(ConnectorApiError):
    """Secret-safe Attio API failure."""

    provider = "Attio"


def identify(arguments: JSONObject, context: ConnectorExecutionContext) -> JSONObject:
    """Identify the current Attio token and workspace."""
    del arguments
    return _call_attio_api("GET", "/self", context.credentials)


def get_workspace_member(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Get one member of the current Attio workspace."""
    workspace_member_id = _path_segment(
        arguments.get("workspace_member_id"), "workspace_member_id"
    )
    return _call_attio_api(
        "GET",
        f"/workspace_members/{workspace_member_id}",
        context.credentials,
    )


def search_records(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search records in one Attio workspace."""
    body: JSONObject = {
        name: arguments[name]
        for name in ("query", "objects", "limit", "request_as")
        if name in arguments
    }
    return _call_attio_api(
        "POST",
        "/objects/records/search",
        context.credentials,
        json_body=body,
    )


def list_meetings(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List meetings in one Attio workspace."""
    return _call_attio_api(
        "GET",
        "/meetings",
        context.credentials,
        params=_query_params(
            arguments,
            (
                "limit",
                "cursor",
                "linked_object",
                "linked_record_id",
                "participants",
                "sort",
                "ends_from",
                "starts_before",
                "timezone",
            ),
        ),
    )


def list_call_recordings(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List call recordings for one Attio meeting."""
    meeting_id = _path_segment(arguments.get("meeting_id"), "meeting_id")
    return _call_attio_api(
        "GET",
        f"/meetings/{meeting_id}/call_recordings",
        context.credentials,
        params=_query_params(arguments, ("limit", "cursor")),
    )


def get_call_transcript(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one page of transcript segments for an Attio call recording."""
    meeting_id = _path_segment(arguments.get("meeting_id"), "meeting_id")
    recording_id = _path_segment(
        arguments.get("call_recording_id"), "call_recording_id"
    )
    return _call_attio_api(
        "GET",
        f"/meetings/{meeting_id}/call_recordings/{recording_id}/transcript",
        context.credentials,
        params=_query_params(arguments, ("cursor",)),
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
    "identify": identify,
    "get_workspace_member": get_workspace_member,
    "search_records": search_records,
    "list_meetings": list_meetings,
    "list_call_recordings": list_call_recordings,
    "get_call_transcript": get_call_transcript,
}


def _call_attio_api(
    method: str,
    path: str,
    credentials: JSONObject,
    *,
    params: dict[str, str] | None = None,
    json_body: JSONObject | None = None,
) -> JSONObject:
    """Call one Attio REST endpoint."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise AttioApiError("invalid_credentials")
    return request_json_object(
        method,
        f"{_ATTIO_API_BASE_URL}{path}",
        error=AttioApiError,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        params=params or {},
        json=json_body,
        http_error_details=_response_error_details,
    )


def _path_segment(value: object, name: str) -> str:
    """Encode one Attio path segment without changing its value."""
    if not isinstance(value, str):
        raise TypeError(f"Attio {name} must be a string.")
    return quote(value, safe="")


def _query_params(arguments: JSONObject, names: tuple[str, ...]) -> dict[str, str]:
    """Serialize supplied Attio query values without renaming them."""
    return {
        name: str(arguments[name])
        for name in names
        if name in arguments and arguments[name] is not None
    }


def _response_error_details(response: requests.Response) -> tuple[str, str | None]:
    """Return Attio's documented error code and message without translation."""
    try:
        payload = response.json()
    except ValueError:
        return "http_error", None
    if not isinstance(payload, dict):
        return "http_error", None
    code = payload.get("code")
    message = payload.get("message")
    return (
        code if isinstance(code, str) and code else "http_error",
        message if isinstance(message, str) and message else None,
    )
