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

import re
from urllib.parse import quote
from uuid import UUID

import requests

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError, request_json_object
from ..json_utils import (
    ConnectorInputError,
    optional_cursor,
    optional_string,
    require_int_range,
    require_string,
)

_ATTIO_API_BASE_URL = "https://api.attio.com/v2"
_ATTIO_MEETING_SORTS = {"start_asc", "start_desc"}
_EMAIL_ADDRESS = re.compile(r"^[^@\s,]+@[^@\s,]+$")
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class AttioApiError(ConnectorApiError):
    """Secret-safe Attio API failure."""

    provider = "Attio"


def search_records(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search records in one Attio workspace."""
    objects = arguments.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ConnectorInputError("Attio objects must be a non-empty list.")
    return _call_attio_api(
        "POST",
        "/objects/records/search",
        context.credentials,
        json_body={
            "query": require_string(arguments.get("query"), "Attio", "query"),
            "objects": [require_string(item, "Attio", "object") for item in objects],
            "request_as": {"type": "workspace"},
            "limit": _limit(arguments, default=25, maximum=25),
        },
    )


def list_meetings(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List meetings in one Attio workspace."""
    linked_object = _optional(arguments, "linked_object")
    linked_record_id = _optional(arguments, "linked_record_id")
    if (linked_object is None) != (linked_record_id is None):
        raise ConnectorInputError(
            "Attio linked_object and linked_record_id must be provided together."
        )
    if linked_record_id is not None:
        linked_record_id = _uuid(linked_record_id, "linked_record_id")
    return _call_attio_api(
        "GET",
        "/meetings",
        context.credentials,
        params={
            "limit": str(_limit(arguments, default=50, maximum=200)),
            "cursor": optional_cursor(
                arguments.get("cursor"), "Attio", "pagination.next_cursor"
            ),
            "linked_object": linked_object,
            "linked_record_id": linked_record_id,
            "participants": _participants(arguments),
            "sort": _meeting_sort(arguments),
        },
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
        params={
            "limit": str(_limit(arguments, default=50, maximum=200)),
            "cursor": optional_cursor(
                arguments.get("cursor"), "Attio", "pagination.next_cursor"
            ),
        },
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
        params={
            "cursor": optional_cursor(
                arguments.get("cursor"), "Attio", "pagination.next_cursor"
            )
        },
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
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
    params: dict[str, str | None] | None = None,
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
        params={k: v for k, v in (params or {}).items() if v is not None},
        json=json_body,
        http_error_code=_response_error_code,
    )


def _path_segment(value: object, name: str) -> str:
    """Validate and encode one Attio path segment."""
    return quote(_uuid(require_string(value, "Attio", name), name), safe="")


def _limit(arguments: JSONObject, *, default: int, maximum: int) -> int:
    """Return one validated Attio page limit."""
    return require_int_range(
        arguments.get("limit", default), "Attio", "limit", maximum=maximum
    )


def _optional(arguments: JSONObject, name: str) -> str | None:
    """Return one optional Attio string argument."""
    return optional_string(arguments.get(name), "Attio", name)


def _meeting_sort(arguments: JSONObject) -> str:
    """Return a validated Attio meeting sort order."""
    sort = optional_string(arguments.get("sort"), "Attio", "sort") or "start_asc"
    if sort not in _ATTIO_MEETING_SORTS:
        raise ConnectorInputError("Attio sort must be 'start_asc' or 'start_desc'.")
    return sort


def _participants(arguments: JSONObject) -> str | None:
    """Return model-native email input in Attio's comma-separated format."""
    participants = arguments.get("participants")
    if participants is None:
        return None
    if isinstance(participants, str):
        # Retain compatibility with calls made before participants became an array.
        emails = [email.strip() for email in participants.split(",")]
    elif isinstance(participants, list):
        emails = [
            require_string(email, "Attio", "participant") for email in participants
        ]
    else:
        raise ConnectorInputError(
            "Attio participants must be a list of email addresses."
        )
    if not emails or any(_EMAIL_ADDRESS.fullmatch(email) is None for email in emails):
        raise ConnectorInputError(
            "Attio participants must contain full email addresses."
        )
    return ",".join(emails)


def _uuid(value: str, name: str) -> str:
    """Return one normalized Attio UUID argument."""
    try:
        return str(UUID(value))
    except ValueError:
        raise ConnectorInputError(f"Attio {name} must be a UUID.") from None


def _response_error_code(response: requests.Response) -> str:
    """Return Attio's structured error code when it is safe to expose."""
    try:
        code = response.json().get("code")
    except (AttributeError, ValueError):
        code = None
    if isinstance(code, str) and _SAFE_ERROR_CODE.fullmatch(code):
        return code
    return "rate_limited" if response.status_code == 429 else "http_error"
