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
"""Read-only Attio connector tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from urllib.parse import quote

import requests

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

ATTIO_CONNECTOR_REF = "attio"
ATTIO_SEARCH_RECORDS_TOOL = "attio_search_records"
ATTIO_LIST_MEETINGS_TOOL = "attio_list_meetings"
ATTIO_LIST_CALL_RECORDINGS_TOOL = "attio_list_call_recordings"
ATTIO_GET_CALL_TRANSCRIPT_TOOL = "attio_get_call_transcript"
ATTIO_TOOL_NAMES = (
    ATTIO_SEARCH_RECORDS_TOOL,
    ATTIO_LIST_MEETINGS_TOOL,
    ATTIO_LIST_CALL_RECORDINGS_TOOL,
    ATTIO_GET_CALL_TRANSCRIPT_TOOL,
)

ATTIO_API_BASE_URL = "https://api.attio.com/v2"

_REQUEST_TIMEOUT = 30.0


class AttioApiError(RuntimeError):
    """Secret-safe Attio API failure."""

    def __init__(self, code: str, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        detail = code if status_code is None else f"{code} ({status_code})"
        super().__init__(f"Attio API request failed: {detail}.")


def make_attio_tools() -> list[JSONObject]:
    """Return model-facing schemas for Attio's read-only operations."""
    page: JSONObject = {
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "cursor": {"type": "string"},
    }
    return [
        _tool(
            ATTIO_SEARCH_RECORDS_TOOL,
            "Search records in Attio.",
            {
                "query": {"type": "string"},
                "objects": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25},
            },
            required=("query", "objects"),
        ),
        _tool(
            ATTIO_LIST_MEETINGS_TOOL,
            "List meetings in Attio.",
            {
                **page,
                "linked_object": {"type": "string"},
                "linked_record_id": {"type": "string"},
                "participants": {"type": "string"},
            },
        ),
        _tool(
            ATTIO_LIST_CALL_RECORDINGS_TOOL,
            "List call recordings for an Attio meeting.",
            {"meeting_id": {"type": "string"}, **page},
            required=("meeting_id",),
        ),
        _tool(
            ATTIO_GET_CALL_TRANSCRIPT_TOOL,
            "Read a call transcript from Attio.",
            {
                "meeting_id": {"type": "string"},
                "call_recording_id": {"type": "string"},
                **page,
            },
            required=("meeting_id", "call_recording_id"),
        ),
    ]


def search_records(
    query: str,
    objects: list[str],
    limit: int = 25,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Search records in one Attio workspace."""
    del config, usage_recorder
    if not isinstance(objects, list) or not objects:
        raise ValueError("Attio objects must be a non-empty list.")
    return _call_attio_api(
        "POST",
        "/objects/records/search",
        credentials,
        json_body={"query": query, "objects": objects, "limit": limit},
    )


def list_meetings(
    limit: int = 50,
    cursor: str | None = None,
    linked_object: str | None = None,
    linked_record_id: str | None = None,
    participants: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """List meetings in one Attio workspace."""
    del config, usage_recorder
    return _call_attio_api(
        "GET",
        "/meetings",
        credentials,
        params=_params(
            limit=limit,
            cursor=cursor,
            linked_object=linked_object,
            linked_record_id=linked_record_id,
            participants=participants,
        ),
    )


def list_call_recordings(
    meeting_id: str,
    limit: int = 50,
    cursor: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """List call recordings for one Attio meeting."""
    del config, usage_recorder
    return _call_attio_api(
        "GET",
        f"/meetings/{_path_segment(meeting_id, 'meeting_id')}/call_recordings",
        credentials,
        params=_params(limit=limit, cursor=cursor),
    )


def get_call_transcript(
    meeting_id: str,
    call_recording_id: str,
    cursor: str | None = None,
    *,
    credentials: JSONObject,
    config: JSONObject,
    usage_recorder: TaskUsageRecorder,
) -> JSONObject:
    """Read one page of transcript segments for a call recording."""
    del config, usage_recorder
    path = (
        f"/meetings/{_path_segment(meeting_id, 'meeting_id')}"
        f"/call_recordings/{_path_segment(call_recording_id, 'call_recording_id')}"
        "/transcript"
    )
    return _call_attio_api("GET", path, credentials, params=_params(cursor=cursor))


ATTIO_TOOL_HANDLERS: dict[str, Callable[..., JSONValue]] = {
    ATTIO_SEARCH_RECORDS_TOOL: search_records,
    ATTIO_LIST_MEETINGS_TOOL: list_meetings,
    ATTIO_LIST_CALL_RECORDINGS_TOOL: list_call_recordings,
    ATTIO_GET_CALL_TRANSCRIPT_TOOL: get_call_transcript,
}


def _tool(
    name: str,
    description: str,
    properties: JSONObject,
    *,
    required: tuple[str, ...] = (),
) -> JSONObject:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


def _call_attio_api(
    method: str,
    path: str,
    credentials: JSONObject,
    *,
    params: dict[str, str | int | bool] | None = None,
    json_body: JSONObject | None = None,
) -> JSONObject:
    """Call one Attio REST endpoint without exposing credentials."""
    access_token = credentials.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise AttioApiError("invalid_credentials")
    try:
        response = requests.request(
            method,
            f"{ATTIO_API_BASE_URL}{path}",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            params=params,
            json=json_body,
            timeout=_REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        raise AttioApiError("request_failed") from None
    if response.status_code >= 400:
        error_code = "rate_limited" if response.status_code == 429 else "http_error"
        raise AttioApiError(error_code, status_code=response.status_code)
    try:
        payload = response.json()
    except ValueError:
        raise AttioApiError("invalid_response") from None
    if not isinstance(payload, dict):
        raise AttioApiError("invalid_response")
    return cast(JSONObject, payload)


def _params(**values: str | int | bool | None) -> dict[str, str | int | bool]:
    return {key: value for key, value in values.items() if value is not None}


def _path_segment(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Attio {name} must be a non-empty string.")
    return quote(value.strip(), safe="")
